from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn

from infoskill.models import (
    ExecutableGroundingHead,
    FidelityPredictor,
    InfoSkillCompressor,
    StateConditionedPrior,
    gaussian_kl,
)

from .losses import auxiliary_loss


@dataclass(frozen=True)
class CompressionReplayBatch:
    state_tokens: Tensor
    state_valid: Tensor
    skill_tokens: Tensor
    skill_valid: Tensor
    skill_kind_ids: Tensor
    epsilon: Tensor

    def validate(self, *, name: str) -> None:
        if self.state_tokens.ndim != 3:
            raise ValueError(f"{name}.state_tokens must be [batch, tokens, width]")
        batch = self.state_tokens.shape[0]
        if batch == 0:
            raise ValueError(f"{name} cannot be empty")
        expected = {
            "state_valid": (batch, self.state_tokens.shape[1]),
            "skill_valid": self.skill_tokens.shape[:-1],
            "skill_kind_ids": self.skill_tokens.shape[:2],
        }
        actual = {
            "state_valid": tuple(self.state_valid.shape),
            "skill_valid": tuple(self.skill_valid.shape),
            "skill_kind_ids": tuple(self.skill_kind_ids.shape),
        }
        if self.skill_tokens.ndim != 4:
            raise ValueError(
                f"{name}.skill_tokens must be [batch, candidates, tokens, width]"
            )
        for field, shape in expected.items():
            if actual[field] != tuple(shape):
                raise ValueError(f"{name}.{field} has shape {actual[field]}, expected {shape}")
        if self.epsilon.ndim != 2 or self.epsilon.shape[0] != batch:
            raise ValueError(f"{name}.epsilon must be [batch, latent_dim]")
        if not self.state_valid.bool().any(dim=1).all():
            raise ValueError(f"{name} requires at least one valid state token per sample")
        if not self.skill_valid.bool().flatten(1).any(dim=1).all():
            raise ValueError(f"{name} requires at least one valid skill token per sample")


@dataclass(frozen=True)
class OnlineAuxiliaryBatch:
    replay: CompressionReplayBatch
    trajectory_index: Tensor
    fidelity_target: Tensor

    def validate(self) -> None:
        self.replay.validate(name="online")
        batch = self.replay.state_tokens.shape[0]
        if tuple(self.trajectory_index.shape) != (batch,):
            raise ValueError("online.trajectory_index must have one value per step")
        if tuple(self.fidelity_target.shape) != (batch,):
            raise ValueError("online.fidelity_target must have one value per step")
        if self.trajectory_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("online.trajectory_index must use an integer dtype")


@dataclass(frozen=True)
class OfflineGroundingBatch:
    replay: CompressionReplayBatch
    command_embeddings: Tensor
    command_valid: Tensor
    grounding_target: Tensor

    def validate(self) -> None:
        self.replay.validate(name="offline")
        batch = self.replay.state_tokens.shape[0]
        if self.command_embeddings.ndim != 3:
            raise ValueError(
                "offline.command_embeddings must be [batch, commands, width]"
            )
        if self.command_embeddings.shape[0] != batch:
            raise ValueError("offline command batch must match replay batch")
        if tuple(self.command_valid.shape) != tuple(self.command_embeddings.shape[:2]):
            raise ValueError("offline.command_valid must match command embeddings")
        if tuple(self.grounding_target.shape) != (batch,):
            raise ValueError("offline.grounding_target must have one value per sample")
        if self.grounding_target.dtype not in (torch.int32, torch.int64):
            raise ValueError("offline.grounding_target must use an integer dtype")
        command_count = self.command_embeddings.shape[1]
        if ((self.grounding_target < 0) | (self.grounding_target >= command_count)).any():
            raise ValueError("offline.grounding_target is outside the command dimension")
        rows = torch.arange(batch, device=self.grounding_target.device)
        if not self.command_valid[rows, self.grounding_target].bool().all():
            raise ValueError("offline.grounding_target must point to a valid command")


@dataclass(frozen=True)
class AuxiliaryTrainingBatch:
    online: OnlineAuxiliaryBatch
    offline: OfflineGroundingBatch

    def validate(self) -> None:
        self.online.validate()
        self.offline.validate()
        online_width = self.online.replay.state_tokens.shape[-1]
        offline_width = self.offline.replay.state_tokens.shape[-1]
        if online_width != offline_width:
            raise ValueError("online and offline semantic widths must match")
        if self.online.replay.epsilon.shape[-1] != self.offline.replay.epsilon.shape[-1]:
            raise ValueError("online and offline latent dimensions must match")


@dataclass(frozen=True, slots=True)
class AuxiliaryUpdateConfig:
    fidelity_weight: float = 1.0
    rate_weight: float = 0.001
    grounding_weight: float = 0.1
    max_grad_norm: float = 1.0

    def validate(self) -> None:
        if min(self.fidelity_weight, self.rate_weight, self.grounding_weight) < 0:
            raise ValueError("auxiliary loss weights must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


class AuxiliaryUpdater:
    """Own one complete compressor-optimizer update behind a small tensor interface."""

    def __init__(
        self,
        *,
        compressor: InfoSkillCompressor,
        prior: StateConditionedPrior,
        fidelity: FidelityPredictor,
        grounding: ExecutableGroundingHead,
        optimizer: torch.optim.Optimizer,
        config: AuxiliaryUpdateConfig = AuxiliaryUpdateConfig(),
    ) -> None:
        config.validate()
        self.compressor = compressor
        self.prior = prior
        self.fidelity = fidelity
        self.grounding = grounding
        self.optimizer = optimizer
        self.config = config
        self._parameters = _unique_trainable_parameters(
            (compressor, prior, fidelity, grounding)
        )
        if not self._parameters:
            raise ValueError("auxiliary updater requires trainable parameters")
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        expected = {id(parameter) for parameter in self._parameters}
        if optimized != expected:
            raise ValueError(
                "optimizer parameters must exactly match the auxiliary modules"
            )

    def update(self, batch: AuxiliaryTrainingBatch) -> Mapping[str, float]:
        batch.validate()
        self.optimizer.zero_grad(set_to_none=True)

        online = _replay(self.compressor, batch.online.replay)
        online_prior_mu, online_prior_logvar = self.prior(online.state_summary)
        online_rate = gaussian_kl(
            online.posterior_mu,
            online.posterior_logvar,
            online_prior_mu,
            online_prior_logvar,
        )
        fidelity_prediction = self.fidelity(online.state_summary, online.latent)

        offline = _replay(self.compressor, batch.offline.replay)
        offline_prior_mu, offline_prior_logvar = self.prior(offline.state_summary)
        offline_rate = gaussian_kl(
            offline.posterior_mu,
            offline.posterior_logvar,
            offline_prior_mu,
            offline_prior_logvar,
        )
        grounding_logits = self.grounding(
            offline.state_summary,
            offline.latent,
            batch.offline.command_embeddings,
            batch.offline.command_valid,
        )
        losses = auxiliary_loss(
            fidelity_prediction=fidelity_prediction,
            fidelity_target=batch.online.fidelity_target,
            online_rate_by_step=online_rate,
            trajectory_index=batch.online.trajectory_index,
            grounding_logits=grounding_logits,
            grounding_targets=batch.offline.grounding_target,
            offline_rate=offline_rate,
            fidelity_weight=self.config.fidelity_weight,
            rate_weight=self.config.rate_weight,
            grounding_weight=self.config.grounding_weight,
        )
        if not torch.isfinite(losses.total):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite auxiliary loss")

        losses.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._parameters,
            self.config.max_grad_norm,
            error_if_nonfinite=False,
        )
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite auxiliary gradient norm")
        self.optimizer.step()

        correlation, correlation_valid = _correlation(
            fidelity_prediction.detach(), batch.online.fidelity_target.detach()
        )
        grounding_accuracy = (
            grounding_logits.detach().argmax(dim=-1)
            == batch.offline.grounding_target
        ).float().mean()
        latent_dim = batch.online.replay.epsilon.shape[-1]
        return {
            "aux/loss": float(losses.total.detach().item()),
            "aux/fidelity_loss": float(losses.fidelity.detach().item()),
            "aux/rate_loss": float(losses.rate.detach().item()),
            "aux/grounding_loss": float(losses.grounding.detach().item()),
            "aux/fidelity_weighted": float(
                (self.config.fidelity_weight * losses.fidelity).detach().item()
            ),
            "aux/rate_weighted": float(
                (self.config.rate_weight * losses.rate).detach().item()
            ),
            "aux/grounding_weighted": float(
                (self.config.grounding_weight * losses.grounding).detach().item()
            ),
            "aux/rate_per_latent_dim": float(
                (losses.rate.detach() / latent_dim).item()
            ),
            "aux/grad_norm_before_clip": float(grad_norm.detach().item()),
            "aux/optimizer_step_applied": 1.0,
            "aux/online_step_count": float(batch.online.fidelity_target.numel()),
            "aux/online_trajectory_count": float(
                torch.unique(batch.online.trajectory_index).numel()
            ),
            "aux/offline_sample_count": float(batch.offline.grounding_target.numel()),
            "aux/fidelity_correlation": correlation,
            "aux/fidelity_correlation_valid": float(correlation_valid),
            "aux/grounding_top1_accuracy": float(grounding_accuracy.item()),
        }


def _replay(compressor: InfoSkillCompressor, batch: CompressionReplayBatch):
    return compressor(
        state_tokens=batch.state_tokens,
        state_valid=batch.state_valid,
        skill_tokens=batch.skill_tokens,
        skill_valid=batch.skill_valid,
        skill_kind_ids=batch.skill_kind_ids,
        latent_mode="replay",
        replay_epsilon=batch.epsilon,
    )


def _unique_trainable_parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if parameter.requires_grad and identity not in seen:
                seen.add(identity)
                parameters.append(parameter)
    return parameters


def _correlation(prediction: Tensor, target: Tensor) -> tuple[float, bool]:
    if prediction.numel() < 2:
        return 0.0, False
    prediction = prediction.float()
    target = target.float()
    prediction_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    denominator = prediction_centered.norm() * target_centered.norm()
    if denominator.item() == 0.0:
        return 0.0, False
    value = torch.dot(prediction_centered, target_centered) / denominator
    result = float(value.item())
    return (result, math.isfinite(result))
