from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Mapping

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F

from infoskill.learning import (
    AuxiliaryUpdateConfig,
    OfflineGroundingBatch,
    OnlineAuxiliaryBatch,
)
from infoskill.models import gaussian_kl


class DistributedAuxiliaryUpdater:
    """Accumulate one globally normalized M1 auxiliary optimizer step."""

    def __init__(
        self,
        *,
        compressor: nn.Module,
        prior: nn.Module,
        fidelity: nn.Module,
        grounding: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        world_size: int,
        config: AuxiliaryUpdateConfig = AuxiliaryUpdateConfig(),
        all_reduce: Callable[[Tensor], None] | None = None,
    ) -> None:
        config.validate()
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.compressor = compressor
        self.prior = prior
        self.fidelity = fidelity
        self.grounding = grounding
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.world_size = world_size
        self.config = config
        self._all_reduce = all_reduce or _distributed_sum
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
        if optimized != {id(parameter) for parameter in self._parameters}:
            raise ValueError(
                "optimizer parameters must exactly match the auxiliary modules"
            )
        if scheduler.optimizer is not optimizer:
            raise ValueError("auxiliary scheduler must own the auxiliary optimizer")

    def update(
        self,
        *,
        online_batches: Iterable[OnlineAuxiliaryBatch],
        offline_batches: Iterable[OfflineGroundingBatch],
        global_trajectory_count: int,
        global_offline_count: int,
    ) -> Mapping[str, float]:
        if global_trajectory_count <= 0 or global_offline_count <= 0:
            raise ValueError("global auxiliary denominators must be positive")
        device = self._parameters[0].device
        statistics = torch.zeros(13, device=device, dtype=torch.float64)
        self.optimizer.zero_grad(set_to_none=True)

        online_calls = 0
        latent_dim: int | None = None
        online_scale = self.world_size / global_trajectory_count
        for batch in online_batches:
            batch.validate()
            if batch.step_weight is None:
                raise ValueError("distributed online batch requires step weights")
            batch_latent_dim = int(batch.replay.epsilon.shape[-1])
            if latent_dim is None:
                latent_dim = batch_latent_dim
            elif latent_dim != batch_latent_dim:
                raise ValueError("online auxiliary latent dimensions must match")
            outputs = _replay(self.compressor, batch.replay)
            prior_mu, prior_logvar = self.prior(outputs.state_summary)
            rate = gaussian_kl(
                outputs.posterior_mu,
                outputs.posterior_logvar,
                prior_mu,
                prior_logvar,
            )
            prediction = self.fidelity(outputs.state_summary, outputs.latent)
            weights = batch.step_weight.to(prediction.dtype)
            errors = (prediction - batch.fidelity_target.detach()).square()
            loss = online_scale * (
                self.config.fidelity_weight * (errors * weights).sum()
                + self.config.rate_weight * 0.5 * (rate * weights).sum()
            )
            loss.backward()
            _accumulate_online_statistics(
                statistics,
                prediction.detach(),
                batch.fidelity_target.detach(),
                errors.detach(),
                rate.detach(),
                weights.detach(),
            )
            online_calls += 1

        offline_calls = 0
        offline_scale = self.world_size / global_offline_count
        for batch in offline_batches:
            batch.validate()
            if batch.sample_weight is None:
                raise ValueError("distributed offline batch requires sample weights")
            outputs = _replay(self.compressor, batch.replay)
            prior_mu, prior_logvar = self.prior(outputs.state_summary)
            rate = gaussian_kl(
                outputs.posterior_mu,
                outputs.posterior_logvar,
                prior_mu,
                prior_logvar,
            )
            logits = self.grounding(
                outputs.state_summary,
                outputs.latent,
                batch.command_embeddings,
                batch.command_valid,
            )
            weights = batch.sample_weight.to(logits.dtype)
            grounding_loss = F.cross_entropy(
                logits,
                batch.grounding_target,
                reduction="none",
            )
            loss = offline_scale * (
                self.config.grounding_weight * (grounding_loss * weights).sum()
                + self.config.rate_weight * 0.5 * (rate * weights).sum()
            )
            loss.backward()
            statistics[6] += (grounding_loss.detach() * weights).sum().double()
            statistics[7] += (rate.detach() * weights).sum().double()
            statistics[8] += (
                (logits.detach().argmax(dim=-1) == batch.grounding_target)
                .to(weights.dtype)
                .mul(weights)
                .sum()
                .double()
            )
            statistics[9] += weights.sum().double()
            offline_calls += 1

        if online_calls == 0 or offline_calls == 0 or latent_dim is None:
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError("distributed auxiliary update requires both batch streams")

        self._all_reduce(statistics)
        if not math.isclose(
            float(statistics[12].item()),
            float(global_trajectory_count),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ) or not math.isclose(
            float(statistics[9].item()),
            float(global_offline_count),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("global auxiliary eligibility weights are inconsistent")

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._parameters,
            self.config.max_grad_norm,
            error_if_nonfinite=False,
        )
        finite = torch.tensor(
            float(torch.isfinite(grad_norm).item()),
            device=device,
            dtype=torch.float32,
        )
        self._all_reduce(finite)
        optimizer_step_applied = finite.item() == float(self.world_size)
        if optimizer_step_applied:
            self.optimizer.step()
            self.scheduler.step()
        else:
            self.optimizer.zero_grad(set_to_none=True)

        return _metrics(
            statistics,
            global_trajectory_count=global_trajectory_count,
            global_offline_count=global_offline_count,
            latent_dim=latent_dim,
            grad_norm=grad_norm,
            optimizer_step_applied=optimizer_step_applied,
            config=self.config,
        )


def _replay(compressor: nn.Module, batch):
    return compressor(
        state_tokens=batch.state_tokens,
        state_valid=batch.state_valid,
        skill_tokens=batch.skill_tokens,
        skill_valid=batch.skill_valid,
        skill_kind_ids=batch.skill_kind_ids,
        latent_mode="replay",
        replay_epsilon=batch.epsilon,
    )


def _accumulate_online_statistics(
    statistics: Tensor,
    prediction: Tensor,
    target: Tensor,
    errors: Tensor,
    rate: Tensor,
    weights: Tensor,
) -> None:
    eligible = weights > 0
    statistics[0] += (errors * weights).sum().double()
    statistics[1] += (rate * weights).sum().double()
    statistics[12] += weights.sum().double()
    statistics[2] += prediction[eligible].double().sum()
    statistics[3] += target[eligible].double().sum()
    statistics[4] += prediction[eligible].double().square().sum()
    statistics[5] += target[eligible].double().square().sum()
    statistics[10] += (
        prediction[eligible].double() * target[eligible].double()
    ).sum()
    statistics[11] += eligible.sum().double()


def _metrics(
    statistics: Tensor,
    *,
    global_trajectory_count: int,
    global_offline_count: int,
    latent_dim: int,
    grad_norm: Tensor,
    optimizer_step_applied: bool,
    config: AuxiliaryUpdateConfig,
) -> dict[str, float]:
    fidelity = statistics[0] / global_trajectory_count
    online_rate = statistics[1] / global_trajectory_count
    grounding = statistics[6] / global_offline_count
    offline_rate = statistics[7] / global_offline_count
    rate = 0.5 * (online_rate + offline_rate)
    total = (
        config.fidelity_weight * fidelity
        + config.rate_weight * rate
        + config.grounding_weight * grounding
    )
    correlation, correlation_valid = _correlation_from_sums(statistics)
    return {
        "aux/loss": float(total.item()),
        "aux/fidelity_loss": float(fidelity.item()),
        "aux/rate_loss": float(rate.item()),
        "aux/grounding_loss": float(grounding.item()),
        "aux/fidelity_weighted": float((config.fidelity_weight * fidelity).item()),
        "aux/rate_weighted": float((config.rate_weight * rate).item()),
        "aux/grounding_weighted": float(
            (config.grounding_weight * grounding).item()
        ),
        "aux/rate_per_latent_dim": float((rate / latent_dim).item()),
        "aux/grad_norm_before_clip": float(grad_norm.detach().item()),
        "aux/optimizer_step_applied": float(optimizer_step_applied),
        "aux/online_step_count": float(statistics[11].item()),
        "aux/online_trajectory_count": float(global_trajectory_count),
        "aux/offline_sample_count": float(statistics[9].item()),
        "aux/fidelity_correlation": correlation,
        "aux/fidelity_correlation_valid": float(correlation_valid),
        "aux/grounding_top1_accuracy": float(
            (statistics[8] / global_offline_count).item()
        ),
    }


def _correlation_from_sums(statistics: Tensor) -> tuple[float, bool]:
    count = statistics[11]
    if count.item() < 2:
        return 0.0, False
    covariance = statistics[10] - statistics[2] * statistics[3] / count
    prediction_variance = statistics[4] - statistics[2].square() / count
    target_variance = statistics[5] - statistics[3].square() / count
    denominator = (prediction_variance * target_variance).clamp_min(0).sqrt()
    if denominator.item() == 0.0:
        return 0.0, False
    value = float((covariance / denominator).item())
    return value, math.isfinite(value)


def _unique_trainable_parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def _distributed_sum(value: Tensor) -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed auxiliary update requires a process group")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
