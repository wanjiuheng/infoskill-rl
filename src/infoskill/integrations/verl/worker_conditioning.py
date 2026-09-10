from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from infoskill.conditioning import (
    InfoSkillConditioningResult,
    InfoSkillConditioningWorkItem,
    InfoSkillReplayTrace,
)
from infoskill.semantic import FrozenSemanticEncoder, SemanticFeatureCache
from infoskill.skills import FixedSkillLibrary


class InfoSkillWorkerConditioner:
    """Run deterministic replayable conditioning on one runtime worker."""

    def __init__(
        self,
        *,
        library: FixedSkillLibrary,
        semantic_encoder: FrozenSemanticEncoder,
        feature_cache: SemanticFeatureCache,
        compressor: nn.Module,
        projector: nn.Module,
    ) -> None:
        self.library = library
        self.semantic_encoder = semantic_encoder
        self.feature_cache = feature_cache
        self.compressor = compressor
        self.projector = projector

    @torch.no_grad()
    def condition(
        self,
        items: tuple[InfoSkillConditioningWorkItem, ...],
    ) -> tuple[InfoSkillConditioningResult, ...]:
        if not items:
            raise ValueError("worker conditioning batch must not be empty")
        modes = {item.latent_mode for item in items}
        if len(modes) != 1:
            raise ValueError("worker conditioning batch must use one latent mode")

        state_features = self.semantic_encoder.encode_tokens(
            [item.compression_view for item in items]
        )
        skill_groups = [
            tuple(
                self.library.get(skill_id)
                for skill_id in item.candidate_skill_ids
            )
            for item in items
        ]
        skill_tokens, skill_valid, skill_kind_ids = (
            self.feature_cache.heterogeneous_skill_batch(skill_groups)
        )
        compressor_module = _unwrap(self.compressor)
        latent_dim = getattr(compressor_module, "latent_dim", None)
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("INFO-SKILL compressor requires a positive latent_dim")

        latent_mode = next(iter(modes))
        if latent_mode == "sample":
            epsilon = torch.stack(
                [
                    torch.randn(
                        latent_dim,
                        generator=torch.Generator(device="cpu").manual_seed(
                            item.latent_seed
                        ),
                    )
                    for item in items
                ]
            ).to(device=state_features.tokens.device)
            compressor_mode = "replay"
        else:
            epsilon = None
            compressor_mode = "mean"
        output = self.compressor(
            state_tokens=state_features.tokens,
            state_valid=state_features.valid,
            skill_tokens=skill_tokens,
            skill_valid=skill_valid,
            skill_kind_ids=skill_kind_ids,
            latent_mode=compressor_mode,
            replay_epsilon=epsilon,
        )
        prefix = self.projector(output.latent)
        return tuple(
            InfoSkillConditioningResult(
                soft_prefix=_cpu_tensor(prefix[index]),
                replay_trace=InfoSkillReplayTrace(
                    latent_seed=item.latent_seed,
                    state_summary=_cpu_tensor(output.state_summary[index]),
                    state_tokens=_trim_state_tokens(
                        state_features.tokens,
                        state_features.valid,
                        index,
                    ),
                    posterior_mu=_cpu_tensor(output.posterior_mu[index]),
                    posterior_logvar=_cpu_tensor(
                        output.posterior_logvar[index]
                    ),
                    latent=_cpu_tensor(output.latent[index]),
                    epsilon=_cpu_tensor(output.epsilon[index]),
                ),
            )
            for index, item in enumerate(items)
        )


def _unwrap(module: nn.Module) -> nn.Module:
    return cast(nn.Module, getattr(module, "module", module))


def _trim_state_tokens(tokens: Tensor, valid: Tensor, index: int) -> Tensor:
    length = int(valid[index].sum().item())
    if length <= 0:
        raise ValueError("semantic encoder returned an empty state")
    return _cpu_tensor(tokens[index, :length])


def _cpu_tensor(value: Tensor) -> Tensor:
    return value.detach().to(device="cpu", copy=True)
