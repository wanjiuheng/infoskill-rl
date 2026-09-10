from __future__ import annotations

from typing import Literal

import torch

from infoskill.domain.state import CanonicalAgentState
from infoskill.models import InfoSkillCompressor, LatentProjector
from infoskill.semantic import FeatureBatch, FrozenSemanticEncoder, SemanticFeatureCache

from .contracts import (
    ConditionedPolicyInput,
    ConditioningContext,
    ConditioningRequest,
    InfoSkillReplayTrace,
)
from .raw_skill import EpisodeRetriever


class InfoSkillConditioner:
    def __init__(
        self,
        *,
        retriever: EpisodeRetriever,
        semantic_encoder: FrozenSemanticEncoder,
        feature_cache: SemanticFeatureCache,
        compressor: InfoSkillCompressor,
        projector: LatentProjector,
        latent_mode: Literal["sample", "mean"] = "sample",
    ) -> None:
        self._retriever = retriever
        self._semantic_encoder = semantic_encoder
        self._feature_cache = feature_cache
        self._compressor = compressor
        self._projector = projector
        self._latent_mode = latent_mode

    def prepare_group(self, initial_state: CanonicalAgentState) -> ConditioningContext:
        retrieval = self._retriever.retrieve(initial_state.goal)
        return ConditioningContext(candidate_skill_ids=retrieval.skill_ids, retrieval=retrieval)

    def condition_batch(
        self,
        requests: tuple[ConditioningRequest, ...],
        context: ConditioningContext,
    ) -> tuple[ConditionedPolicyInput, ...]:
        if not requests:
            raise ValueError("conditioning requests must not be empty")
        if context.retrieval is None:
            raise ValueError("INFO-SKILL conditioning requires episode-level retrieval")
        records = tuple(item.record for item in context.retrieval.skills)
        state_features = self._semantic_encoder.encode_tokens(
            [request.views.compression_view for request in requests]
        )
        skill_tokens, skill_valid, kind_ids = self._feature_cache.skill_batch(
            records, batch_size=len(requests)
        )
        with torch.no_grad():
            if self._latent_mode == "sample":
                epsilon = torch.stack(
                    [
                        torch.randn(
                            self._compressor.latent_dim,
                            generator=torch.Generator(device="cpu").manual_seed(
                                request.latent_seed
                            ),
                        )
                        for request in requests
                    ]
                ).to(device=state_features.tokens.device)
                latent_mode = "replay"
            else:
                epsilon = None
                latent_mode = "mean"
            output = self._compressor(
                state_tokens=state_features.tokens,
                state_valid=state_features.valid,
                skill_tokens=skill_tokens,
                skill_valid=skill_valid,
                skill_kind_ids=kind_ids,
                latent_mode=latent_mode,
                replay_epsilon=epsilon,
            )
            prefix = self._projector(output.latent)
        return tuple(
            ConditionedPolicyInput(
                user_message=request.views.policy_view,
                candidate_skill_ids=context.candidate_skill_ids,
                soft_prefix=prefix[index].detach().clone(),
                conditioning_trace=InfoSkillReplayTrace(
                    latent_seed=request.latent_seed,
                    state_summary=_cpu_replay_tensor(output.state_summary[index]),
                    state_tokens=_trim_state_tokens(state_features, index),
                    posterior_mu=_cpu_replay_tensor(output.posterior_mu[index]),
                    posterior_logvar=_cpu_replay_tensor(
                        output.posterior_logvar[index]
                    ),
                    latent=_cpu_replay_tensor(output.latent[index]),
                    epsilon=_cpu_replay_tensor(output.epsilon[index]),
                ),
            )
            for index, request in enumerate(requests)
        )


def _trim_state_tokens(features: FeatureBatch, index: int):
    length = int(features.valid[index].sum().item())
    return _cpu_replay_tensor(features.tokens[index, :length])


def _cpu_replay_tensor(value: torch.Tensor) -> torch.Tensor:
    """Detach rollout replay data from both autograd and scarce GPU storage."""

    return value.detach().to(device="cpu", copy=True)
