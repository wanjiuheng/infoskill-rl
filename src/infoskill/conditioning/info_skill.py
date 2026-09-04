from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from infoskill.domain.state import CanonicalAgentState, StateViews
from infoskill.models import InfoSkillCompressor, LatentProjector
from infoskill.semantic import FrozenSemanticEncoder, SemanticFeatureCache

from .contracts import ConditionedPolicyInput, ConditioningContext
from .raw_skill import EpisodeRetriever


@dataclass(frozen=True)
class InfoSkillTrace:
    state_summary: object
    posterior_mu: object
    posterior_logvar: object
    latent: object
    epsilon: object


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
        states: tuple[CanonicalAgentState, ...],
        views: tuple[StateViews, ...],
        context: ConditioningContext,
    ) -> tuple[ConditionedPolicyInput, ...]:
        if len(states) != len(views) or not states:
            raise ValueError("states and views must be non-empty and have equal length")
        if context.retrieval is None:
            raise ValueError("INFO-SKILL conditioning requires episode-level retrieval")
        records = tuple(item.record for item in context.retrieval.skills)
        state_features = self._semantic_encoder.encode_tokens(
            [view.compression_view for view in views]
        )
        skill_tokens, skill_valid, kind_ids = self._feature_cache.skill_batch(
            records, batch_size=len(states)
        )
        output = self._compressor(
            state_tokens=state_features.tokens,
            state_valid=state_features.valid,
            skill_tokens=skill_tokens,
            skill_valid=skill_valid,
            skill_kind_ids=kind_ids,
            latent_mode=self._latent_mode,
        )
        prefix = self._projector(output.latent)
        return tuple(
            ConditionedPolicyInput(
                user_message=view.policy_view,
                candidate_skill_ids=context.candidate_skill_ids,
                soft_prefix=prefix[index],
                conditioning_trace=InfoSkillTrace(
                    state_summary=output.state_summary[index],
                    posterior_mu=output.posterior_mu[index],
                    posterior_logvar=output.posterior_logvar[index],
                    latent=output.latent[index],
                    epsilon=output.epsilon[index],
                ),
            )
            for index, view in enumerate(views)
        )
