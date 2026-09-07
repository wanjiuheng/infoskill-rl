from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from infoskill.domain.state import CanonicalAgentState, StateViews
from infoskill.skills import RetrievalResult


@dataclass(frozen=True, slots=True)
class ConditioningContext:
    candidate_skill_ids: tuple[str, ...] = ()
    retrieval: RetrievalResult | None = None


@dataclass(frozen=True, slots=True)
class ConditioningRequest:
    state: CanonicalAgentState
    views: StateViews
    rollout_id: int
    global_update: int
    latent_seed: int

    def __post_init__(self) -> None:
        if self.rollout_id < 0 or self.global_update < 0 or self.latent_seed < 0:
            raise ValueError("conditioning identities and seed must be non-negative")


@dataclass(frozen=True, slots=True)
class ConditionedPolicyInput:
    user_message: str
    candidate_skill_ids: tuple[str, ...]
    soft_prefix: object | None = None
    conditioning_trace: object | None = None


class SkillConditioner(Protocol):
    def prepare_group(self, initial_state: CanonicalAgentState) -> ConditioningContext: ...

    def condition_batch(
        self,
        requests: tuple[ConditioningRequest, ...],
        context: ConditioningContext,
    ) -> tuple[ConditionedPolicyInput, ...]: ...
