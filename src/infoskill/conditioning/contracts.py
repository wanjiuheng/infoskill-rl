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
class ConditionedPolicyInput:
    user_message: str
    candidate_skill_ids: tuple[str, ...]
    soft_prefix: object | None = None
    conditioning_trace: object | None = None


class SkillConditioner(Protocol):
    def prepare_group(self, initial_state: CanonicalAgentState) -> ConditioningContext: ...

    def condition_batch(
        self,
        states: tuple[CanonicalAgentState, ...],
        views: tuple[StateViews, ...],
        context: ConditioningContext,
    ) -> tuple[ConditionedPolicyInput, ...]: ...
