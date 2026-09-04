from __future__ import annotations

from infoskill.domain.state import CanonicalAgentState, StateViews

from .contracts import ConditionedPolicyInput, ConditioningContext


class NoSkillConditioner:
    """Keep the policy text identical while injecting no skill information."""

    def prepare_group(self, initial_state: CanonicalAgentState) -> ConditioningContext:
        return ConditioningContext()

    def condition_batch(
        self,
        states: tuple[CanonicalAgentState, ...],
        views: tuple[StateViews, ...],
        context: ConditioningContext,
    ) -> tuple[ConditionedPolicyInput, ...]:
        if len(states) != len(views):
            raise ValueError("states and views must have equal length")
        return tuple(
            ConditionedPolicyInput(user_message=view.policy_view, candidate_skill_ids=()) for view in views
        )
