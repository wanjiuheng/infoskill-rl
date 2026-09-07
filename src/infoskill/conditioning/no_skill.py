from __future__ import annotations

from infoskill.domain.state import CanonicalAgentState

from .contracts import ConditionedPolicyInput, ConditioningContext, ConditioningRequest


class NoSkillConditioner:
    """Keep the policy text identical while injecting no skill information."""

    def prepare_group(self, initial_state: CanonicalAgentState) -> ConditioningContext:
        return ConditioningContext()

    def condition_batch(
        self,
        requests: tuple[ConditioningRequest, ...],
        context: ConditioningContext,
    ) -> tuple[ConditionedPolicyInput, ...]:
        return tuple(
            ConditionedPolicyInput(
                user_message=request.views.policy_view,
                candidate_skill_ids=(),
            )
            for request in requests
        )
