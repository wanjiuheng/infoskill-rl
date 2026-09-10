from __future__ import annotations

from typing import Literal, Mapping

from infoskill.domain.state import CanonicalAgentState

from .contracts import (
    ConditionedPolicyInput,
    ConditioningContext,
    ConditioningRequest,
    InfoSkillConditioningRuntime,
)
from .raw_skill import EpisodeRetriever


class RuntimeInfoSkillConditioner:
    """Delegate all trainable INFO-SKILL conditioning to runtime workers."""

    def __init__(
        self,
        *,
        retriever: EpisodeRetriever,
        runtime: InfoSkillConditioningRuntime,
        latent_mode: Literal["sample", "mean"] = "sample",
        query_by_task_id: Mapping[str, str] | None = None,
    ) -> None:
        self._retriever = retriever
        self._runtime = runtime
        self._latent_mode = latent_mode
        self._query_by_task_id = (
            None if query_by_task_id is None else dict(query_by_task_id)
        )

    def prepare_group(self, initial_state: CanonicalAgentState) -> ConditioningContext:
        query = initial_state.goal
        if self._query_by_task_id is not None:
            try:
                query = self._query_by_task_id[initial_state.task_id]
            except KeyError as error:
                raise KeyError(
                    "INFO-SKILL retrieval has no registered query for task "
                    f"{initial_state.task_id!r}"
                ) from error
        retrieval = self._retriever.retrieve(query)
        if not retrieval.skill_ids:
            raise ValueError("INFO-SKILL retrieval requires candidate skills")
        return ConditioningContext(
            candidate_skill_ids=retrieval.skill_ids,
            retrieval=retrieval,
        )

    def condition_batch(
        self,
        requests: tuple[ConditioningRequest, ...],
        context: ConditioningContext,
    ) -> tuple[ConditionedPolicyInput, ...]:
        if not requests:
            raise ValueError("conditioning requests must not be empty")
        if context.retrieval is None or not context.candidate_skill_ids:
            raise ValueError("INFO-SKILL conditioning requires episode retrieval")
        outputs = self._runtime.condition_infoskill(
            requests,
            context.candidate_skill_ids,
            latent_mode=self._latent_mode,
        )
        if len(outputs) != len(requests):
            raise RuntimeError("INFO-SKILL runtime must return one output per request")
        return tuple(
            ConditionedPolicyInput(
                user_message=request.views.policy_view,
                candidate_skill_ids=context.candidate_skill_ids,
                soft_prefix=output.soft_prefix,
                conditioning_trace=output.replay_trace,
            )
            for request, output in zip(requests, outputs)
        )
