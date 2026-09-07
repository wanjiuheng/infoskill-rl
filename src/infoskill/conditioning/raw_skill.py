from __future__ import annotations

from typing import Mapping, Protocol

from infoskill.domain.state import CanonicalAgentState, render_policy_message
from infoskill.skills import RetrievalResult

from .contracts import ConditionedPolicyInput, ConditioningContext, ConditioningRequest


class EpisodeRetriever(Protocol):
    def retrieve(self, query: str) -> RetrievalResult: ...


class RawSkillPromptConditioner:
    def __init__(
        self,
        retriever: EpisodeRetriever,
        *,
        history_length: int = 2,
        query_by_task_id: Mapping[str, str] | None = None,
    ) -> None:
        self._retriever = retriever
        self._history_length = history_length
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
                    "raw skill retrieval has no registered query for task "
                    f"{initial_state.task_id!r}"
                ) from error
        retrieval = self._retriever.retrieve(query)
        return ConditioningContext(candidate_skill_ids=retrieval.skill_ids, retrieval=retrieval)

    def condition_batch(
        self,
        requests: tuple[ConditioningRequest, ...],
        context: ConditioningContext,
    ) -> tuple[ConditionedPolicyInput, ...]:
        if context.retrieval is None:
            raise ValueError("raw skill conditioning requires an episode retrieval result")
        skill_block = format_raw_skill_block(context.retrieval)
        return tuple(
            ConditionedPolicyInput(
                user_message=render_policy_message(
                    state,
                    history_limit=(
                        request.history_limit
                        if request.history_limit is not None
                        else self._history_length
                    ),
                    retrieved_skill_block=skill_block,
                ),
                candidate_skill_ids=context.candidate_skill_ids,
            )
            for request, state in ((item, item.state) for item in requests)
        )


def format_raw_skill_block(retrieval: RetrievalResult) -> str:
    blocks: list[str] = []
    for item in retrieval.skills:
        record = item.record
        header = f"[{record.skill_id}] type={record.kind}"
        if record.category:
            header += f" category={record.category}"
        fields = [
            f"{name}: {value}"
            for name, value in record.fields.items()
            if name not in {"skill_id", "mistake_id"}
        ]
        blocks.append("\n".join([header, *fields]))
    return "\n\n".join(blocks)
