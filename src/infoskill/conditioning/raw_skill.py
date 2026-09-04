from __future__ import annotations

from typing import Protocol

from infoskill.domain.state import CanonicalAgentState, StateViews, render_policy_message
from infoskill.skills import RetrievalResult

from .contracts import ConditionedPolicyInput, ConditioningContext


class EpisodeRetriever(Protocol):
    def retrieve(self, query: str) -> RetrievalResult: ...


class RawSkillPromptConditioner:
    def __init__(self, retriever: EpisodeRetriever, *, history_length: int = 2) -> None:
        self._retriever = retriever
        self._history_length = history_length

    def prepare_group(self, initial_state: CanonicalAgentState) -> ConditioningContext:
        retrieval = self._retriever.retrieve(initial_state.goal)
        return ConditioningContext(candidate_skill_ids=retrieval.skill_ids, retrieval=retrieval)

    def condition_batch(
        self,
        states: tuple[CanonicalAgentState, ...],
        views: tuple[StateViews, ...],
        context: ConditioningContext,
    ) -> tuple[ConditionedPolicyInput, ...]:
        if len(states) != len(views):
            raise ValueError("states and views must have equal length")
        if context.retrieval is None:
            raise ValueError("raw skill conditioning requires an episode retrieval result")
        skill_block = format_raw_skill_block(context.retrieval)
        return tuple(
            ConditionedPolicyInput(
                user_message=render_policy_message(
                    state,
                    history_limit=self._history_length,
                    retrieved_skill_block=skill_block,
                ),
                candidate_skill_ids=context.candidate_skill_ids,
            )
            for state in states
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
