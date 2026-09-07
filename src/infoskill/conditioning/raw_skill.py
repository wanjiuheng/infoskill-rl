from __future__ import annotations

from typing import Literal, Mapping, Protocol

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
        prompt_format: Literal["full", "skillrl"] = "full",
    ) -> None:
        if prompt_format not in {"full", "skillrl"}:
            raise ValueError("raw skill prompt format must be full or skillrl")
        self._retriever = retriever
        self._history_length = history_length
        self._prompt_format = prompt_format
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
        skill_block = format_raw_skill_block(
            context.retrieval,
            style=self._prompt_format,
        )
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


def format_raw_skill_block(
    retrieval: RetrievalResult,
    *,
    style: Literal["full", "skillrl"] = "full",
) -> str:
    if style == "skillrl":
        return _format_skillrl_block(retrieval)
    if style != "full":
        raise ValueError("raw skill prompt format must be full or skillrl")
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


def _format_skillrl_block(retrieval: RetrievalResult) -> str:
    general: list[str] = []
    task_specific: list[str] = []
    mistakes: list[str] = []
    task_categories: list[str] = []
    for item in retrieval.skills:
        record = item.record
        fields = record.fields
        if record.kind == "general":
            general.append(
                f"- **{fields.get('title', '')}**: {fields.get('principle', '')}"
            )
        elif record.kind == "task_specific":
            line = f"- **{fields.get('title', '')}**: {fields.get('principle', '')}"
            when = fields.get("when_to_apply", "")
            if when:
                line += f"\n  _Apply when: {when}_"
            task_specific.append(line)
            if record.category:
                task_categories.append(record.category)
        elif record.kind == "common_mistake":
            description = fields.get("description", "")
            if not description:
                continue
            line = f"- **Don't**: {description}"
            fix = fields.get("how_to_avoid", "")
            if fix:
                line += f"\n  **Instead**: {fix}"
            mistakes.append(line)

    sections: list[str] = []
    if general:
        sections.append("### General Principles\n" + "\n".join(general))
    if task_specific:
        if retrieval.mode == "embedding":
            heading = "### Task-Relevant Skills"
        else:
            category = task_categories[0] if task_categories else "task relevant"
            heading = f"### {category.replace('_', ' ').title()} Skills"
        sections.append(heading + "\n" + "\n".join(task_specific))
    if mistakes:
        sections.append("### Mistakes to Avoid\n" + "\n".join(mistakes))
    return "\n\n".join(sections) if sections else "No relevant skills found for this task."
