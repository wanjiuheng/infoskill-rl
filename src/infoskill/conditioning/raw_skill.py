from __future__ import annotations

from typing import Literal, Mapping, Protocol, Sequence

from infoskill.domain.state import (
    AgentHistoryEntry,
    CanonicalAgentState,
    render_policy_message,
)
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


_SKILLRL_ALFWORLD_TEMPLATE_NO_HISTORY = (
    "\nYou are an expert agent operating in the ALFRED Embodied Environment.\n"
    "Your current observation is: {current_observation}\n"
    "Your admissible actions of the current situation are: "
    "[{admissible_actions}].\n\n"
    "Now it's your turn to take an action.\n"
    "You should first reason step-by-step about the current situation. This "
    "reasoning process MUST be enclosed within <think> </think> tags. \n"
    "Once you've finished your reasoning, you should choose an admissible "
    "action for current step and present it within <action> </action> tags.\n"
)

_SKILLRL_ALFWORLD_TEMPLATE_WITH_MEMORY = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

## Retrieved Relevant Experience

{retrieved_memories}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


class SkillRlGrpoPromptConditioner:
    """Reproduce the pinned SkillRL ALFWorld GRPO prompt for diagnosis only.

    SkillRL retrieves one template-selected skill set at reset, omits the skill
    block from its initial ``NO_HIS`` prompt, then inserts that same set in all
    subsequent prompts.  This intentionally does not implement INFO-SKILL's
    registered unified policy-prompt protocol.
    """

    def __init__(
        self,
        retriever: EpisodeRetriever,
        *,
        history_length: int = 2,
    ) -> None:
        if history_length < 0:
            raise ValueError("history_length must be non-negative")
        self._retriever = retriever
        self._history_length = history_length

    def prepare_group(self, initial_state: CanonicalAgentState) -> ConditioningContext:
        # The pinned SkillRL manager retrieves from the task string extracted
        # from ALFWorld's reset observation, represented here by state.goal.
        retrieval = self._retriever.retrieve(initial_state.goal)
        return ConditioningContext(
            candidate_skill_ids=retrieval.skill_ids,
            retrieval=retrieval,
        )

    def condition_batch(
        self,
        requests: tuple[ConditioningRequest, ...],
        context: ConditioningContext,
    ) -> tuple[ConditionedPolicyInput, ...]:
        if context.retrieval is None:
            raise ValueError("SkillRL GRPO prompt requires an episode retrieval result")
        skill_block = format_raw_skill_block(context.retrieval, style="skillrl")
        conditioned = []
        for request in requests:
            state = request.state
            history_limit = (
                request.history_limit
                if request.history_limit is not None
                else self._history_length
            )
            if state.step_index == 0:
                message = _render_skillrl_initial_message(state)
                skills_injected = False
            else:
                message = _render_skillrl_memory_message(
                    state,
                    skill_block=skill_block,
                    history_limit=history_limit,
                )
                skills_injected = True
            omitted = max(0, len(state.history) - history_limit)
            conditioned.append(
                ConditionedPolicyInput(
                    user_message=message,
                    candidate_skill_ids=context.candidate_skill_ids,
                    conditioning_trace={
                        "prompt_protocol": "skillrl-grpo-alfworld-v1",
                        "skills_injected": skills_injected,
                    },
                    history_entries_omitted=omitted,
                    history_entries_omitted_by_window=omitted,
                )
            )
        return tuple(conditioned)


class SkillRlSftPromptConditioner:
    """Reproduce the released SkillRL ALFWorld SFT instruction shape.

    The contract is derived from the published ``SkillRL-SFT-Data`` ALFWorld
    parquet artifact.  Skills are visible from step zero, history is capped at
    five entries and re-indexed within the visible window, and admissible
    actions are rendered as the dataset's unquoted comma-separated list.
    """

    def __init__(
        self,
        retriever: EpisodeRetriever,
        *,
        history_length: int = 5,
    ) -> None:
        if history_length != 5:
            raise ValueError("SkillRL SFT prompt requires history_length=5")
        self._retriever = retriever
        self._history_length = history_length

    def prepare_group(self, initial_state: CanonicalAgentState) -> ConditioningContext:
        retrieval = self._retriever.retrieve(initial_state.goal)
        return ConditioningContext(
            candidate_skill_ids=retrieval.skill_ids,
            retrieval=retrieval,
        )

    def condition_batch(
        self,
        requests: tuple[ConditioningRequest, ...],
        context: ConditioningContext,
    ) -> tuple[ConditionedPolicyInput, ...]:
        if context.retrieval is None:
            raise ValueError("SkillRL SFT prompt requires an episode retrieval result")
        skill_block = format_raw_skill_block(context.retrieval, style="skillrl")
        conditioned = []
        for request in requests:
            state = request.state
            history_limit = (
                request.history_limit
                if request.history_limit is not None
                else self._history_length
            )
            recent = state.history[-history_limit:] if history_limit else ()
            message = _render_skillrl_sft_message(
                state,
                skill_block=skill_block,
                recent_history=recent,
            )
            omitted = max(0, len(state.history) - history_limit)
            conditioned.append(
                ConditionedPolicyInput(
                    user_message=message,
                    candidate_skill_ids=context.candidate_skill_ids,
                    conditioning_trace={
                        "prompt_protocol": "skillrl-sft-alfworld-v1",
                        "skills_injected": True,
                    },
                    history_entries_omitted=omitted,
                    history_entries_omitted_by_window=omitted,
                )
            )
        return tuple(conditioned)


def _skillrl_admissible_actions(state: CanonicalAgentState) -> str:
    return "\n ".join(
        f"'{command}'"
        for command in state.admissible_commands
        if command != "help"
    )


def _render_skillrl_history(entries: Sequence[AgentHistoryEntry]) -> str:
    return "\n".join(
        f"[Observation {entry.step_index + 1}: '{entry.observation}', "
        f"Action {entry.step_index + 1}: '{entry.executed_action}']"
        for entry in entries
    )


def _render_skillrl_sft_history(entries: Sequence[AgentHistoryEntry]) -> str:
    return "\n".join(
        f"[Observation {index}: '{entry.observation}', "
        f"Action {index}: '{entry.executed_action}']"
        for index, entry in enumerate(entries, start=1)
    )


def _render_skillrl_sft_message(
    state: CanonicalAgentState,
    *,
    skill_block: str,
    recent_history: Sequence[AgentHistoryEntry],
) -> str:
    prefix = (
        "You are an expert agent operating in the ALFRED Embodied Environment.\n"
        f"Your task is to: {state.goal}\n\n"
        "## Retrieved Relevant Experience\n\n"
        f"{skill_block}\n\n"
        "## Current Progress\n"
    )
    actions = ", ".join(state.admissible_commands)
    if state.step_index == 0:
        progress = (
            f"Your current observation is: {state.observation}\n"
            "Your admissible actions of the current situation are: "
            f"[{actions}].\n\n"
        )
    else:
        history = _render_skillrl_sft_history(recent_history)
        progress = (
            "\nPrior to this step, you have already taken "
            f"{state.step_index} step(s). Below are the most recent "
            f"{len(recent_history)} observations and the corresponding actions "
            f"you took: {history}\n\n"
            f"You are now at step {state.step_index + 1} and your current "
            f"observation is: {state.observation}\n"
            "Your admissible actions of the current situation are: "
            f"[{actions}].\n\n"
        )
    return (
        prefix
        + progress
        + "Now it's your turn to take an action.\n"
        "You should first reason step-by-step about the current situation. This "
        "reasoning process MUST be enclosed within <think> </think> tags.\n"
        "Once you've finished your reasoning, you should choose an admissible "
        "action for current step and present it within <action> </action> tags."
    )


def _render_skillrl_initial_message(state: CanonicalAgentState) -> str:
    # INFO-SKILL's canonical seam splits the raw reset string at this marker;
    # reconstruct it because SkillRL feeds the unsplit reset observation into
    # ALFWORLD_TEMPLATE_NO_HIS.
    raw_initial_observation = (
        f"{state.observation}\n\nYour task is to: {state.goal}"
    )
    return _SKILLRL_ALFWORLD_TEMPLATE_NO_HISTORY.format(
        current_observation=raw_initial_observation,
        admissible_actions=_skillrl_admissible_actions(state),
    )


def _render_skillrl_memory_message(
    state: CanonicalAgentState,
    *,
    skill_block: str,
    history_limit: int,
) -> str:
    recent = state.history[-history_limit:] if history_limit else ()
    return _SKILLRL_ALFWORLD_TEMPLATE_WITH_MEMORY.format(
        task_description=state.goal,
        retrieved_memories=skill_block,
        step_count=len(state.history),
        history_length=len(recent),
        action_history=_render_skillrl_history(recent),
        current_step=len(state.history) + 1,
        current_observation=state.observation,
        admissible_actions=_skillrl_admissible_actions(state),
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
