from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class AgentHistoryEntry:
    step_index: int
    observation: str
    executed_action: str

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("history step_index must be non-negative")
        if not self.observation.strip():
            raise ValueError("history observation must not be empty")
        if not self.executed_action.strip():
            raise ValueError("history executed_action must not be empty")


@dataclass(frozen=True, slots=True)
class CanonicalAgentState:
    task_id: str
    split: str
    task_type: str
    goal: str
    step_index: int
    observation: str
    history: tuple[AgentHistoryEntry, ...]
    admissible_commands: tuple[str, ...]
    done: bool = False
    won: bool = False
    candidate_skill_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("task_id", "split", "task_type", "goal", "observation"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if self.won and not self.done:
            raise ValueError("won=True requires done=True")
        if any(not command.strip() for command in self.admissible_commands):
            raise ValueError("admissible_commands must not contain empty commands")


@dataclass(frozen=True, slots=True)
class StateViews:
    retrieval_view: str
    compression_view: str
    policy_view: str


def _render_history(entries: Sequence[AgentHistoryEntry]) -> str:
    if not entries:
        return "None"
    return "\n".join(
        (
            f"Step {entry.step_index + 1}:\n"
            f"Observation before action: {entry.observation}\n"
            f"Executed action: {entry.executed_action}"
        )
        for entry in entries
    )


def render_policy_message(
    state: CanonicalAgentState,
    *,
    history_limit: int = 2,
    retrieved_skill_block: str | None = None,
) -> str:
    if history_limit < 0:
        raise ValueError("history_limit must be non-negative")
    recent = state.history[-history_limit:] if history_limit else ()
    history_text = _render_history(recent)
    skill_text = ""
    if retrieved_skill_block:
        skill_text = f"\n\n## Retrieved Relevant Skills\n{retrieved_skill_block.strip()}"
    actions = "\n ".join(f"'{command}'" for command in state.admissible_commands)
    return (
        "You are an expert agent operating in the ALFRED Embodied Environment. "
        f"Your task is to: {state.goal}{skill_text}\n"
        f"Prior to this step, you have already taken {state.step_index} step(s). "
        f"Below are the most recent {len(recent)} observations and the corresponding actions you took:\n"
        f"{history_text}\n"
        f"You are now at step {state.step_index + 1} and your current observation is: "
        f"{state.observation}\n"
        f"Your admissible actions of the current situation are: [{actions}].\n\n"
        "Now it's your turn to take an action.\n"
        "You should first reason step-by-step about the current situation. "
        "This reasoning process MUST be enclosed within <think> </think> tags.\n"
        "Once you've finished your reasoning, you should choose an admissible action for current step "
        "and present it within <action> </action> tags."
    )


def render_state_views(state: CanonicalAgentState, *, history_limit: int = 2) -> StateViews:
    """Render the registered retrieval, compression, and policy views."""

    if history_limit < 0:
        raise ValueError("history_limit must be non-negative")
    recent = state.history[-history_limit:] if history_limit else ()
    history_text = _render_history(recent)
    compression_view = (
        f"Task: {state.goal}\n"
        f"Completed environment steps: {state.step_index}\n"
        f"Recent history:\n{history_text}\n"
        f"Current observation: {state.observation}"
    )
    policy_view = render_policy_message(state, history_limit=history_limit)
    return StateViews(
        retrieval_view=state.goal,
        compression_view=compression_view,
        policy_view=policy_view,
    )
