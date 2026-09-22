from __future__ import annotations

import hashlib
import json

from infoskill.domain.actions import INVALID_ACTION_SENTINEL
from infoskill.domain.state import AgentHistoryEntry, CanonicalAgentState
from infoskill.episode import EnvironmentTransition, TaskSpec

from .action_adapter import DemonstrationActionAdapter
from .demonstrations import normalize_available_actions


def _instruction_text(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("instruction:"):
        text = text[len("instruction:") :].strip()
    if not text:
        raise RuntimeError("WebShop reset returned an empty goal")
    return text


def _checksum(observation: str, actions: tuple[str, ...], done: bool) -> str:
    payload = {"observation": observation, "actions": actions, "done": done}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


class WebShopEnvironment:
    """Adapt one frozen WebShop session to the canonical environment seam."""

    def __init__(self, raw_environment: object, *, task: TaskSpec, session_index: int) -> None:
        self._raw_environment = DemonstrationActionAdapter(raw_environment)
        self._task = task
        self._session_index = session_index
        self._state: CanonicalAgentState | None = None

    def reset(self) -> CanonicalAgentState:
        observation, _ = self._raw_environment.reset(session=self._session_index)  # type: ignore[attr-defined]
        goal = _instruction_text(self._raw_environment.instruction_text)
        if goal != self._task.goal:
            raise RuntimeError("WebShop reset returned a different goal than the frozen manifest")
        actions = normalize_available_actions(
            self._raw_environment.get_available_actions()
        )
        self._state = CanonicalAgentState(
            task_id=self._task.task_id,
            split=self._task.split,
            task_type=self._task.task_type,
            goal=goal,
            step_index=0,
            observation=str(observation).strip(),
            history=(),
            admissible_commands=actions,
        )
        return self._state

    def step(self, action: str) -> EnvironmentTransition:
        if self._state is None:
            raise RuntimeError("reset must be called before step")
        state_before = self._state
        pre_checksum = _checksum(
            state_before.observation,
            state_before.admissible_commands,
            state_before.done,
        )
        if action == INVALID_ACTION_SENTINEL:
            next_state = CanonicalAgentState(
                task_id=state_before.task_id,
                split=state_before.split,
                task_type=state_before.task_type,
                goal=state_before.goal,
                step_index=state_before.step_index + 1,
                observation=state_before.observation,
                history=state_before.history
                + (
                    AgentHistoryEntry(
                        state_before.step_index,
                        state_before.observation,
                        action,
                    ),
                ),
                admissible_commands=state_before.admissible_commands,
                candidate_skill_ids=state_before.candidate_skill_ids,
            )
            self._state = next_state
            return EnvironmentTransition(
                next_state=next_state,
                raw_observation=state_before.observation,
                raw_reward=0.0,
                raw_done=False,
                raw_won=False,
                info={"invalid_action_noop": True},
                pre_world_state_checksum=pre_checksum,
                post_world_state_checksum=pre_checksum,
            )
        observation, reward, done, info = self._raw_environment.step(action)
        actions = (
            ()
            if done
            else normalize_available_actions(
                self._raw_environment.get_available_actions()
            )
        )
        history = state_before.history + (
            AgentHistoryEntry(
                state_before.step_index,
                state_before.observation,
                action,
            ),
        )
        won = bool(done and float(reward) >= 1.0)
        next_state = CanonicalAgentState(
            task_id=state_before.task_id,
            split=state_before.split,
            task_type=state_before.task_type,
            goal=state_before.goal,
            step_index=state_before.step_index + 1,
            observation=str(observation).strip(),
            history=history,
            admissible_commands=actions,
            done=bool(done),
            won=won,
            candidate_skill_ids=state_before.candidate_skill_ids,
        )
        post_checksum = _checksum(
            next_state.observation,
            next_state.admissible_commands,
            next_state.done,
        )
        self._state = next_state
        return EnvironmentTransition(
            next_state=next_state,
            raw_observation=str(observation),
            raw_reward=float(reward),
            raw_done=bool(done),
            raw_won=won,
            info=dict(info) if isinstance(info, dict) else {"raw_info": info},
            pre_world_state_checksum=pre_checksum,
            post_world_state_checksum=post_checksum,
        )

    def close(self) -> None:
        self._raw_environment.close()
