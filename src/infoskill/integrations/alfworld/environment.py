from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from infoskill.domain.actions import INVALID_ACTION_SENTINEL
from infoskill.domain.state import AgentHistoryEntry, CanonicalAgentState
from infoskill.episode import EnvironmentTransition, TaskSpec


_TASK_MARKER = "Your task is to:"


def _single_item(value: object) -> object:
    if isinstance(value, (str, bytes, Mapping)):
        return value
    try:
        if len(value) == 1:  # type: ignore[arg-type]
            return value[0]  # type: ignore[index]
    except (TypeError, KeyError, IndexError):
        pass
    return value


def _single_info(raw_infos: object) -> dict[str, object]:
    if isinstance(raw_infos, (list, tuple)):
        if len(raw_infos) != 1 or not isinstance(raw_infos[0], Mapping):
            raise TypeError("expected exactly one ALFWorld info mapping")
        return dict(raw_infos[0])
    if not isinstance(raw_infos, Mapping):
        raise TypeError(f"unexpected ALFWorld infos type: {type(raw_infos)!r}")
    return {str(key): _single_item(value) for key, value in raw_infos.items()}


def _single_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"ALFWorld {field} must be a sequence, not text")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"ALFWorld {field} must be a sequence") from error
    return items


def _world_checksum(info: Mapping[str, object]) -> str:
    facts = info.get("facts")
    if facts is not None:
        material: object = sorted(str(item) for item in _single_sequence(facts, field="facts"))
    else:
        commands = _single_sequence(info.get("admissible_commands", ()), field="admissible_commands")
        material = {
            "admissible_commands": sorted(str(command) for command in commands),
            "won": bool(info.get("won", False)),
        }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_initial_observation(raw_observation: str) -> tuple[str, str]:
    if _TASK_MARKER not in raw_observation:
        raise RuntimeError("ALFWorld initial observation did not contain a task description")
    observation, goal = raw_observation.rsplit(_TASK_MARKER, 1)
    observation = observation.strip()
    goal = goal.strip()
    if not observation or not goal:
        raise RuntimeError("ALFWorld initial observation could not be split into observation and goal")
    return observation, goal


class AlfworldEnvironment:
    """Translate one ALFWorld batch-size-1 environment into the canonical seam."""

    def __init__(self, raw_environment: object, *, task: TaskSpec) -> None:
        self._raw_environment = raw_environment
        self._task = task
        self._state: CanonicalAgentState | None = None
        self._last_info: dict[str, object] | None = None
        self._raw_feedback: str | None = None

    def reset(self) -> CanonicalAgentState:
        raw_observations, raw_infos = self._raw_environment.reset()  # type: ignore[attr-defined]
        observations = _single_sequence(raw_observations, field="observations")
        if len(observations) != 1:
            raise RuntimeError("ALFWorld adapter requires batch_size=1")
        info = _single_info(raw_infos)
        gamefile = info.get("extra.gamefile")
        if gamefile and self._task.environment_path:
            if Path(str(gamefile)).resolve() != Path(self._task.environment_path).resolve():
                raise RuntimeError(f"ALFWorld returned an unexpected gamefile: {gamefile}")
        observation, goal = _split_initial_observation(str(observations[0]))
        commands = tuple(
            str(command)
            for command in _single_sequence(info.get("admissible_commands", ()), field="admissible_commands")
        )
        self._state = CanonicalAgentState(
            task_id=self._task.task_id,
            split=self._task.split,
            task_type=self._task.task_type,
            goal=goal,
            step_index=0,
            observation=observation,
            history=(),
            admissible_commands=commands,
            done=False,
            won=False,
        )
        self._last_info = info
        self._raw_feedback = str(observations[0])
        return self._state

    def step(self, action: str) -> EnvironmentTransition:
        if self._state is None or self._last_info is None:
            raise RuntimeError("reset must be called before step")
        state_before = self._state
        pre_checksum = _world_checksum(self._last_info)
        raw_observations, raw_scores, raw_dones, raw_infos = self._raw_environment.step([action])  # type: ignore[attr-defined]
        observations = _single_sequence(raw_observations, field="observations")
        scores = _single_sequence(raw_scores, field="scores")
        dones = _single_sequence(raw_dones, field="dones")
        if not (len(observations) == len(scores) == len(dones) == 1):
            raise RuntimeError("ALFWorld adapter requires batch_size=1")
        info = _single_info(raw_infos)
        post_checksum = _world_checksum(info)
        if action == INVALID_ACTION_SENTINEL and pre_checksum != post_checksum:
            raise RuntimeError("invalid-action sentinel changed ALFWorld world state")
        won = bool(info.get("won", False))
        done = bool(dones[0])
        commands = tuple(
            str(command)
            for command in _single_sequence(info.get("admissible_commands", ()), field="admissible_commands")
        )
        history = state_before.history + (
            AgentHistoryEntry(state_before.step_index, state_before.observation, action),
        )
        next_state = CanonicalAgentState(
            task_id=state_before.task_id,
            split=state_before.split,
            task_type=state_before.task_type,
            goal=state_before.goal,
            step_index=state_before.step_index + 1,
            observation=str(observations[0]).strip(),
            history=history,
            admissible_commands=commands,
            done=done,
            won=won,
            candidate_skill_ids=state_before.candidate_skill_ids,
        )
        self._state = next_state
        self._last_info = info
        self._raw_feedback = str(observations[0])
        return EnvironmentTransition(
            next_state=next_state,
            raw_observation=str(observations[0]),
            raw_reward=float(scores[0]),
            raw_done=done,
            raw_won=won,
            info=info,
            pre_world_state_checksum=pre_checksum,
            post_world_state_checksum=post_checksum,
        )

    def expert_payload(self) -> Mapping[str, object]:
        if self._state is None or self._last_info is None or self._raw_feedback is None:
            raise RuntimeError("reset must be called before requesting expert payload")
        payload = dict(self._last_info)
        payload["feedback"] = self._raw_feedback
        payload["won"] = self._state.won
        payload["admissible_commands"] = self._state.admissible_commands
        return payload

    def close(self) -> None:
        self._raw_environment.close()  # type: ignore[attr-defined]
