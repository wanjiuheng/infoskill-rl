from __future__ import annotations

from pathlib import Path
from typing import Mapping

from infoskill.domain.actions import INVALID_ACTION_SENTINEL
from infoskill.domain.state import AgentHistoryEntry, CanonicalAgentState
from infoskill.episode import EnvironmentTransition, TaskSpec

from .environment import _single_sequence, _split_initial_observation, _world_checksum


def _batch_infos(raw_infos: object, *, batch_size: int) -> tuple[dict[str, object], ...]:
    if not isinstance(raw_infos, Mapping):
        raise TypeError(f"unexpected ALFWorld infos type: {type(raw_infos)!r}")
    rows = [dict() for _ in range(batch_size)]
    for key, value in raw_infos.items():
        values = _single_sequence(value, field=str(key))
        if len(values) != batch_size:
            raise RuntimeError(
                f"ALFWorld info field {key!r} has {len(values)} values; expected {batch_size}"
            )
        for index, item in enumerate(values):
            rows[index][str(key)] = item
    return tuple(rows)


def _commands(info: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        str(command)
        for command in _single_sequence(
            info.get("admissible_commands", ()), field="admissible_commands"
        )
    )


class AlfworldEnvironmentBatch:
    """Adapt one TextWorld multiprocessing batch while preserving fixed slot identity."""

    def __init__(self, raw_environment: object, *, tasks: tuple[TaskSpec, ...]) -> None:
        if len(tasks) < 2:
            raise ValueError("native ALFWorld batching requires at least two tasks")
        if any(task.environment_path is None for task in tasks):
            raise ValueError("every batched ALFWorld task requires environment_path")
        self._raw_environment = raw_environment
        self._tasks = tasks
        self._states: list[CanonicalAgentState] = []
        self._last_infos: list[dict[str, object]] = []

        # TextworldBatchGymEnv normally shuffles its game pool during seed(). A rollout
        # batch needs a stable task/rollout slot mapping, so feed reset() the exact list
        # once. This is verified again through the returned extra.gamefile values.
        if not hasattr(raw_environment, "_gamefiles_iterator"):
            raise RuntimeError(
                "TextWorld batch environment does not expose the pinned game iterator seam"
            )
        raw_environment._gamefiles_iterator = iter(  # type: ignore[attr-defined]
            task.environment_path for task in tasks
        )

    def reset(self) -> tuple[CanonicalAgentState, ...]:
        raw_observations, raw_infos = self._raw_environment.reset()  # type: ignore[attr-defined]
        observations = _single_sequence(raw_observations, field="observations")
        if len(observations) != len(self._tasks):
            raise RuntimeError(
                f"ALFWorld returned {len(observations)} observations for {len(self._tasks)} slots"
            )
        infos = _batch_infos(raw_infos, batch_size=len(self._tasks))
        states = []
        for task, raw_observation, info in zip(self._tasks, observations, infos):
            gamefile = info.get("extra.gamefile")
            if gamefile is None:
                raise RuntimeError("ALFWorld batch reset did not return extra.gamefile")
            if Path(str(gamefile)).resolve() != Path(str(task.environment_path)).resolve():
                raise RuntimeError(
                    f"ALFWorld batch slot {task.task_id} returned unexpected gamefile: {gamefile}"
                )
            observation, goal = _split_initial_observation(str(raw_observation))
            states.append(
                CanonicalAgentState(
                    task_id=task.task_id,
                    split=task.split,
                    task_type=task.task_type,
                    goal=goal,
                    step_index=0,
                    observation=observation,
                    history=(),
                    admissible_commands=_commands(info),
                    done=False,
                    won=False,
                )
            )
        self._states = states
        self._last_infos = [dict(info) for info in infos]
        return tuple(states)

    def step(
        self, actions: tuple[str | None, ...]
    ) -> tuple[EnvironmentTransition | None, ...]:
        if len(actions) != len(self._tasks):
            raise ValueError("action count must match the ALFWorld batch size")
        if len(self._states) != len(self._tasks):
            raise RuntimeError("reset must be called before step")

        # TextWorld requires one command for every fixed slot. With auto_reset=False,
        # completed slots ignore later commands; `look` is only a transport filler and
        # never becomes an INFO-SKILL trajectory step.
        submitted = tuple(action if action is not None else "look" for action in actions)
        raw_observations, raw_scores, raw_dones, raw_infos = self._raw_environment.step(  # type: ignore[attr-defined]
            submitted
        )
        observations = _single_sequence(raw_observations, field="observations")
        scores = _single_sequence(raw_scores, field="scores")
        dones = _single_sequence(raw_dones, field="dones")
        infos = _batch_infos(raw_infos, batch_size=len(self._tasks))
        if not (
            len(observations)
            == len(scores)
            == len(dones)
            == len(infos)
            == len(self._tasks)
        ):
            raise RuntimeError("ALFWorld batch step returned inconsistent slot counts")

        transitions: list[EnvironmentTransition | None] = []
        for index, (action, raw_observation, raw_score, raw_done, info) in enumerate(
            zip(actions, observations, scores, dones, infos)
        ):
            if action is None:
                if not self._states[index].done:
                    raise RuntimeError("only completed ALFWorld batch slots may be inactive")
                transitions.append(None)
                continue
            state_before = self._states[index]
            pre_checksum = _world_checksum(self._last_infos[index])
            post_checksum = _world_checksum(info)
            if action == INVALID_ACTION_SENTINEL and pre_checksum != post_checksum:
                raise RuntimeError("invalid-action sentinel changed ALFWorld world state")
            won = bool(info.get("won", False))
            done = bool(raw_done)
            next_state = CanonicalAgentState(
                task_id=state_before.task_id,
                split=state_before.split,
                task_type=state_before.task_type,
                goal=state_before.goal,
                step_index=state_before.step_index + 1,
                observation=str(raw_observation).strip(),
                history=state_before.history
                + (
                    AgentHistoryEntry(
                        state_before.step_index, state_before.observation, action
                    ),
                ),
                admissible_commands=_commands(info),
                done=done,
                won=won,
                candidate_skill_ids=state_before.candidate_skill_ids,
            )
            self._states[index] = next_state
            self._last_infos[index] = dict(info)
            transitions.append(
                EnvironmentTransition(
                    next_state=next_state,
                    raw_observation=str(raw_observation),
                    raw_reward=float(raw_score),
                    raw_done=done,
                    raw_won=won,
                    info=info,
                    pre_world_state_checksum=pre_checksum,
                    post_world_state_checksum=post_checksum,
                )
            )
        return tuple(transitions)

    def close(self) -> None:
        self._raw_environment.close()  # type: ignore[attr-defined]
