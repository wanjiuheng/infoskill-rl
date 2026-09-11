from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Protocol, Sequence

from infoskill.domain.actions import resolve_action
from infoskill.domain.state import CanonicalAgentState
from infoskill.episode import EnvironmentTransition, TaskSpec

from .expert_replay import (
    ExpertActionMismatch,
    ExpertReplayResult,
    GroundingSample,
    _exception_message,
    _string_sequence,
)
from .planner_expert import PlannerPayloadExpert


class PlannerBatchEnvironment(Protocol):
    forced_worker_terminations: int

    def reset(self) -> tuple[CanonicalAgentState, ...]: ...

    def expert_payloads(self) -> tuple[Mapping[str, object], ...]: ...

    def step(
        self,
        actions: tuple[str | None, ...],
    ) -> tuple[EnvironmentTransition | None, ...]: ...

    def close(self) -> None: ...


class StrictPlannerBatchReplay:
    """Replay independent ALFWorld planner tasks in fixed native batch slots."""

    def __init__(
        self,
        *,
        max_replay_steps: int = 150,
        persist_horizon: int = 30,
    ) -> None:
        if max_replay_steps <= 0 or persist_horizon <= 0:
            raise ValueError("planner batch replay limits must be positive")
        if persist_horizon > max_replay_steps:
            raise ValueError("persist_horizon cannot exceed max_replay_steps")
        self._max_replay_steps = max_replay_steps
        self._persist_horizon = persist_horizon

    def run(
        self,
        *,
        tasks: Sequence[TaskSpec],
        environment: PlannerBatchEnvironment,
        candidate_skill_ids: Sequence[tuple[str, ...]],
    ) -> tuple[ExpertReplayResult, ...]:
        if len(tasks) < 2:
            raise ValueError("planner native batch replay requires at least two tasks")
        if len(candidate_skill_ids) != len(tasks):
            raise ValueError("candidate skill IDs must match planner batch slots")

        size = len(tasks)
        samples: list[list[GroundingSample]] = [[] for _ in tasks]
        total_steps = [0] * size
        results: list[ExpertReplayResult | None] = [None] * size
        transport_done = [False] * size
        experts = [PlannerPayloadExpert() for _ in tasks]
        states: list[CanonicalAgentState] = []
        payloads: tuple[Mapping[str, object], ...] = ()
        try:
            states = [
                replace(state, candidate_skill_ids=skill_ids)
                for state, skill_ids in zip(environment.reset(), candidate_skill_ids)
            ]
            payloads = environment.expert_payloads()
            self._require_slot_count(states, size, "reset states")
            self._require_slot_count(payloads, size, "expert payloads")
            for task, expert, payload, state in zip(tasks, experts, payloads, states):
                if task.environment_path is None:
                    raise ValueError("planner batch task requires environment_path")
                expert.reset(task.environment_path)
                expert.observe(str(payload.get("feedback", state.observation)))

            for decision_index in range(self._max_replay_steps):
                actions: list[str | None] = []
                executing = [False] * size
                for index, (task, state, payload, expert) in enumerate(
                    zip(tasks, states, payloads, experts)
                ):
                    if results[index] is not None:
                        actions.append(None if transport_done[index] else "look")
                        continue
                    try:
                        proposed = (
                            "look"
                            if decision_index == 0
                            else expert.act(payload, 0.0, state.done, "")
                        )
                    except Exception as error:
                        results[index] = self._quarantine(
                            task,
                            total_steps[index],
                            f"expert_exception:{type(error).__name__}",
                            exception_stage="expert_act",
                            exception_type=type(error).__name__,
                            exception_message=_exception_message(error),
                        )
                        actions.append("look")
                        continue
                    resolution = resolve_action(
                        f"<action>{proposed}</action>",
                        state.admissible_commands,
                    )
                    if not resolution.is_executable or resolution.resolved_action is None:
                        results[index] = self._quarantine(
                            task,
                            total_steps[index],
                            "expert_action_not_admissible",
                            action_mismatch=ExpertActionMismatch(
                                step_index=state.step_index,
                                proposed_action=str(proposed),
                                last_action=(
                                    samples[index][-1].expert_action
                                    if samples[index]
                                    else ""
                                ),
                                observation=state.observation,
                                admissible_commands=state.admissible_commands,
                                environment_expert_plan=_string_sequence(
                                    payload.get("extra.expert_plan")
                                ),
                            ),
                        )
                        actions.append("look")
                        continue
                    samples[index].append(
                        GroundingSample(
                            state=state,
                            expert_action=resolution.resolved_action,
                        )
                    )
                    actions.append(resolution.resolved_action)
                    executing[index] = True

                if not any(executing):
                    break
                transitions = environment.step(tuple(actions))
                self._require_slot_count(transitions, size, "environment transitions")
                payloads = environment.expert_payloads()
                self._require_slot_count(payloads, size, "expert payloads")
                for index, transition in enumerate(transitions):
                    if transition is None:
                        if executing[index]:
                            raise RuntimeError(
                                "active planner slot returned no environment transition"
                            )
                        transport_done[index] = True
                        continue
                    if not executing[index]:
                        transport_done[index] = transition.next_state.done
                        continue
                    total_steps[index] += 1
                    states[index] = replace(
                        transition.next_state,
                        candidate_skill_ids=candidate_skill_ids[index],
                    )
                    if states[index].won:
                        results[index] = ExpertReplayResult(
                            task_id=tasks[index].task_id,
                            succeeded=True,
                            samples=tuple(samples[index][: self._persist_horizon]),
                            total_steps=total_steps[index],
                            quarantine_reason=None,
                        )
                        transport_done[index] = True
                    elif states[index].done:
                        results[index] = self._quarantine(
                            tasks[index],
                            total_steps[index],
                            "terminated_without_win",
                        )
                        transport_done[index] = True

            for index, result in enumerate(results):
                if result is None:
                    results[index] = self._quarantine(
                        tasks[index],
                        total_steps[index],
                        "expert_replay_limit",
                    )
        finally:
            environment.close()

        if environment.forced_worker_terminations:
            raise RuntimeError(
                "native planner batch required forced worker termination: "
                f"{environment.forced_worker_terminations}"
            )
        return tuple(result for result in results if result is not None)

    @staticmethod
    def _require_slot_count(values: Sequence[object], expected: int, label: str) -> None:
        if len(values) != expected:
            raise RuntimeError(
                f"planner batch returned {len(values)} {label}; expected {expected}"
            )

    @staticmethod
    def _quarantine(
        task: TaskSpec,
        total_steps: int,
        reason: str,
        *,
        exception_stage: str | None = None,
        exception_type: str | None = None,
        exception_message: str | None = None,
        action_mismatch: ExpertActionMismatch | None = None,
    ) -> ExpertReplayResult:
        return ExpertReplayResult(
            task_id=task.task_id,
            succeeded=False,
            samples=(),
            total_steps=total_steps,
            quarantine_reason=reason,
            exception_stage=exception_stage,
            exception_type=exception_type,
            exception_message=exception_message,
            action_mismatch=action_mismatch,
        )
