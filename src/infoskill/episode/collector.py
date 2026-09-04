from __future__ import annotations

import hashlib

from infoskill.conditioning import SkillConditioner
from infoskill.domain.actions import resolve_action
from infoskill.domain.rewards import trajectory_reward
from infoskill.domain.state import CanonicalAgentState, render_state_views
from infoskill.rollout import GenerationParameters, GenerationRequest, RolloutBackend

from .contracts import EnvironmentFactory, TaskSpec, Trajectory, TrajectoryGroup, TrajectoryStep


def _semantic_seed(
    stream: str,
    master_seed: int,
    task_id: str,
    rollout_id: int,
    env_step: int = 0,
    global_update: int = 0,
) -> int:
    material = (
        f"{stream}|{master_seed}|{global_update}|{task_id}|{rollout_id}|{env_step}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)


class TrajectoryCollector:
    def __init__(
        self,
        *,
        environment_factory: EnvironmentFactory,
        conditioner: SkillConditioner,
        rollout_backend: RolloutBackend,
        max_steps: int,
        history_limit: int,
        invalid_action_penalty: float,
        generation_parameters: GenerationParameters | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._environment_factory = environment_factory
        self._conditioner = conditioner
        self._rollout_backend = rollout_backend
        self._max_steps = max_steps
        self._history_limit = history_limit
        self._invalid_action_penalty = invalid_action_penalty
        self._generation_parameters = generation_parameters or GenerationParameters.training()

    def collect_task_group(
        self,
        task: TaskSpec,
        *,
        rollouts_per_task: int,
        master_seed: int,
        global_update: int = 0,
    ) -> TrajectoryGroup:
        return self.collect_task_groups(
            (task,),
            rollouts_per_task=rollouts_per_task,
            master_seed=master_seed,
            global_update=global_update,
        )[0]

    def collect_task_groups(
        self,
        tasks: tuple[TaskSpec, ...],
        *,
        rollouts_per_task: int,
        master_seed: int,
        global_update: int = 0,
    ) -> tuple[TrajectoryGroup, ...]:
        if rollouts_per_task <= 0:
            raise ValueError("rollouts_per_task must be positive")
        if not tasks:
            raise ValueError("at least one task is required")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("task IDs must be unique within one rollout update")

        environments = tuple(
            tuple(
                self._environment_factory.create(
                    task,
                    rollout_id=rollout_id,
                    seed=_semantic_seed(
                        "environment",
                        master_seed,
                        task.task_id,
                        rollout_id,
                        global_update=global_update,
                    ),
                )
                for rollout_id in range(rollouts_per_task)
            )
            for task in tasks
        )
        states: list[list[CanonicalAgentState]] = []
        step_records: list[list[list[TrajectoryStep]]] = [
            [[] for _ in range(rollouts_per_task)] for _ in tasks
        ]
        try:
            states = [[environment.reset() for environment in group] for group in environments]
            contexts = [self._conditioner.prepare_group(group[0]) for group in states]
            active = [
                (task_index, rollout_id)
                for task_index, group in enumerate(states)
                for rollout_id, state in enumerate(group)
                if not state.done
            ]

            for env_step in range(self._max_steps):
                if not active:
                    break
                prepared: list[tuple[int, int, CanonicalAgentState, object]] = []
                for task_index in range(len(tasks)):
                    rollout_ids = [rollout_id for index, rollout_id in active if index == task_index]
                    if not rollout_ids:
                        continue
                    active_states = tuple(states[task_index][rollout_id] for rollout_id in rollout_ids)
                    views = tuple(
                        render_state_views(state, history_limit=self._history_limit)
                        for state in active_states
                    )
                    conditioned = self._conditioner.condition_batch(
                        active_states, views, contexts[task_index]
                    )
                    if len(conditioned) != len(rollout_ids):
                        raise RuntimeError("conditioner returned a different batch size")
                    prepared.extend(
                        (task_index, rollout_id, state, policy_input)
                        for rollout_id, state, policy_input in zip(
                            rollout_ids, active_states, conditioned
                        )
                    )

                requests = tuple(
                    GenerationRequest(
                        request_id=f"{tasks[task_index].task_id}:{rollout_id}:{env_step}",
                        task_id=tasks[task_index].task_id,
                        rollout_id=rollout_id,
                        env_step=env_step,
                        user_message=policy_input.user_message,  # type: ignore[attr-defined]
                        parameters=self._generation_parameters,
                        soft_prefix=policy_input.soft_prefix,  # type: ignore[attr-defined]
                        seed=_semantic_seed(
                            "policy_sampling",
                            master_seed,
                            tasks[task_index].task_id,
                            rollout_id,
                            env_step,
                            global_update,
                        ),
                    )
                    for task_index, rollout_id, _, policy_input in prepared
                )
                results = self._rollout_backend.generate(requests)
                by_request_id = {result.request_id: result for result in results}
                if len(by_request_id) != len(requests) or set(by_request_id) != {
                    request.request_id for request in requests
                }:
                    raise RuntimeError("rollout backend did not return exactly one result per request")

                next_active: list[tuple[int, int]] = []
                for (task_index, rollout_id, state, policy_input), request in zip(prepared, requests):
                    generation = by_request_id[request.request_id]
                    action = resolve_action(generation.text, state.admissible_commands)
                    transition = environments[task_index][rollout_id].step(action.executed_action)
                    step_records[task_index][rollout_id].append(
                        TrajectoryStep(
                            state_before=state,
                            conditioned_input=policy_input,
                            generation=generation,
                            action=action,
                            transition=transition,
                        )
                    )
                    states[task_index][rollout_id] = transition.next_state
                    if not transition.next_state.done:
                        next_active.append((task_index, rollout_id))
                active = next_active

            groups = []
            for task_index, task in enumerate(tasks):
                trajectories = []
                for rollout_id, (state, records) in enumerate(
                    zip(states[task_index], step_records[task_index])
                ):
                    invalid_count = sum(not record.action.is_executable for record in records)
                    trajectories.append(
                        Trajectory(
                            task=task,
                            rollout_id=rollout_id,
                            steps=tuple(records),
                            won=state.won,
                            environment_done=state.done,
                            horizon_exhausted=not state.done and len(records) == self._max_steps,
                            invalid_action_count=invalid_count,
                            reward=trajectory_reward(
                                won=state.won,
                                invalid_action_count=invalid_count,
                                invalid_action_penalty=self._invalid_action_penalty,
                            ),
                        )
                    )
                groups.append(TrajectoryGroup(task=task, trajectories=tuple(trajectories)))
            return tuple(groups)
        finally:
            for group in environments:
                for environment in group:
                    environment.close()
