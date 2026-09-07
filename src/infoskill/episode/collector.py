from __future__ import annotations

import hashlib
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from typing import Callable, Literal, TypeVar, cast

from infoskill.conditioning import ConditioningRequest, SkillConditioner
from infoskill.domain.actions import resolve_action
from infoskill.domain.rewards import trajectory_reward
from infoskill.domain.state import CanonicalAgentState, render_state_views
from infoskill.rollout import GenerationParameters, GenerationRequest, RolloutBackend

from .contracts import (
    Environment,
    EnvironmentFactory,
    EnvironmentTransition,
    TaskSpec,
    Trajectory,
    TrajectoryGroup,
    TrajectoryStep,
)


_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


def _reset_environment(environment: Environment) -> CanonicalAgentState:
    return environment.reset()


def _step_environment(item: tuple[Environment, str]) -> EnvironmentTransition:
    environment, action = item
    return environment.step(action)


def _close_environment(environment: Environment) -> None:
    environment.close()


def _ordered_map(
    executor: Executor | None,
    function: Callable[[_Input], _Output],
    items: tuple[_Input, ...],
) -> tuple[_Output, ...]:
    if executor is None:
        return tuple(map(function, items))
    return tuple(executor.map(function, items))


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
        environment_workers: int = 1,
        environment_backend: Literal["individual", "native_batch"] = "individual",
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if environment_workers <= 0:
            raise ValueError("environment_workers must be positive")
        if environment_backend not in {"individual", "native_batch"}:
            raise ValueError("environment_backend must be individual or native_batch")
        if environment_backend == "native_batch" and environment_workers != 1:
            raise ValueError(
                "native_batch owns its process count; environment_workers must remain 1"
            )
        self._environment_factory = environment_factory
        self._conditioner = conditioner
        self._rollout_backend = rollout_backend
        self._max_steps = max_steps
        self._history_limit = history_limit
        self._invalid_action_penalty = invalid_action_penalty
        self._generation_parameters = generation_parameters or GenerationParameters.training()
        self._environment_workers = environment_workers
        self._environment_backend = environment_backend
        self._rollout_session_depth = 0
        self._last_performance_metrics: dict[str, float] = {}

    def performance_metrics(self) -> dict[str, float]:
        return dict(self._last_performance_metrics)

    @contextmanager
    def rollout_session(self):
        if self._rollout_session_depth > 0:
            self._rollout_session_depth += 1
            try:
                yield
            finally:
                self._rollout_session_depth -= 1
            return
        session_factory = getattr(self._rollout_backend, "rollout_session", nullcontext)
        self._rollout_session_depth = 1
        try:
            with session_factory():
                yield
        finally:
            self._rollout_session_depth = 0

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

        collection_started = time.perf_counter()
        stage_started = collection_started
        environment_reset_seconds = 0.0
        environment_step_seconds = 0.0
        environment_close_seconds = 0.0
        conditioning_seconds = 0.0
        backend_generate_seconds = 0.0
        action_resolution_seconds = 0.0
        environment_forced_terminations = 0.0
        native_batch = None
        use_native_batch = (
            self._environment_backend == "native_batch"
            and len(tasks) * rollouts_per_task > 1
        )
        if use_native_batch:
            create_batch = getattr(self._environment_factory, "create_batch", None)
            if not callable(create_batch):
                raise TypeError(
                    "native_batch requires an environment factory with create_batch()"
                )
            slot_tasks = tuple(
                task for task in tasks for _ in range(rollouts_per_task)
            )
            slot_seeds = tuple(
                _semantic_seed(
                    "environment",
                    master_seed,
                    task.task_id,
                    rollout_id,
                    global_update=global_update,
                )
                for task in tasks
                for rollout_id in range(rollouts_per_task)
            )
            native_batch = create_batch(slot_tasks, seeds=slot_seeds)
            environments: tuple[tuple[Environment, ...], ...] = ()
        else:
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
        environment_create_seconds = time.perf_counter() - stage_started
        flat_environments = tuple(
            environment for group in environments for environment in group
        )
        executor: ThreadPoolExecutor | None = None
        if self._environment_workers > 1 and len(flat_environments) > 1:
            executor = ThreadPoolExecutor(
                max_workers=min(self._environment_workers, len(flat_environments)),
                thread_name_prefix="infoskill-env",
            )
        states: list[list[CanonicalAgentState]] = []
        step_records: list[list[list[TrajectoryStep]]] = [
            [[] for _ in range(rollouts_per_task)] for _ in tasks
        ]
        try:
            stage_started = time.perf_counter()
            # TextWorld's PDDL loader uses a module-level Tatsu parser whose
            # mutable parse stacks are not thread-safe. Loading happens inside
            # reset(), so resets must remain serial even when already-loaded
            # environment steps are allowed to run concurrently.
            if native_batch is not None:
                flat_states = tuple(native_batch.reset())
            else:
                flat_states = _ordered_map(
                    None,
                    _reset_environment,
                    flat_environments,
                )
            expected_slots = len(tasks) * rollouts_per_task
            if len(flat_states) != expected_slots:
                raise RuntimeError(
                    f"environment reset returned {len(flat_states)} slots; expected {expected_slots}"
                )
            states = [
                list(flat_states[offset : offset + rollouts_per_task])
                for offset in range(0, len(flat_states), rollouts_per_task)
            ]
            environment_reset_seconds = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            contexts = [self._conditioner.prepare_group(group[0]) for group in states]
            conditioning_seconds += time.perf_counter() - stage_started
            active = [
                (task_index, rollout_id)
                for task_index, group in enumerate(states)
                for rollout_id, state in enumerate(group)
                if not state.done
            ]

            with self.rollout_session():
                for env_step in range(self._max_steps):
                    if not active:
                        break
                    prepared: list[tuple[int, int, CanonicalAgentState, object]] = []
                    for task_index in range(len(tasks)):
                        rollout_ids = [
                            rollout_id for index, rollout_id in active if index == task_index
                        ]
                        if not rollout_ids:
                            continue
                        active_states = tuple(
                            states[task_index][rollout_id] for rollout_id in rollout_ids
                        )
                        views = tuple(
                            render_state_views(state, history_limit=self._history_limit)
                            for state in active_states
                        )
                        stage_started = time.perf_counter()
                        conditioned = self._conditioner.condition_batch(
                            tuple(
                                ConditioningRequest(
                                    state=state,
                                    views=view,
                                    rollout_id=rollout_id,
                                    global_update=global_update,
                                    latent_seed=_semantic_seed(
                                        "latent_epsilon",
                                        master_seed,
                                        tasks[task_index].task_id,
                                        rollout_id,
                                        env_step,
                                        global_update,
                                    ),
                                )
                                for rollout_id, state, view in zip(
                                    rollout_ids, active_states, views
                                )
                            ),
                            contexts[task_index],
                        )
                        conditioning_seconds += time.perf_counter() - stage_started
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
                            request_id=(
                                f"{tasks[task_index].task_id}:{rollout_id}:{env_step}"
                            ),
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
                    stage_started = time.perf_counter()
                    results = self._rollout_backend.generate(requests)
                    backend_generate_seconds += time.perf_counter() - stage_started
                    by_request_id = {result.request_id: result for result in results}
                    if len(by_request_id) != len(requests) or set(by_request_id) != {
                        request.request_id for request in requests
                    }:
                        raise RuntimeError(
                            "rollout backend did not return exactly one result per request"
                        )

                    stage_started = time.perf_counter()
                    resolved_steps = []
                    for (task_index, rollout_id, state, policy_input), request in zip(
                        prepared, requests
                    ):
                        generation = by_request_id[request.request_id]
                        action = resolve_action(generation.text, state.admissible_commands)
                        resolved_steps.append(
                            (
                                task_index,
                                rollout_id,
                                state,
                                policy_input,
                                generation,
                                action,
                            )
                        )
                    action_resolution_seconds += time.perf_counter() - stage_started
                    stage_started = time.perf_counter()
                    if native_batch is not None:
                        slot_actions: list[str | None] = [
                            None
                        ] * (len(tasks) * rollouts_per_task)
                        active_slots = []
                        for (
                            task_index,
                            rollout_id,
                            _,
                            _,
                            _,
                            action,
                        ) in resolved_steps:
                            slot = task_index * rollouts_per_task + rollout_id
                            slot_actions[slot] = action.executed_action
                            active_slots.append(slot)
                        all_transitions = tuple(native_batch.step(tuple(slot_actions)))
                        if len(all_transitions) != len(slot_actions):
                            raise RuntimeError(
                                "native environment batch returned an unexpected slot count"
                            )
                        transitions = tuple(
                            cast(EnvironmentTransition, all_transitions[slot])
                            for slot in active_slots
                        )
                        if any(transition is None for transition in transitions):
                            raise RuntimeError(
                                "native environment batch omitted an active transition"
                            )
                    else:
                        transitions = _ordered_map(
                            executor,
                            _step_environment,
                            tuple(
                                (
                                    environments[task_index][rollout_id],
                                    action.executed_action,
                                )
                                for (
                                    task_index,
                                    rollout_id,
                                    _,
                                    _,
                                    _,
                                    action,
                                ) in resolved_steps
                            ),
                        )
                    environment_step_seconds += time.perf_counter() - stage_started

                    next_active: list[tuple[int, int]] = []
                    for resolved_step, transition in zip(
                        resolved_steps,
                        transitions,
                    ):
                        (
                            task_index,
                            rollout_id,
                            state,
                            policy_input,
                            generation,
                            action,
                        ) = resolved_step
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
            try:
                stage_started = time.perf_counter()
                if native_batch is not None:
                    native_batch.close()
                    environment_forced_terminations = float(
                        getattr(native_batch, "forced_worker_terminations", 0)
                    )
                else:
                    _ordered_map(executor, _close_environment, flat_environments)
                environment_close_seconds = time.perf_counter() - stage_started
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
                self._last_performance_metrics = {
                    "perf/collector_seconds": time.perf_counter()
                    - collection_started,
                    "perf/environment_create_seconds": environment_create_seconds,
                    "perf/environment_reset_seconds": environment_reset_seconds,
                    "perf/environment_step_seconds": environment_step_seconds,
                    "perf/environment_close_seconds": environment_close_seconds,
                    "perf/rollout_conditioning_seconds": conditioning_seconds,
                    "perf/rollout_backend_generate_seconds": backend_generate_seconds,
                    "perf/rollout_action_resolution_seconds": action_resolution_seconds,
                    "perf/environment_workers": float(self._environment_workers),
                    "perf/native_environment_batch": float(
                        use_native_batch
                    ),
                    "perf/environment_forced_terminations": environment_forced_terminations,
                }
