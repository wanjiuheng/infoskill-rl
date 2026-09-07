from __future__ import annotations

import unittest
from contextlib import contextmanager
from threading import Barrier, Lock
from time import sleep

from infoskill.conditioning import NoSkillConditioner
from infoskill.domain.state import AgentHistoryEntry, CanonicalAgentState
from infoskill.episode import EnvironmentTransition, TaskSpec, TrajectoryCollector
from infoskill.rollout import GenerationRequest, GenerationResult


class _FakeEnvironment:
    def __init__(self, task: TaskSpec, rollout_id: int) -> None:
        self.task = task
        self.rollout_id = rollout_id
        self.step_index = 0
        self.history: tuple[AgentHistoryEntry, ...] = ()

    def _state(self, *, done: bool = False, won: bool = False) -> CanonicalAgentState:
        return CanonicalAgentState(
            task_id=self.task.task_id,
            split=self.task.split,
            task_type=self.task.task_type,
            goal=self.task.goal,
            step_index=self.step_index,
            observation="The fridge is closed.",
            history=self.history,
            admissible_commands=("look", "open fridge 1"),
            done=done,
            won=won,
        )

    def reset(self) -> CanonicalAgentState:
        return self._state()

    def step(self, action: str) -> EnvironmentTransition:
        before = self._state()
        won = self.rollout_id == 0 and action == "open fridge 1"
        self.history += (AgentHistoryEntry(self.step_index, before.observation, action),)
        self.step_index += 1
        after = self._state(done=won, won=won)
        return EnvironmentTransition(
            next_state=after,
            raw_observation=after.observation,
            raw_reward=float(won),
            raw_done=won,
            raw_won=won,
            info={},
        )

    def close(self) -> None:
        return None


class _FakeEnvironmentFactory:
    def create(self, task: TaskSpec, *, rollout_id: int, seed: int) -> _FakeEnvironment:
        return _FakeEnvironment(task, rollout_id)


class _FakeEnvironmentBatch:
    def __init__(self, environments: tuple[_FakeEnvironment, ...]) -> None:
        self.environments = environments
        self.actions: list[tuple[str | None, ...]] = []

    def reset(self) -> tuple[CanonicalAgentState, ...]:
        return tuple(environment.reset() for environment in self.environments)

    def step(
        self, actions: tuple[str | None, ...]
    ) -> tuple[EnvironmentTransition | None, ...]:
        self.actions.append(actions)
        return tuple(
            None if action is None else environment.step(action)
            for environment, action in zip(self.environments, actions)
        )

    def close(self) -> None:
        for environment in self.environments:
            environment.close()


class _FakeNativeBatchFactory(_FakeEnvironmentFactory):
    def __init__(self, *, rollouts_per_task: int) -> None:
        self.rollouts_per_task = rollouts_per_task
        self.individual_create_calls = 0
        self.batch: _FakeEnvironmentBatch | None = None

    def create(self, task: TaskSpec, *, rollout_id: int, seed: int) -> _FakeEnvironment:
        self.individual_create_calls += 1
        return super().create(task, rollout_id=rollout_id, seed=seed)

    def create_batch(
        self, tasks: tuple[TaskSpec, ...], *, seeds: tuple[int, ...]
    ) -> _FakeEnvironmentBatch:
        self.batch = _FakeEnvironmentBatch(
            tuple(
                _FakeEnvironment(task, index % self.rollouts_per_task)
                for index, task in enumerate(tasks)
            )
        )
        return self.batch


class _BarrierEnvironment(_FakeEnvironment):
    def __init__(self, task: TaskSpec, rollout_id: int, barrier: Barrier) -> None:
        super().__init__(task, rollout_id)
        self._barrier = barrier

    def step(self, action: str) -> EnvironmentTransition:
        self._barrier.wait(timeout=1.0)
        return super().step(action)


class _BarrierEnvironmentFactory:
    def __init__(self, parties: int) -> None:
        self._barrier = Barrier(parties)

    def create(self, task: TaskSpec, *, rollout_id: int, seed: int) -> _FakeEnvironment:
        return _BarrierEnvironment(task, rollout_id, self._barrier)


class _ParserLikeEnvironment(_BarrierEnvironment):
    def __init__(
        self,
        task: TaskSpec,
        rollout_id: int,
        factory: "_ParserLikeEnvironmentFactory",
    ) -> None:
        super().__init__(task, rollout_id, factory.step_barrier)
        self._factory = factory

    def reset(self) -> CanonicalAgentState:
        with self._factory.reset_lock:
            self._factory.active_resets += 1
            overlap = self._factory.active_resets > 1
        try:
            sleep(0.02)
            if overlap:
                raise RuntimeError("shared parser reset overlapped")
            return super().reset()
        finally:
            with self._factory.reset_lock:
                self._factory.active_resets -= 1


class _ParserLikeEnvironmentFactory:
    def __init__(self, parties: int) -> None:
        self.step_barrier = Barrier(parties)
        self.reset_lock = Lock()
        self.active_resets = 0

    def create(self, task: TaskSpec, *, rollout_id: int, seed: int) -> _FakeEnvironment:
        return _ParserLikeEnvironment(task, rollout_id, self)


class _FakeRolloutBackend:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.events: list[str] = []

    @contextmanager
    def rollout_session(self):
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")

    def generate(self, requests: tuple[GenerationRequest, ...]) -> tuple[GenerationResult, ...]:
        self.events.append("generate")
        self.batch_sizes.append(len(requests))
        results = []
        for request in requests:
            text = (
                "<action>open fridge 1</action>"
                if request.rollout_id == 0
                else "I might open fridge 1 after thinking."
            )
            results.append(
                GenerationResult(
                    request_id=request.request_id,
                    text=text,
                    finish_reason="stop",
                    token_ids=(1,),
                    token_logprobs=(-0.1,),
                    prompt_token_count=10,
                )
            )
        return tuple(results)


class _RecordingConditioner(NoSkillConditioner):
    def __init__(self) -> None:
        self.identities: list[tuple[str, int, int, int, int]] = []

    def condition_batch(self, requests, context):
        self.identities.extend(
            (
                request.state.task_id,
                request.rollout_id,
                request.state.step_index,
                request.global_update,
                request.latent_seed,
            )
            for request in requests
        )
        return super().condition_batch(requests, context)


class TrajectoryCollectorTests(unittest.TestCase):
    def test_latent_seeds_are_semantic_and_independent_of_collection_instance(self) -> None:
        task = TaskSpec("game-1", "train", "pick_and_place_simple", "look")

        def collect(global_update: int):
            conditioner = _RecordingConditioner()
            TrajectoryCollector(
                environment_factory=_FakeEnvironmentFactory(),
                conditioner=conditioner,
                rollout_backend=_FakeRolloutBackend(),
                max_steps=2,
                history_limit=2,
                invalid_action_penalty=0.01,
            ).collect_task_group(
                task,
                rollouts_per_task=2,
                master_seed=17,
                global_update=global_update,
            )
            return conditioner.identities

        first = collect(3)
        repeated = collect(3)
        next_update = collect(4)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_update)
        self.assertEqual(len({item[-1] for item in first}), len(first))

    def test_group_collection_batches_active_envs_and_preserves_invalid_failures(self) -> None:
        backend = _FakeRolloutBackend()
        collector = TrajectoryCollector(
            environment_factory=_FakeEnvironmentFactory(),
            conditioner=NoSkillConditioner(),
            rollout_backend=backend,
            max_steps=2,
            history_limit=2,
            invalid_action_penalty=0.01,
        )
        task = TaskSpec(
            task_id="game-1",
            split="train",
            task_type="pick_and_place_simple",
            goal="put the apple in the fridge",
        )

        group = collector.collect_task_group(task, rollouts_per_task=2, master_seed=0)

        self.assertEqual(backend.batch_sizes, [2, 1])
        self.assertEqual(backend.events, ["enter", "generate", "generate", "exit"])
        self.assertEqual(len(group.trajectories), 2)
        self.assertTrue(group.trajectories[0].won)
        self.assertEqual(group.trajectories[0].reward, 1.0)
        self.assertFalse(group.trajectories[1].won)
        self.assertEqual(group.trajectories[1].invalid_action_count, 2)
        self.assertEqual(group.trajectories[1].reward, -0.02)
        self.assertEqual(group.trajectories[1].steps[-1].action.executed_action, "__invalid_action__")

    def test_multiple_task_groups_share_one_global_generation_batch(self) -> None:
        backend = _FakeRolloutBackend()
        collector = TrajectoryCollector(
            environment_factory=_FakeEnvironmentFactory(),
            conditioner=NoSkillConditioner(),
            rollout_backend=backend,
            max_steps=1,
            history_limit=2,
            invalid_action_penalty=0.01,
        )
        tasks = tuple(
            TaskSpec(
                task_id=f"game-{index}",
                split="train",
                task_type="pick_and_place_simple",
                goal="put the apple in the fridge",
            )
            for index in range(2)
        )

        groups = collector.collect_task_groups(tasks, rollouts_per_task=2, master_seed=0)

        self.assertEqual(len(groups), 2)
        self.assertEqual(backend.batch_sizes, [4])

    def test_nested_collections_share_one_backend_rollout_session(self) -> None:
        backend = _FakeRolloutBackend()
        collector = TrajectoryCollector(
            environment_factory=_FakeEnvironmentFactory(),
            conditioner=NoSkillConditioner(),
            rollout_backend=backend,
            max_steps=1,
            history_limit=2,
            invalid_action_penalty=0.01,
        )
        task = TaskSpec("game-1", "train", "pick_and_place_simple", "look")

        with collector.rollout_session():
            collector.collect_task_group(task, rollouts_per_task=1, master_seed=0)
            collector.collect_task_group(task, rollouts_per_task=1, master_seed=0)

        self.assertEqual(
            backend.events,
            ["enter", "generate", "generate", "exit"],
        )

    def test_independent_environment_steps_can_run_concurrently(self) -> None:
        collector = TrajectoryCollector(
            environment_factory=_BarrierEnvironmentFactory(parties=2),
            conditioner=NoSkillConditioner(),
            rollout_backend=_FakeRolloutBackend(),
            max_steps=1,
            history_limit=2,
            invalid_action_penalty=0.01,
            environment_workers=2,
        )
        task = TaskSpec("game-1", "train", "pick_and_place_simple", "look")

        group = collector.collect_task_group(
            task,
            rollouts_per_task=2,
            master_seed=0,
        )

        self.assertEqual(len(group.trajectories), 2)
        metrics = collector.performance_metrics()
        for key in (
            "perf/collector_seconds",
            "perf/environment_create_seconds",
            "perf/environment_reset_seconds",
            "perf/environment_step_seconds",
            "perf/environment_close_seconds",
        ):
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)
        self.assertEqual(metrics["perf/environment_workers"], 2.0)

    def test_parallel_environment_results_preserve_serial_order_and_semantics(
        self,
    ) -> None:
        task = TaskSpec("game-1", "train", "pick_and_place_simple", "look")

        def collect(environment_workers: int):
            collector = TrajectoryCollector(
                environment_factory=_FakeEnvironmentFactory(),
                conditioner=NoSkillConditioner(),
                rollout_backend=_FakeRolloutBackend(),
                max_steps=2,
                history_limit=2,
                invalid_action_penalty=0.01,
                environment_workers=environment_workers,
            )
            return collector.collect_task_group(
                task,
                rollouts_per_task=2,
                master_seed=0,
            )

        self.assertEqual(collect(1), collect(2))

    def test_non_thread_safe_environment_loading_stays_serial(self) -> None:
        collector = TrajectoryCollector(
            environment_factory=_ParserLikeEnvironmentFactory(parties=2),
            conditioner=NoSkillConditioner(),
            rollout_backend=_FakeRolloutBackend(),
            max_steps=1,
            history_limit=2,
            invalid_action_penalty=0.01,
            environment_workers=2,
        )
        task = TaskSpec("game-1", "train", "pick_and_place_simple", "look")

        group = collector.collect_task_group(
            task,
            rollouts_per_task=2,
            master_seed=0,
        )

        self.assertEqual(len(group.trajectories), 2)

    def test_native_batch_backend_matches_individual_collector_semantics(self) -> None:
        tasks = tuple(
            TaskSpec(
                f"game-{index}", "train", "pick_and_place_simple", "look"
            )
            for index in range(2)
        )
        baseline = TrajectoryCollector(
            environment_factory=_FakeEnvironmentFactory(),
            conditioner=NoSkillConditioner(),
            rollout_backend=_FakeRolloutBackend(),
            max_steps=2,
            history_limit=2,
            invalid_action_penalty=0.01,
        ).collect_task_groups(tasks, rollouts_per_task=2, master_seed=0)
        native_factory = _FakeNativeBatchFactory(rollouts_per_task=2)
        native_collector = TrajectoryCollector(
            environment_factory=native_factory,
            conditioner=NoSkillConditioner(),
            rollout_backend=_FakeRolloutBackend(),
            max_steps=2,
            history_limit=2,
            invalid_action_penalty=0.01,
            environment_backend="native_batch",
        )

        actual = native_collector.collect_task_groups(
            tasks, rollouts_per_task=2, master_seed=0
        )

        self.assertEqual(actual, baseline)
        self.assertEqual(native_factory.individual_create_calls, 0)
        self.assertIsNotNone(native_factory.batch)
        self.assertEqual(
            native_factory.batch.actions[1],  # type: ignore[union-attr]
            (None, "__invalid_action__", None, "__invalid_action__"),
        )
        self.assertEqual(
            native_collector.performance_metrics()["perf/native_environment_batch"],
            1.0,
        )
        self.assertEqual(
            native_collector.performance_metrics()[
                "perf/environment_forced_terminations"
            ],
            0.0,
        )

    def test_native_batch_backend_falls_back_for_one_evaluation_slot(self) -> None:
        factory = _FakeNativeBatchFactory(rollouts_per_task=1)
        collector = TrajectoryCollector(
            environment_factory=factory,
            conditioner=NoSkillConditioner(),
            rollout_backend=_FakeRolloutBackend(),
            max_steps=1,
            history_limit=2,
            invalid_action_penalty=0.01,
            environment_backend="native_batch",
        )

        groups = collector.collect_task_group(
            TaskSpec("game-1", "train", "pick_and_place_simple", "look"),
            rollouts_per_task=1,
            master_seed=0,
        )

        self.assertEqual(len(groups.trajectories), 1)
        self.assertEqual(factory.individual_create_calls, 1)
        self.assertEqual(
            collector.performance_metrics()["perf/native_environment_batch"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
