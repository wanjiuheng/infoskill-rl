from __future__ import annotations

import unittest

from infoskill.episode import TaskSpec, Trajectory, TrajectoryGroup
from infoskill.training import InfoSkillTrainer, TaskSchedule


class _Collector:
    def __init__(self) -> None:
        self.collected_task_ids: list[str] = []

    def collect_task_groups(
        self,
        tasks: tuple[TaskSpec, ...],
        *,
        rollouts_per_task: int,
        master_seed: int,
        global_update: int,
    ) -> tuple[TrajectoryGroup, ...]:
        del master_seed, global_update
        self.collected_task_ids.extend(task.task_id for task in tasks)
        return tuple(
            TrajectoryGroup(
                task=task,
                trajectories=tuple(
                    Trajectory(
                        task=task,
                        rollout_id=index,
                        steps=(),
                        won=False,
                        environment_done=True,
                        horizon_exhausted=False,
                        invalid_action_count=0,
                        reward=0.0,
                    )
                    for index in range(rollouts_per_task)
                ),
            )
            for task in tasks
        )


class _Runtime:
    def __init__(self) -> None:
        self.updated: list[int] = []
        self.policy_advantages = None
        self.fidelity_targets = None

    def update_policy(self, groups, policy_advantages, *, global_update: int):
        del groups
        self.policy_advantages = policy_advantages
        self.updated.append(global_update)
        return {"actor/ppo_kl": 0.0, "actor/grad_norm": 1.0}

    def update_auxiliary(self, groups, fidelity_targets, *, global_update: int):
        del groups, global_update
        self.fidelity_targets = fidelity_targets
        return {"aux/fidelity_loss": 0.0}

    def synchronize_rollout_weights(self) -> None:
        return None


def _tasks() -> tuple[TaskSpec, ...]:
    return tuple(TaskSpec(str(index), "train", "kind", "goal") for index in range(4))


class InfoSkillTrainerTests(unittest.TestCase):
    def test_pause_at_a_scheduled_boundary_keeps_its_evaluation(self) -> None:
        evaluation_calls: list[int] = []
        pause_requested = False
        tasks = tuple(
            TaskSpec(
                task_id=f"task-{index}",
                split="train",
                task_type="pick_and_place_simple",
                goal="put object in receptacle",
            )
            for index in range(26)
        )

        def request_pause(update, groups) -> None:
            nonlocal pause_requested
            del groups
            pause_requested = update.global_update == 25

        trainer = InfoSkillTrainer(
            collector=_Collector(),  # type: ignore[arg-type]
            runtime=_Runtime(),  # type: ignore[arg-type]
            schedule=TaskSchedule(tasks, master_seed=0),
            task_groups_per_update=1,
            rollouts_per_task=2,
            master_seed=0,
            auxiliary_enabled=False,
            on_update=request_pause,
            on_evaluate=evaluation_calls.append,
            on_checkpoint=lambda update, schedule: None,
            should_pause=lambda: pause_requested,
            evaluate_every=25,
            checkpoint_every=5,
        )

        trainer.fit(max_updates=26)

        self.assertEqual(evaluation_calls, [0, 25])
        self.assertEqual(trainer.global_update, 25)
        self.assertTrue(trainer.paused)

    def test_pause_on_the_final_nonperiodic_update_keeps_final_evaluation(
        self,
    ) -> None:
        evaluation_calls: list[int] = []
        pause_requested = False
        tasks = tuple(
            TaskSpec(
                task_id=f"task-{index}",
                split="train",
                task_type="pick_and_place_simple",
                goal="put object in receptacle",
            )
            for index in range(29)
        )

        def request_pause(update, groups) -> None:
            nonlocal pause_requested
            del groups
            pause_requested = update.global_update == 29

        trainer = InfoSkillTrainer(
            collector=_Collector(),  # type: ignore[arg-type]
            runtime=_Runtime(),  # type: ignore[arg-type]
            schedule=TaskSchedule(tasks, master_seed=0),
            task_groups_per_update=1,
            rollouts_per_task=2,
            master_seed=0,
            auxiliary_enabled=False,
            on_update=request_pause,
            on_evaluate=evaluation_calls.append,
            on_checkpoint=lambda update, schedule: None,
            should_pause=lambda: pause_requested,
            evaluate_every=25,
            checkpoint_every=5,
        )

        trainer.fit(max_updates=29)

        self.assertEqual(evaluation_calls, [0, 25, 29])
        self.assertFalse(trainer.paused)

    def test_requested_pause_commits_the_completed_update_without_final_evaluation(
        self,
    ) -> None:
        checkpoint_calls: list[int] = []
        evaluation_calls: list[int] = []
        pause_requested = False

        def request_pause(update, groups) -> None:
            nonlocal pause_requested
            del update, groups
            pause_requested = True

        trainer = InfoSkillTrainer(
            collector=_Collector(),  # type: ignore[arg-type]
            runtime=_Runtime(),  # type: ignore[arg-type]
            schedule=TaskSchedule(_tasks(), master_seed=0),
            task_groups_per_update=1,
            rollouts_per_task=2,
            master_seed=0,
            auxiliary_enabled=False,
            on_update=request_pause,
            on_checkpoint=lambda update, schedule: checkpoint_calls.append(update),
            on_evaluate=evaluation_calls.append,
            checkpoint_every=5,
            evaluate_every=25,
            should_pause=lambda: pause_requested,
        )

        trainer.fit(max_updates=4)

        self.assertEqual(trainer.global_update, 1)
        self.assertTrue(trainer.paused)
        self.assertEqual(checkpoint_calls, [1])
        self.assertEqual(evaluation_calls, [0])

    def test_one_25_update_cycle_evaluates_only_at_start_and_end(self) -> None:
        evaluation_calls: list[int] = []
        tasks = tuple(
            TaskSpec(str(index), "train", "kind", "goal") for index in range(25)
        )
        trainer = InfoSkillTrainer(
            collector=_Collector(),  # type: ignore[arg-type]
            runtime=_Runtime(),  # type: ignore[arg-type]
            schedule=TaskSchedule(tasks, master_seed=0),
            task_groups_per_update=1,
            rollouts_per_task=2,
            master_seed=0,
            auxiliary_enabled=False,
            on_evaluate=evaluation_calls.append,
            checkpoint_every=5,
            evaluate_every=25,
        )

        trainer.fit(max_updates=25)

        self.assertEqual(evaluation_calls, [0, 25])

    def test_update_boundary_checkpoint_is_emitted_once_and_resume_skips_consumed_tasks(
        self,
    ) -> None:
        first_collector = _Collector()
        first_schedule = TaskSchedule(_tasks(), master_seed=0)
        checkpoint_states = []
        first = InfoSkillTrainer(
            collector=first_collector,  # type: ignore[arg-type]
            runtime=_Runtime(),  # type: ignore[arg-type]
            schedule=first_schedule,
            task_groups_per_update=1,
            rollouts_per_task=2,
            master_seed=0,
            auxiliary_enabled=False,
            on_checkpoint=lambda update, schedule: checkpoint_states.append(
                (update, schedule.state())
            ),
            checkpoint_every=1,
        )

        first.fit(max_updates=1, evaluate_at_start=False)

        self.assertEqual([update for update, _ in checkpoint_states], [1])
        second_collector = _Collector()
        second_schedule = TaskSchedule(_tasks(), master_seed=0)
        resumed = InfoSkillTrainer(
            collector=second_collector,  # type: ignore[arg-type]
            runtime=_Runtime(),  # type: ignore[arg-type]
            schedule=second_schedule,
            task_groups_per_update=1,
            rollouts_per_task=2,
            master_seed=0,
            auxiliary_enabled=False,
            checkpoint_every=1,
        )
        resumed.restore(global_update=1, schedule_state=checkpoint_states[0][1])

        resumed.fit(max_updates=2, evaluate_at_start=False)

        self.assertEqual(resumed.global_update, 2)
        self.assertNotEqual(
            first_collector.collected_task_ids,
            second_collector.collected_task_ids,
        )

    def test_resume_at_target_update_is_a_checkpoint_no_op(self) -> None:
        tasks = _tasks()
        source = TaskSchedule(tasks, master_seed=0)
        source.next_batch(1)
        checkpoint_calls: list[int] = []
        evaluation_calls: list[int] = []
        resumed = InfoSkillTrainer(
            collector=_Collector(),  # type: ignore[arg-type]
            runtime=_Runtime(),  # type: ignore[arg-type]
            schedule=TaskSchedule(tasks, master_seed=0),
            task_groups_per_update=1,
            rollouts_per_task=2,
            master_seed=0,
            auxiliary_enabled=False,
            on_checkpoint=lambda update, schedule: checkpoint_calls.append(update),
            on_evaluate=evaluation_calls.append,
            checkpoint_every=1,
            evaluate_every=25,
        )
        resumed.restore(global_update=1, schedule_state=source.state())

        resumed.fit(max_updates=1, evaluate_at_start=False)

        self.assertEqual(checkpoint_calls, [])
        self.assertEqual(evaluation_calls, [])

    def test_update_reports_nonnegative_core_stage_timings(self) -> None:
        captured = []
        trainer = InfoSkillTrainer(
            collector=_Collector(),  # type: ignore[arg-type]
            runtime=_Runtime(),  # type: ignore[arg-type]
            schedule=TaskSchedule(_tasks(), master_seed=0),
            task_groups_per_update=1,
            rollouts_per_task=2,
            master_seed=0,
            auxiliary_enabled=False,
            on_update=lambda update, groups: captured.append(update.values),
            checkpoint_every=1,
        )

        trainer.fit(max_updates=1, evaluate_at_start=False)

        self.assertEqual(len(captured), 1)
        for key in (
            "perf/rollout_seconds",
            "perf/advantage_seconds",
            "perf/policy_update_seconds",
            "perf/rollout_weight_sync_seconds",
            "perf/core_update_seconds",
        ):
            self.assertIn(key, captured[0])
            self.assertGreaterEqual(captured[0][key], 0.0)

    def test_auxiliary_receives_success_only_target(self) -> None:
        task = TaskSpec("task", "train", "kind", "goal")
        group = TrajectoryGroup(
            task=task,
            trajectories=(
                Trajectory(task, 0, (), False, True, False, 0, 0.0),
                Trajectory(task, 1, (), False, True, False, 3, -0.03),
            ),
        )

        class _FixedCollector:
            def collect_task_groups(self, *args, **kwargs):
                return (group,)

        runtime = _Runtime()
        captured = []
        trainer = InfoSkillTrainer(
            collector=_FixedCollector(),  # type: ignore[arg-type]
            runtime=runtime,  # type: ignore[arg-type]
            schedule=TaskSchedule((task,), master_seed=0),
            task_groups_per_update=1,
            rollouts_per_task=2,
            master_seed=0,
            auxiliary_enabled=True,
            on_update=lambda update, groups: captured.append(update.values),
        )

        trainer.fit(max_updates=1, evaluate_at_start=False)

        self.assertNotEqual(runtime.policy_advantages, ((0.0, 0.0),))
        self.assertEqual(runtime.fidelity_targets, ((0.0, 0.0),))
        self.assertEqual(captured[0]["grpo_signal/shaping_only_group_rate"], 1.0)
        self.assertEqual(captured[0]["grpo_signal/task_success_signal_group_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
