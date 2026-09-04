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

    def update_policy(self, groups, advantages, *, global_update: int):
        del groups, advantages
        self.updated.append(global_update)
        return {"actor/ppo_kl": 0.0, "actor/grad_norm": 1.0}

    def update_auxiliary(self, groups, advantages, *, global_update: int):
        raise AssertionError("M0 must not run an auxiliary update")

    def synchronize_rollout_weights(self) -> None:
        return None


def _tasks() -> tuple[TaskSpec, ...]:
    return tuple(TaskSpec(str(index), "train", "kind", "goal") for index in range(4))


class InfoSkillTrainerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
