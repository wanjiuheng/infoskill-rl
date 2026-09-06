from __future__ import annotations

import unittest

from infoskill.config import EvaluationConfig
from infoskill.episode import TaskSpec, Trajectory, TrajectoryGroup
from infoskill.evaluation import EvaluationRunner


class _RecordingCollector:
    def __init__(self) -> None:
        self.global_updates: list[int] = []

    def collect_task_groups(
        self,
        tasks: tuple[TaskSpec, ...],
        *,
        rollouts_per_task: int,
        master_seed: int,
        global_update: int,
    ) -> tuple[TrajectoryGroup, ...]:
        del master_seed
        self.global_updates.append(global_update)
        return tuple(
            TrajectoryGroup(
                task=task,
                trajectories=(
                    Trajectory(
                        task=task,
                        rollout_id=0,
                        steps=(),
                        won=False,
                        environment_done=True,
                        horizon_exhausted=False,
                        invalid_action_count=0,
                        reward=0.0,
                    ),
                ),
            )
            for task in tasks
        )


class EvaluationRunnerTests(unittest.TestCase):
    def test_checkpoint_label_does_not_change_the_evaluation_random_stream(self) -> None:
        config = EvaluationConfig()
        tasks = tuple(
            TaskSpec(
                task_id=f"{denominator.task_type}-{index}",
                split="valid_seen",
                task_type=denominator.task_type,
                goal="goal",
            )
            for denominator in config.denominators
            for index in range(denominator.count)
        )
        collector = _RecordingCollector()
        runner = EvaluationRunner(
            collector_factory=lambda: collector,  # type: ignore[arg-type]
            config=config,
            task_batch_size=140,
            master_seed=0,
        )

        result = runner.run(tasks, checkpoint_step=25)

        self.assertTrue(result.summary.is_complete)
        self.assertEqual(collector.global_updates, [0])


if __name__ == "__main__":
    unittest.main()
