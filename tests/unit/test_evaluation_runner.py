from __future__ import annotations

import unittest

from infoskill.config import EvaluationConfig
from infoskill.episode import TaskSpec, Trajectory, TrajectoryGroup
from infoskill.evaluation import EvaluationRunner
from infoskill.rollout import PromptLengthError


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

    def test_prompt_overflow_preserves_the_exact_failed_input(self) -> None:
        task = TaskSpec(
            task_id="task-1",
            split="valid_seen",
            task_type="pick_and_place_simple",
            goal="goal",
        )

        class OverflowCollector:
            def collect_task_groups(self, *args, **kwargs):
                del args, kwargs
                raise PromptLengthError(
                    request_id="task-1:0:3",
                    token_count=4_101,
                    max_prompt_tokens=4_096,
                    user_message="full raw skill prompt",
                )

        run = EvaluationRunner(
            collector_factory=OverflowCollector,  # type: ignore[arg-type]
            config=EvaluationConfig(),
            task_batch_size=1,
        ).run((task,))

        record = run.records[0]
        self.assertIn("PromptLengthError", record.infrastructure_error or "")
        self.assertEqual(
            record.infrastructure_detail,
            {
                "request_id": "task-1:0:3",
                "token_count": 4_101,
                "max_prompt_tokens": 4_096,
                "user_message": "full raw skill prompt",
            },
        )

    def test_performance_metrics_are_aggregated_across_evaluation_batches(self) -> None:
        task = TaskSpec(
            task_id="task-1",
            split="valid_seen",
            task_type="pick_and_place_simple",
            goal="goal",
        )

        class TimedCollector(_RecordingCollector):
            def performance_metrics(self):
                return {
                    "perf/collector_seconds": 2.0,
                    "perf/rollout_conditioning_seconds": 0.5,
                    "perf/environment_workers": 1.0,
                    "perf/native_environment_batch": 1.0,
                }

        run = EvaluationRunner(
            collector_factory=TimedCollector,  # type: ignore[arg-type]
            config=EvaluationConfig(),
            task_batch_size=1,
        ).run((task, task))

        self.assertEqual(run.performance_metrics["perf/collector_seconds"], 4.0)
        self.assertEqual(
            run.performance_metrics["perf/rollout_conditioning_seconds"],
            1.0,
        )
        self.assertEqual(run.performance_metrics["perf/environment_workers"], 1.0)
        self.assertEqual(
            run.performance_metrics["perf/native_environment_batch"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
