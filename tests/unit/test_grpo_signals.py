from __future__ import annotations

import unittest

from infoskill.episode import TaskSpec, Trajectory, TrajectoryGroup
from infoskill.learning import build_group_advantage_signals, summarize_grpo_signals


def _group(
    task_type: str,
    outcomes: tuple[tuple[bool, int], ...],
) -> TrajectoryGroup:
    task = TaskSpec(task_type, "train", task_type, "goal")
    return TrajectoryGroup(
        task=task,
        trajectories=tuple(
            Trajectory(
                task=task,
                rollout_id=index,
                steps=(),
                won=won,
                environment_done=True,
                horizon_exhausted=False,
                invalid_action_count=invalid_count,
                reward=float(won) - 0.01 * invalid_count,
            )
            for index, (won, invalid_count) in enumerate(outcomes)
        ),
    )


class GrpoSignalTests(unittest.TestCase):
    def test_success_target_excludes_invalid_action_shaping(self) -> None:
        groups = (
            _group("all_fail", ((False, 0), (False, 3))),
            _group("mixed", ((False, 0), (True, 2))),
        )

        signals = build_group_advantage_signals(groups)

        self.assertNotEqual(signals[0].shaped_policy, (0.0, 0.0))
        self.assertEqual(signals[0].task_success, (0.0, 0.0))
        self.assertNotEqual(signals[1].task_success, (0.0, 0.0))

    def test_summary_reports_group_and_task_type_signal_coverage(self) -> None:
        groups = (
            _group("heat", ((False, 0), (False, 3))),
            _group("heat", ((False, 0), (True, 2))),
            _group("cool", ((True, 0), (True, 0))),
        )

        metrics = summarize_grpo_signals(groups, build_group_advantage_signals(groups))

        self.assertEqual(metrics["grpo_signal/group_count"], 3.0)
        self.assertEqual(metrics["grpo_signal/mixed_outcome_group_count"], 1.0)
        self.assertEqual(metrics["grpo_signal/all_failure_group_count"], 1.0)
        self.assertEqual(metrics["grpo_signal/all_success_group_count"], 1.0)
        self.assertEqual(metrics["grpo_signal/shaping_only_group_count"], 1.0)
        self.assertEqual(metrics["grpo_signal/zero_policy_signal_group_count"], 1.0)
        self.assertEqual(metrics["grpo_signal/task_success_signal_group_rate"], 1 / 3)
        self.assertEqual(
            metrics["grpo_signal/task_type/heat/task_success_signal_group_rate"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
