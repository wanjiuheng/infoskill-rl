from __future__ import annotations

import unittest

from infoskill.episode import TaskSpec
from infoskill.integrations.alfworld.expert_replay import ExpertReplayResult
from infoskill.integrations.alfworld.planner_pilot import (
    build_planner_pilot_report,
    select_stratified_tasks,
)
from infoskill.integrations.alfworld.tasks import ALFWORLD_TASK_TYPES


class PlannerPilotTests(unittest.TestCase):
    def test_selection_is_balanced_deterministic_and_order_independent(self) -> None:
        tasks = tuple(
            TaskSpec(
                task_id=f"{task_type}-{index}",
                split="train",
                task_type=task_type,
                goal=f"goal {index}",
            )
            for task_type in ALFWORLD_TASK_TYPES
            for index in range(4)
        )

        selected = select_stratified_tasks(
            tasks=tasks,
            tasks_per_type=2,
            selection_seed=17,
        )
        repeated = select_stratified_tasks(
            tasks=tuple(reversed(tasks)),
            tasks_per_type=2,
            selection_seed=17,
        )

        self.assertEqual(selected, repeated)
        self.assertEqual(len(selected), 12)
        for task_type in ALFWORLD_TASK_TYPES:
            self.assertEqual(
                sum(task.task_type == task_type for task in selected),
                2,
            )

    def test_selection_rejects_an_underfilled_task_type(self) -> None:
        tasks = (
            TaskSpec(
                task_id="only-one",
                split="train",
                task_type=ALFWORLD_TASK_TYPES[0],
                goal="goal",
            ),
        )

        with self.assertRaisesRegex(ValueError, "requires 2"):
            select_stratified_tasks(
                tasks=tasks,
                tasks_per_type=2,
                selection_seed=0,
            )

    def test_report_is_explicitly_non_formal_and_reports_tail_lengths(self) -> None:
        results = []
        for index, task_type in enumerate(ALFWORLD_TASK_TYPES):
            succeeded = index != len(ALFWORLD_TASK_TYPES) - 1
            results.append(
                (
                    task_type,
                    ExpertReplayResult(
                        task_id=f"task-{index}",
                        succeeded=succeeded,
                        samples=(),
                        total_steps=10 + index * 10,
                        quarantine_reason=None if succeeded else "expert_replay_limit",
                    ),
                )
            )

        report = build_planner_pilot_report(
            results=results,
            tasks_per_type=1,
            selection_seed=3,
            train_task_manifest_sha256="abc",
            code_revision="def",
            max_replay_steps=150,
            persist_horizon=30,
        )

        self.assertTrue(report["pilot_only"])
        self.assertTrue(report["formal_run_required"])
        self.assertEqual(report["selected_games"], 6)
        self.assertEqual(report["successful_games"], 5)
        self.assertEqual(report["over_persist_horizon"], 3)
        self.assertIn("p90", report["trajectory_lengths"])
        per_type = report["task_type_counts"]
        self.assertEqual(per_type[ALFWORLD_TASK_TYPES[-1]]["success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
