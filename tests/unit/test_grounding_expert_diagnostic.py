from __future__ import annotations

import unittest

from infoskill.episode import TaskSpec
from infoskill.integrations.alfworld.grounding_diagnostic import (
    PlannerPayloadExpert,
    _summary,
    select_quarantined_tasks,
)


class GroundingExpertDiagnosticTests(unittest.TestCase):
    def test_planner_payload_expert_returns_first_current_policy_command(self) -> None:
        expert = PlannerPayloadExpert()

        action = expert.act(
            {"extra.expert_plan": ["open fridge 1", "take apple 1"]},
            reward=0.0,
            done=False,
            last_action="look",
        )

        self.assertEqual(action, "open fridge 1")

    def test_stratified_selection_is_deterministic_and_uses_only_quarantines(self) -> None:
        tasks = tuple(
            TaskSpec(
                task_id=task_id,
                split="train",
                task_type=task_type,
                goal="goal",
                environment_path=f"/data/{task_id}",
            )
            for task_type, task_id in (
                ("pick_and_place_simple", "simple-b"),
                ("pick_and_place_simple", "simple-a"),
                ("pick_and_place_simple", "simple-c"),
                ("pick_two_obj_and_place", "two-b"),
                ("pick_two_obj_and_place", "two-a"),
            )
        )
        quarantine = (
            {"task_id": "simple-c", "reason": "expert_action_not_admissible"},
            {"task_id": "simple-a", "reason": "expert_action_not_admissible"},
            {"task_id": "simple-b", "reason": "expert_exception:OSError"},
            {"task_id": "two-b", "reason": "expert_action_not_admissible"},
            {"task_id": "two-a", "reason": "expert_action_not_admissible"},
        )

        selected = select_quarantined_tasks(
            tasks=tasks,
            quarantine_rows=quarantine,
            tasks_per_type=1,
            selection_seed=17,
        )
        repeated = select_quarantined_tasks(
            tasks=tuple(reversed(tasks)),
            quarantine_rows=tuple(reversed(quarantine)),
            tasks_per_type=1,
            selection_seed=17,
        )

        self.assertEqual(selected, repeated)
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            {task.task_type for task in selected},
            {"pick_and_place_simple", "pick_two_obj_and_place"},
        )

    def test_summary_distinguishes_reproduced_failures_and_planner_rescues(self) -> None:
        rows = (
            {
                "source_quarantine": {
                    "reason": "expert_action_not_admissible",
                    "total_steps": 7,
                },
                "handcoded": {
                    "succeeded": False,
                    "reason": "expert_action_not_admissible",
                    "total_steps": 7,
                },
                "planner": {
                    "succeeded": True,
                    "reason": None,
                    "total_steps": 9,
                },
            },
            {
                "source_quarantine": {
                    "reason": "expert_action_not_admissible",
                    "total_steps": 5,
                },
                "handcoded": {
                    "succeeded": False,
                    "reason": "expert_replay_limit",
                    "total_steps": 150,
                },
                "planner": {
                    "succeeded": False,
                    "reason": "terminated_without_win",
                    "total_steps": 8,
                },
            },
        )

        summary = _summary(rows)

        self.assertEqual(summary["historical_failure_reproduced_count"], 1)
        self.assertEqual(summary["planner_rescue_count"], 1)
        self.assertEqual(summary["planner"]["success_count"], 1)


if __name__ == "__main__":
    unittest.main()
