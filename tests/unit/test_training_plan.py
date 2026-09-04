from __future__ import annotations

import unittest

from infoskill.training import TrainingProfile, resolve_training_plan


class TrainingPlanTests(unittest.TestCase):
    def test_registered_profiles_keep_development_and_formal_budgets_distinct(self) -> None:
        smoke = resolve_training_plan(TrainingProfile.SMOKE)
        integration = resolve_training_plan(TrainingProfile.INTEGRATION)
        pilot = resolve_training_plan(TrainingProfile.PILOT)
        formal = resolve_training_plan(TrainingProfile.FORMAL)

        self.assertEqual(
            (smoke.max_updates, smoke.task_groups_per_update, smoke.rollouts_per_task),
            (2, 1, 2),
        )
        self.assertEqual(
            (
                integration.max_updates,
                integration.task_groups_per_update,
                integration.rollouts_per_task,
            ),
            (20, 4, 4),
        )
        self.assertEqual(
            (pilot.max_updates, pilot.evaluation_kind),
            (100, "train_monitor"),
        )
        self.assertEqual(
            (
                formal.max_updates,
                formal.task_groups_per_update,
                formal.rollouts_per_task,
                formal.evaluation_kind,
                formal.include_monitor_tasks,
            ),
            (445, 8, 8, "valid_seen", True),
        )

    def test_only_development_profiles_allow_an_explicit_update_override(self) -> None:
        smoke = resolve_training_plan(TrainingProfile.SMOKE, max_updates=1)

        self.assertEqual(smoke.max_updates, 1)
        with self.assertRaisesRegex(ValueError, "formal training budget"):
            resolve_training_plan(TrainingProfile.FORMAL, max_updates=2)


if __name__ == "__main__":
    unittest.main()
