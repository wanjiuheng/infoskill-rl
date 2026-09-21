from __future__ import annotations

import unittest

from infoskill.training.drift_guard import TrainingDriftGuard
from infoskill.training.trainer import UpdateMetrics


class TrainingDriftGuardTests(unittest.TestCase):
    def test_requires_consecutive_simultaneous_threshold_breaches(self) -> None:
        guard = TrainingDriftGuard(
            ppo_kl_threshold=0.02,
            invalid_action_rate_threshold=0.05,
            consecutive_updates=2,
        )

        first = guard.observe(
            UpdateMetrics(
                global_update=196,
                values={
                    "actor/ppo_kl": 0.03,
                    "rollout/invalid_action_rate": 0.06,
                },
            )
        )
        reset = guard.observe(
            UpdateMetrics(
                global_update=197,
                values={
                    "actor/ppo_kl": 0.03,
                    "rollout/invalid_action_rate": 0.04,
                },
            )
        )
        second_first = guard.observe(
            UpdateMetrics(
                global_update=198,
                values={
                    "actor/ppo_kl": 0.04,
                    "rollout/invalid_action_rate": 0.08,
                },
            )
        )
        triggered = guard.observe(
            UpdateMetrics(
                global_update=199,
                values={
                    "actor/ppo_kl": 0.05,
                    "rollout/invalid_action_rate": 0.09,
                },
            )
        )

        self.assertFalse(first.triggered)
        self.assertEqual(first.consecutive_breaches, 1)
        self.assertFalse(reset.breached)
        self.assertEqual(reset.consecutive_breaches, 0)
        self.assertFalse(second_first.triggered)
        self.assertTrue(triggered.triggered)
        self.assertEqual(triggered.trigger_update, 199)
        self.assertTrue(guard.triggered)

    def test_trigger_is_latched_for_checkpoint_boundary_pause(self) -> None:
        guard = TrainingDriftGuard(
            ppo_kl_threshold=0.02,
            invalid_action_rate_threshold=0.05,
            consecutive_updates=1,
        )

        guard.observe(
            UpdateMetrics(
                global_update=201,
                values={
                    "actor/ppo_kl": 0.03,
                    "rollout/invalid_action_rate": 0.06,
                },
            )
        )
        after_recovery = guard.observe(
            UpdateMetrics(
                global_update=202,
                values={
                    "actor/ppo_kl": 0.0,
                    "rollout/invalid_action_rate": 0.0,
                },
            )
        )

        self.assertTrue(after_recovery.triggered)
        self.assertEqual(after_recovery.trigger_update, 201)


if __name__ == "__main__":
    unittest.main()
