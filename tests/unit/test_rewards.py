from __future__ import annotations

import unittest

from infoskill.domain.rewards import trajectory_reward


class TrajectoryRewardTests(unittest.TestCase):
    def test_reward_is_win_minus_invalid_action_penalty(self) -> None:
        self.assertAlmostEqual(
            trajectory_reward(won=True, invalid_action_count=2, invalid_action_penalty=0.01),
            0.98,
        )
        self.assertAlmostEqual(
            trajectory_reward(won=False, invalid_action_count=2, invalid_action_penalty=0.01),
            -0.02,
        )


if __name__ == "__main__":
    unittest.main()
