from __future__ import annotations

import unittest

from infoskill.learning import group_relative_advantages


class GroupAdvantageTests(unittest.TestCase):
    def test_episode_rewards_use_sample_standard_deviation(self) -> None:
        advantages = group_relative_advantages((1.0, 0.0, 0.0))

        self.assertAlmostEqual(advantages[0], 1.154698538, places=7)
        self.assertAlmostEqual(advantages[1], -0.577349269, places=7)
        self.assertAlmostEqual(advantages[2], -0.577349269, places=7)
        self.assertEqual(group_relative_advantages((0.5, 0.5, 0.5)), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
