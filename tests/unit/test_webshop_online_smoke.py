from __future__ import annotations

import unittest

from infoskill.integrations.webshop.online_smoke import map_runtime_goals


class WebShopOnlineSmokeTests(unittest.TestCase):
    def test_runtime_goals_map_to_first_official_goal_index(self) -> None:
        official = ["red walking shoes", "blue travel bag", "red walking shoes"]
        runtime = [
            {"instruction_text": "blue travel bag, and price lower than 50.00 dollars"},
            {"instruction_text": "red walking shoes"},
        ]
        mapping, unknown = map_runtime_goals(official, runtime)
        self.assertEqual(mapping, {1: 0, 0: 1})
        self.assertEqual(unknown, ())

    def test_unmatched_runtime_goal_is_reported_not_assigned_a_split(self) -> None:
        mapping, unknown = map_runtime_goals(
            ["red walking shoes"],
            [{"instruction_text": "unknown goal"}],
        )
        self.assertEqual(mapping, {})
        self.assertEqual(unknown, ("unknown goal",))


if __name__ == "__main__":
    unittest.main()
