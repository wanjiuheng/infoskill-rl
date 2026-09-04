from __future__ import annotations

import unittest
from pathlib import Path

from infoskill.integrations.alfworld import discover_tasks


class AlfworldTaskDiscoveryTests(unittest.TestCase):
    def test_only_solvable_supported_games_are_discovered_with_human_goal(self) -> None:
        root = Path(__file__).resolve().parents[1] / "fixtures" / "alfworld_data"
        tasks = discover_tasks(root, split="valid_seen")

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].goal, "Put an apple in the fridge.")
        self.assertEqual(tasks[0].task_type, "pick_and_place_simple")
        self.assertTrue(tasks[0].task_id.endswith("trial_1/game.tw-pddl"))


if __name__ == "__main__":
    unittest.main()
