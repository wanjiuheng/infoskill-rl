from __future__ import annotations

import unittest
from pathlib import Path

from infoskill.episode import TaskSpec
from infoskill.integrations.alfworld import discover_tasks, task_manifest_sha256


class AlfworldTaskDiscoveryTests(unittest.TestCase):
    def test_only_solvable_supported_games_are_discovered_with_human_goal(self) -> None:
        root = Path(__file__).resolve().parents[1] / "fixtures" / "alfworld_data"
        tasks = discover_tasks(root, split="valid_seen")

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].goal, "Put an apple in the fridge.")
        self.assertEqual(tasks[0].task_type, "pick_and_place_simple")
        self.assertTrue(tasks[0].task_id.endswith("trial_1/game.tw-pddl"))

    def test_task_manifest_checksum_is_stable_and_order_independent(self) -> None:
        tasks = (
            TaskSpec("b", "train", "type-b", "goal b"),
            TaskSpec("a", "train", "type-a", "goal a"),
        )

        expected = "6eab28b527df528c9cdce593fed02f908ef3c7a583db7057a4703c96d5fa91df"
        self.assertEqual(task_manifest_sha256(tasks), expected)
        self.assertEqual(task_manifest_sha256(tuple(reversed(tasks))), expected)


if __name__ == "__main__":
    unittest.main()
