from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from infoskill.integrations.webshop.online_smoke import (
    _page_goal_text,
    map_runtime_goals,
    run_smoke,
)


class WebShopOnlineSmokeTests(unittest.TestCase):
    def test_page_instruction_heading_does_not_mask_wrong_goal(self) -> None:
        self.assertEqual(_page_goal_text("Instruction: blue travel bag"), "blue travel bag")
        self.assertNotEqual(_page_goal_text("Instruction: blue travel bag"), "red shoes")

    def test_page_instruction_label_does_not_change_goal(self) -> None:
        official = [f"goal {index}" for index in range(1501)]
        runtime_goals = [
            {"instruction_text": official[index], "query": "product"}
            for index in (0, 500, 1500)
        ]

        class FakeEnvironment:
            server = type("Server", (), {"goals": runtime_goals})()

            def reset(self, session: int) -> tuple[str, None]:
                self.instruction_text = (
                    "Instruction: " + runtime_goals[session]["instruction_text"]
                )
                return "initial observation", None

            def get_available_actions(self) -> dict[str, object]:
                return {"has_search_bar": True, "clickables": ["item"]}

            def step(self, action: str) -> tuple[str, float, bool, dict[str, object]]:
                return "search results", 0.0, False, {}

            def close(self) -> None:
                pass

        goals_path = Path("human_goals.json")
        with patch.object(Path, "read_text", return_value=json.dumps(official)), patch(
            "infoskill.integrations.webshop.online_smoke._required_assets",
            return_value={"human_goals": goals_path, "index_manifest": goals_path},
        ), patch(
            "infoskill.integrations.webshop.online_smoke._open_external_text_env",
            return_value=FakeEnvironment(),
        ):
            report = run_smoke(Path.cwd(), Path.cwd())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(set(report["examples"]), {"test", "validation", "train"})

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
