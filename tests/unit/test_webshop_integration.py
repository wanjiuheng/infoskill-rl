from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from infoskill.imitation.dataset import prepare_demonstration_imitation_data
from infoskill.integrations.webshop import (
    OfficialWebShopHumanDemonstrationProvider,
    REGISTERED_HUMAN_DEMONSTRATIONS_SHA256,
    REGISTERED_HUMAN_GOALS_SHA256,
    REGISTERED_TRAIN_TRAJECTORY_COUNT,
    WebShopSplit,
    goal_indices,
    split_for_goal_index,
)


def _demo(goal: str, item: str, *, price: str = "40.00") -> dict[str, object]:
    full_goal = f"{goal}, and price lower than {price} dollars"
    return {
        "states": [
            "Amazon Shopping Game\nInstruction:"
            + full_goal
            + "\n[button] Search [button_]",
            f"Results for red shoes [SEP] {item}",
        ],
        "available_actions": [
            ["search[red walking shoes]"],
            [f"click[{item}]", "click[next >]"],
        ],
        "action_idxs": [-1, 0],
        "actions": ["search[red walking shoes]", f"click[{item}]"],
    }


class WebShopIntegrationTests(unittest.TestCase):
    def test_registered_human_demonstration_snapshot_is_pinned(self) -> None:
        self.assertEqual(REGISTERED_TRAIN_TRAJECTORY_COUNT, 1010)
        self.assertEqual(
            REGISTERED_HUMAN_DEMONSTRATIONS_SHA256,
            "0f3ef1890245a283f8116b7abcabebd4acdf355d773edd99977e8ed6de63ec6c",
        )
        self.assertEqual(
            REGISTERED_HUMAN_GOALS_SHA256,
            "b68746ed66cd31fdc5f70eb3f5831a46b38163563b57a66ddcab8fd60ee0cbdc",
        )

    def test_official_goal_splits_are_disjoint_and_complete(self) -> None:
        goal_count = 12087
        test = set(goal_indices("test", goal_count=goal_count))
        validation = set(goal_indices("eval", goal_count=goal_count))
        train = set(goal_indices("train", goal_count=goal_count))
        self.assertEqual((len(test), len(validation), len(train)), (500, 1000, 10587))
        self.assertFalse(test & validation)
        self.assertFalse(test & train)
        self.assertFalse(validation & train)
        self.assertEqual(test | validation | train, set(range(goal_count)))
        self.assertIs(
            split_for_goal_index(1499, goal_count=goal_count),
            WebShopSplit.VALIDATION,
        )
        self.assertIs(
            split_for_goal_index(1500, goal_count=goal_count),
            WebShopSplit.TRAIN,
        )

    def test_official_provider_filters_to_train_and_matches_online_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            goals = [f"unused goal {index}" for index in range(1503)]
            goals[700] = "validation red shoe"
            goals[1500] = "train red shoe"
            goals_path = root / "human_goals.json"
            goals_path.write_text(json.dumps(goals), encoding="utf-8")
            demos_path = root / "demos.jsonl"
            demos_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        _demo(goals[700], "validation item"),
                        _demo(goals[1500], "train item"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            trajectories = OfficialWebShopHumanDemonstrationProvider(
                demos_path,
                goals_path,
            ).trajectories()

            self.assertEqual(len(trajectories), 1)
            trajectory = trajectories[0]
            self.assertIn("goal-01500", trajectory.trajectory_id)
            self.assertEqual(trajectory.steps[0].action, "search[red walking shoes]")
            self.assertIn("search[<your query>]", trajectory.steps[0].prompt)
            self.assertIn("price lower than 40.00 dollars", trajectory.steps[0].prompt)
            self.assertIn(
                "current observation is: [button] Search [button_]",
                trajectory.steps[0].prompt,
            )
            self.assertIn("Prior to this step", trajectory.steps[1].prompt)
            self.assertIn("Action 1: 'search[red walking shoes]'", trajectory.steps[1].prompt)
            self.assertNotIn("validation red shoe", trajectory.steps[0].prompt)

    def test_webshop_imitation_preparation_splits_whole_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            goals = [f"unused goal {index}" for index in range(1502)]
            goals[1500] = "train red shoe"
            goals[1501] = "train blue shoe"
            goals_path = root / "human_goals.json"
            goals_path.write_text(json.dumps(goals), encoding="utf-8")
            demos_path = root / "demos.jsonl"
            demos_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        _demo(goals[1500], "red item"),
                        _demo(goals[1501], "blue item"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            provider = OfficialWebShopHumanDemonstrationProvider(
                demos_path,
                goals_path,
            )

            manifest = prepare_demonstration_imitation_data(
                provider=provider,
                output_directory=root / "prepared",
                validation_fraction=0.5,
                split_seed=7,
                expected_trajectory_count=2,
            )

            train = [
                json.loads(line)
                for line in (root / "prepared/train.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            validation = [
                json.loads(line)
                for line in (root / "prepared/validation.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertFalse(
                {row["task_id"] for row in train}
                & {row["task_id"] for row in validation}
            )
            self.assertEqual(manifest["trajectory_count"], 2)
            self.assertEqual(manifest["sample_count"], 4)
            self.assertEqual(manifest["source_split"], "train")
            self.assertEqual(
                set(manifest["source_checksums"]),
                {"human_demonstrations", "human_goals"},
            )
            for row in train + validation:
                self.assertRegex(
                    row["response"],
                    r"^<think>.+</think>\n<action>.+</action>$",
                )

    def test_webshop_preparation_rejects_unregistered_source_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            goals = [f"unused goal {index}" for index in range(1502)]
            goals[1500] = "train red shoe"
            goals[1501] = "train blue shoe"
            goals_path = root / "human_goals.json"
            goals_path.write_text(json.dumps(goals), encoding="utf-8")
            demos_path = root / "demos.jsonl"
            demos_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        _demo(goals[1500], "red item"),
                        _demo(goals[1501], "blue item"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "human_demonstrations expected 0+",
            ):
                prepare_demonstration_imitation_data(
                    provider=OfficialWebShopHumanDemonstrationProvider(
                        demos_path,
                        goals_path,
                    ),
                    output_directory=root / "prepared",
                    validation_fraction=0.5,
                    expected_source_checksums={
                        "human_demonstrations": "0" * 64,
                    },
                )
            self.assertFalse((root / "prepared").exists())


if __name__ == "__main__":
    unittest.main()
