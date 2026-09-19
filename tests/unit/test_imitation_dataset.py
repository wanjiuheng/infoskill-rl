from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from infoskill.imitation.dataset import prepare_alfworld_imitation_data


def _row(task_id: str, task_type: str, step: int, action: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "task_type": task_type,
        "state": {
            "task_id": task_id,
            "split": "train",
            "task_type": task_type,
            "goal": "put two apple in bowl",
            "step_index": step,
            "observation": "You see an apple and a bowl.",
            "history": [],
            "admissible_commands": [action, "look"],
            "candidate_skill_ids": ["legacy"],
            "done": False,
            "won": False,
        },
        "expert_action": action,
    }


class ImitationDatasetTests(unittest.TestCase):
    def test_preparation_splits_by_trajectory_and_emits_maskable_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "grounding"
            source.mkdir()
            (source / "manifest.json").write_text(
                json.dumps({
                    "schema_version": 2,
                    "source_split": "train",
                    "successful_games": 2,
                    "expert_type": "planner",
                    "expert_identity_gate_passed": True,
                    "formal_gate_passed": True,
                }),
                encoding="utf-8",
            )
            rows = [
                _row("task-a", "pick_two_obj_and_place", 0, "take apple 1 from table 1"),
                _row("task-a", "pick_two_obj_and_place", 1, "move apple 1 to bowl 1"),
                _row("task-b", "pick_and_place_simple", 0, "take mug 1 from desk 1"),
            ]
            (source / "grounding_samples.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            manifest = prepare_alfworld_imitation_data(
                grounding_directory=source,
                output_directory=root / "prepared",
                validation_fraction=0.5,
                split_seed=7,
            )

            train = [json.loads(line) for line in (root / "prepared/train.jsonl").read_text().splitlines()]
            valid = [json.loads(line) for line in (root / "prepared/validation.jsonl").read_text().splitlines()]
            train_ids = {row["task_id"] for row in train}
            valid_ids = {row["task_id"] for row in valid}
            self.assertFalse(train_ids & valid_ids)
            self.assertEqual(manifest["trajectory_count"], 2)
            self.assertEqual(manifest["sample_count"], 3)
            for row in train + valid:
                self.assertIn("Your task is to:", row["prompt"])
                self.assertRegex(row["response"], r"^<think>.+</think>\n<action>.+</action>$")


if __name__ == "__main__":
    unittest.main()
