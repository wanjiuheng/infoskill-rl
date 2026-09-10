from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from infoskill.integrations.alfworld import (
    ExpertReplayResult,
    GroundingDataset,
    build_grounding_manifest,
)


class GroundingManifestTests(unittest.TestCase):
    def test_formal_gate_reports_low_coverage_and_long_replays(self) -> None:
        results = [
            (
                "pick_and_place_simple",
                ExpertReplayResult("ok", True, (), 31, None),
            ),
            (
                "pick_and_place_simple",
                ExpertReplayResult("bad", False, (), 1, "expert_action_not_admissible"),
            ),
        ]

        manifest = build_grounding_manifest(
            results=results,
            source_checksums={"data": "abc"},
            code_revision="test",
            max_replay_steps=150,
            persist_horizon=30,
        )

        self.assertFalse(manifest.formal_gate_passed)
        self.assertEqual(manifest.success_coverage, 0.5)
        self.assertEqual(manifest.over_persist_horizon_rate, 0.5)
        self.assertEqual(manifest.quarantine_reasons["expert_action_not_admissible"], 1)

    def test_dataset_loads_only_a_formal_train_manifest_and_samples_games(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_split": "train",
                        "total_games": 2,
                        "successful_games": 2,
                        "quarantined_games": 0,
                        "success_coverage": 1.0,
                        "over_persist_horizon": 0,
                        "over_persist_horizon_rate": 0.0,
                        "task_type_counts": {},
                        "quarantine_reasons": {},
                        "trajectory_lengths": {"min": 1, "max": 1, "mean": 1, "median": 1},
                        "source_checksums": {"train_task_manifest": "abc"},
                        "code_revision": "test",
                        "expert_name": "test",
                        "max_replay_steps": 150,
                        "persist_horizon": 30,
                        "formal_gate_passed": True,
                        "formal_gate_failures": [],
                    }
                ),
                encoding="utf-8",
            )
            rows = [
                _grounding_row("task-b", step=0, action="look"),
                _grounding_row("task-a", step=0, action="look"),
                _grounding_row("task-a", step=1, action="inventory"),
            ]
            (root / "grounding_samples.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            dataset = GroundingDataset.load(root)
            first = dataset.sample_games(game_count=2, seed=17)
            second = dataset.sample_games(game_count=2, seed=17)

            self.assertEqual(dataset.game_count, 2)
            self.assertEqual(dataset.sample_count, 3)
            self.assertEqual(first, second)
            self.assertEqual({sample.state.task_id for sample in first}, {"task-a", "task-b"})

    def test_dataset_rejects_non_train_state_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "schema_version": 1,
                "source_split": "train",
                "successful_games": 1,
                "formal_gate_passed": True,
                "formal_gate_failures": [],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            row = _grounding_row("task-a", step=0, action="look")
            row["state"]["split"] = "valid_seen"
            (root / "grounding_samples.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "train-only"):
                GroundingDataset.load(root)


def _grounding_row(task_id: str, *, step: int, action: str) -> dict[str, object]:
    commands = ["look", "inventory"]
    return {
        "task_id": task_id,
        "task_type": "pick_and_place_simple",
        "state": {
            "task_id": task_id,
            "split": "train",
            "task_type": "pick_and_place_simple",
            "goal": "put an object somewhere",
            "step_index": step,
            "observation": "You are in a room.",
            "history": [],
            "admissible_commands": commands,
            "done": False,
            "won": False,
            "candidate_skill_ids": ["general-1"],
        },
        "expert_action": action,
    }


if __name__ == "__main__":
    unittest.main()
