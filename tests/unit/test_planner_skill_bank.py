from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from infoskill.builders import _load_skill_library_provenance
from infoskill.imitation.skill_bank import (
    build_planner_skill_bank,
    rewrite_grounding_skill_ids,
)
from infoskill.skills import FixedSkillLibrary, TemplateRetriever


class PlannerSkillBankTests(unittest.TestCase):
    def test_pick_two_bank_is_phase_complete_and_retrievable(self) -> None:
        rows = [
            {
                "task_id": "two-a",
                "task_type": "pick_two_obj_and_place",
                "state": {"step_index": 0},
                "expert_action": "take apple 1 from table 1",
            },
            {
                "task_id": "two-a",
                "task_type": "pick_two_obj_and_place",
                "state": {"step_index": 1},
                "expert_action": "move apple 1 to bowl 1",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "samples.jsonl"
            destination = Path(temporary) / "skills.json"
            source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            report = build_planner_skill_bank(source, destination)
            provenance = json.loads(
                destination.with_suffix(".manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            library = FixedSkillLibrary.load(destination)
            loaded_provenance = _load_skill_library_provenance(
                library,
                manifest_path=str(destination.with_suffix(".manifest.json")),
            )
            result = TemplateRetriever(library, task_count=8).retrieve(
                "put two apples in the bowl"
            )

        phases = {
            record.fields.get("phase")
            for record in library.task_specific
            if record.category == "pick_two_obj_and_place"
        }
        self.assertEqual(
            phases,
            {"first_object", "first_delivery", "second_object", "second_delivery"},
        )
        self.assertEqual(report["source_successful_trajectories"], 1)
        self.assertEqual(provenance["trajectory_count"], 1)
        self.assertEqual(
            provenance["skill_bank_sha256"], library.source_sha256
        )
        self.assertEqual(loaded_provenance["source_split"], "train")
        self.assertTrue(any("pick_two" in skill_id for skill_id in result.skill_ids))

    def test_grounding_is_derived_with_only_new_candidate_ids(self) -> None:
        row = {
            "task_id": "two-a",
            "task_type": "pick_two_obj_and_place",
            "state": {
                "task_id": "two-a",
                "split": "train",
                "task_type": "pick_two_obj_and_place",
                "goal": "put two apples in bowl",
                "step_index": 0,
                "observation": "room",
                "history": [],
                "admissible_commands": ["take apple 1 from table 1"],
                "candidate_skill_ids": ["old-skill"],
                "done": False,
                "won": False,
            },
            "expert_action": "take apple 1 from table 1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            samples = source / "grounding_samples.jsonl"
            samples.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (source / "manifest.json").write_text(
                json.dumps({
                    "schema_version": 2,
                    "source_split": "train",
                    "successful_games": 1,
                    "expert_type": "planner",
                    "expert_identity_gate_passed": True,
                    "formal_gate_passed": True,
                    "source_checksums": {},
                }),
                encoding="utf-8",
            )
            bank = root / "skills.json"
            build_planner_skill_bank(samples, bank)
            report = rewrite_grounding_skill_ids(source, bank, root / "derived")
            derived = json.loads(
                (root / "derived/grounding_samples.jsonl").read_text()
            )
        ids = derived["state"]["candidate_skill_ids"]
        self.assertNotIn("old-skill", ids)
        self.assertIn("pick_two_first_object", ids)
        self.assertEqual(report["trajectory_count"], 1)


if __name__ == "__main__":
    unittest.main()
