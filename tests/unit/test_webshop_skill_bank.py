from __future__ import annotations

import json
import unittest

from infoskill.imitation.audit import validate_webshop_audit_report
from infoskill.imitation.skill_bank import (
    derive_webshop_skill_bank,
)


class WebShopSkillBankTests(unittest.TestCase):
    def test_skill_bank_requires_passed_matching_audit(self) -> None:
        checksums = {
            "manifest": "a" * 64,
            "train": "b" * 64,
            "validation": "c" * 64,
        }
        valid = {
            "passed": True,
            "data_directory": "/data/prepared",
            "source_checksums": checksums,
        }

        validate_webshop_audit_report(
            valid,
            data_directory="/data/prepared",
            source_checksums=checksums,
        )

        with self.assertRaisesRegex(ValueError, "did not pass"):
            validate_webshop_audit_report(
                {**valid, "passed": False},
                data_directory="/data/prepared",
                source_checksums=checksums,
            )
        with self.assertRaisesRegex(ValueError, "checksums"):
            validate_webshop_audit_report(
                valid,
                data_directory="/data/prepared",
                source_checksums={**checksums, "train": "d" * 64},
            )

    def test_demonstrations_produce_phase_complete_generalizable_bank(self) -> None:
        rows = [
            {"task_id": "a", "response": "<think>x</think>\n<action>search[red shoe]</action>"},
            {"task_id": "a", "response": "<think>x</think>\n<action>click[ASIN1]</action>"},
            {"task_id": "a", "response": "<think>x</think>\n<action>click[large]</action>"},
            {"task_id": "a", "response": "<think>x</think>\n<action>click[Buy Now]</action>"},
        ]
        payload, provenance = derive_webshop_skill_bank(
            rows,
            source_samples_sha256="a" * 64,
            source_trajectory_count=1,
        )

        skills = payload["task_specific_skills"]["webshop"]
        self.assertEqual(
            {item["phase"] for item in skills},
            {
                "query",
                "results",
                "product_verification",
                "option_selection",
                "backtracking",
                "purchase",
            },
        )
        self.assertEqual(
            payload["metadata"]["action_family_counts"],
            {"click": 2, "purchase": 1, "search": 1},
        )
        self.assertFalse(payload["metadata"]["category_gate"])
        self.assertEqual(payload["metadata"]["total_memories_analyzed"], 1)
        self.assertEqual(provenance["source_split"], "train")
        self.assertEqual(provenance["environment"], "webshop")
        self.assertEqual(len(provenance["skill_bank_sha256"]), 64)
        serialized = json.dumps(payload, sort_keys=True)
        self.assertNotIn("red shoe", serialized)
        self.assertNotIn("ASIN1", serialized)


if __name__ == "__main__":
    unittest.main()
