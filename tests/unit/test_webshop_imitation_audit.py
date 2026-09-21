from __future__ import annotations

import unittest
from unittest.mock import patch

from infoskill.imitation.audit import (
    audit_prepared_imitation_data,
    audit_imitation_rows,
    registered_webshop_manifest_failures,
)
from infoskill.integrations.webshop import (
    REGISTERED_HUMAN_DEMONSTRATIONS_SHA256,
    REGISTERED_HUMAN_GOALS_SHA256,
    REGISTERED_TRAIN_TRAJECTORY_COUNT,
)


class _WordTokenizer:
    eos_token_id = 99
    name_or_path = "word-tokenizer"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        del tokenize, add_generation_prompt
        return list(range(len(messages[0]["content"].split()) + 2))

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": list(range(len(text.split())))}


def _prompt(*actions: str) -> str:
    rendered = "\n".join(f"'{action}'," for action in actions)
    return (
        "Your task is to: buy a red shoe.\n"
        "Your admissible actions of the current situation are:\n[\n"
        f"{rendered}\n].\n"
    )


def _row(task_id: str, step: int, action: str, *actions: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "task_type": "webshop",
        "step_index": step,
        "prompt": _prompt(*actions),
        "response": f"<think>advance the demonstrated plan</think>\n<action>{action}</action>",
    }


class WebShopImitationAuditTests(unittest.TestCase):
    def test_registered_manifest_identity_is_accepted(self) -> None:
        failures = registered_webshop_manifest_failures(
            {
                "provider": "OfficialWebShopHumanDemonstrationProvider",
                "source_split": "train",
                "split_method": "deterministic_content_grouped_sha256",
                "trajectory_count": REGISTERED_TRAIN_TRAJECTORY_COUNT,
                "expected_trajectory_count": REGISTERED_TRAIN_TRAJECTORY_COUNT,
                "source_checksums": {
                    "human_demonstrations": REGISTERED_HUMAN_DEMONSTRATIONS_SHA256,
                    "human_goals": REGISTERED_HUMAN_GOALS_SHA256,
                },
            }
        )

        self.assertEqual(failures, [])

    def test_valid_rows_pass_contract_leakage_and_token_gates(self) -> None:
        train = [
            _row("train-a", 0, "search[red shoe]", "search[<your query>]"),
            _row("train-a", 1, "click[ASIN1]", "click[ASIN1]", "click[Back to Search]"),
        ]
        validation = [
            _row("valid-a", 0, "click[Buy Now]", "click[Buy Now]"),
        ]
        report = audit_imitation_rows(
            manifest={
                "environment": "webshop",
                "sample_count": 3,
                "train_sample_count": 2,
                "validation_sample_count": 1,
                "train_trajectory_count": 1,
                "validation_trajectory_count": 1,
                "trajectory_count": 2,
            },
            train_rows=train,
            validation_rows=validation,
            tokenizer=_WordTokenizer(),
            max_length=64,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["failure_reasons"], [])
        self.assertEqual(
            report["action_family_counts"],
            {"click": 1, "purchase": 1, "search": 1},
        )
        self.assertEqual(report["cross_split_trajectory_count"], 0)
        self.assertEqual(report["token_lengths"]["over_limit_count"], 0)

    def test_leakage_and_unavailable_action_fail_the_audit(self) -> None:
        report = audit_imitation_rows(
            manifest={
                "environment": "webshop",
                "sample_count": 2,
                "train_sample_count": 1,
                "validation_sample_count": 1,
                "train_trajectory_count": 1,
                "validation_trajectory_count": 1,
                "trajectory_count": 1,
            },
            train_rows=[
                _row("shared", 0, "click[wrong]", "click[right]"),
            ],
            validation_rows=[
                _row("shared", 0, "click[Buy Now]", "click[Buy Now]"),
            ],
            tokenizer=_WordTokenizer(),
            max_length=64,
        )

        self.assertFalse(report["passed"])
        self.assertIn("cross_split_trajectory_leakage", report["failure_reasons"])
        self.assertIn("demonstrated_action_not_admissible", report["failure_reasons"])

    def test_renamed_duplicate_rows_and_trajectories_fail_closed(self) -> None:
        copied_train = [
            _row("train-copy", 0, "search[red shoe]", "search[<your query>]"),
            _row("train-copy", 1, "click[ASIN1]", "click[ASIN1]"),
        ]
        original_train = [
            {**row, "task_id": "train-original"} for row in copied_train
        ]
        renamed_validation = [
            {**row, "task_id": "valid-renamed"} for row in copied_train
        ]
        report = audit_imitation_rows(
            manifest={
                "environment": "webshop",
                "sample_count": 6,
                "train_sample_count": 4,
                "validation_sample_count": 2,
                "train_trajectory_count": 2,
                "validation_trajectory_count": 1,
                "trajectory_count": 3,
            },
            train_rows=original_train + copied_train,
            validation_rows=renamed_validation,
            tokenizer=_WordTokenizer(),
            max_length=64,
        )

        self.assertFalse(report["passed"])
        self.assertIn("cross_split_content_leakage", report["failure_reasons"])
        self.assertIn("duplicate_row_content", report["warning_reasons"])
        self.assertIn("duplicate_trajectory_content", report["warning_reasons"])
        self.assertGreater(report["duplicate_row_content_group_count"], 0)
        self.assertGreater(report["duplicate_trajectory_content_group_count"], 0)
        self.assertGreater(report["cross_split_content_group_count"], 0)

    def test_same_split_official_duplicates_are_warnings_not_leakage(self) -> None:
        original = [
            _row("train-original", 0, "search[red shoe]", "search[<your query>]"),
            _row("train-original", 1, "click[ASIN1]", "click[ASIN1]"),
        ]
        duplicate = [
            {**row, "task_id": "train-duplicate"} for row in original
        ]
        report = audit_imitation_rows(
            manifest={
                "environment": "webshop",
                "sample_count": 5,
                "train_sample_count": 4,
                "validation_sample_count": 1,
                "train_trajectory_count": 2,
                "validation_trajectory_count": 1,
                "trajectory_count": 3,
            },
            train_rows=original + duplicate,
            validation_rows=[
                _row("valid", 0, "click[Buy Now]", "click[Buy Now]")
            ],
            tokenizer=_WordTokenizer(),
            max_length=64,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["failure_reasons"], [])
        self.assertEqual(
            report["warning_reasons"],
            ["duplicate_row_content", "duplicate_trajectory_content"],
        )

    @patch("infoskill.imitation.audit._atomic_json")
    @patch(
        "infoskill.imitation.audit._read_jsonl",
        side_effect=ValueError("malformed input"),
    )
    @patch("infoskill.imitation.audit.json.loads", return_value={})
    @patch("infoskill.imitation.audit.Path.read_text", return_value="{}")
    def test_malformed_input_returns_and_writes_deterministic_failure_report(
        self,
        _read_text,
        _json_loads,
        _read_jsonl,
        atomic_json,
    ) -> None:
        report = audit_prepared_imitation_data(
            "prepared",
            tokenizer=_WordTokenizer(),
            max_length=64,
            output_path="audit.json",
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["failure_reasons"], ["load_train_failed"])
        self.assertEqual(report["failure_stage"], "load_train")
        self.assertEqual(report["error_type"], "ValueError")
        atomic_json.assert_called_once()


if __name__ == "__main__":
    unittest.main()
