from __future__ import annotations

import unittest

from scripts.compare_rollout_session_runs import compare_records


def _record(*, token: int = 7, logprob: float = -0.5) -> dict:
    return {
        "task_id": "task-1",
        "rollout_id": 0,
        "won": True,
        "steps": [
            {
                "model_raw_response": "<action>look</action>",
                "response_token_ids": [token],
                "old_token_logprobs": [logprob],
                "action_resolution": {"executed_action": "look"},
            }
        ],
    }


def _record_with_facts(facts: list[str]) -> dict:
    record = _record()
    record["steps"][0]["environment_raw_output"] = {
        "observation": "same",
        "info": {"facts": facts},
    }
    return record


class RolloutSessionParityTests(unittest.TestCase):
    def test_small_logprob_drift_with_identical_semantics_passes(self) -> None:
        report = compare_records(
            [_record(logprob=-0.5)],
            [_record(logprob=-0.5001)],
            logprob_tolerance=1e-3,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["semantic_exact"])

    def test_unused_environment_expert_plan_does_not_create_a_false_mismatch(
        self,
    ) -> None:
        baseline = _record()
        optimized = _record()
        baseline["steps"][0]["environment_raw_output"] = {
            "observation": "same",
            "info": {
                "facts": ["same fact"],
                "extra.expert_plan": ["take mug 1 from countertop 1"],
            },
        }
        optimized["steps"][0]["environment_raw_output"] = {
            "observation": "same",
            "info": {
                "facts": ["same fact"],
                "extra.expert_plan": ["take mug 2 from countertop 1"],
            },
        }

        report = compare_records(
            [baseline],
            [optimized],
            logprob_tolerance=1e-3,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["semantic_exact"])

    def test_token_difference_is_a_semantic_failure(self) -> None:
        report = compare_records(
            [_record(token=7)],
            [_record(token=8)],
            logprob_tolerance=1e-3,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["semantic_exact"])
        self.assertIn(
            "$.steps[0].response_token_ids[0]: 7 != 8",
            report["semantic_mismatches"][0],
        )

    def test_unordered_environment_facts_do_not_create_a_false_mismatch(self) -> None:
        report = compare_records(
            [_record_with_facts(["a", "b"])],
            [_record_with_facts(["b", "a"])],
            logprob_tolerance=1e-3,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["semantic_exact"])


if __name__ == "__main__":
    unittest.main()
