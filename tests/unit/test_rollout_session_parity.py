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


class RolloutSessionParityTests(unittest.TestCase):
    def test_small_logprob_drift_with_identical_semantics_passes(self) -> None:
        report = compare_records(
            [_record(logprob=-0.5)],
            [_record(logprob=-0.5001)],
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


if __name__ == "__main__":
    unittest.main()
