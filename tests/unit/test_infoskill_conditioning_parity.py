from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compare_infoskill_conditioning_runs import compare_runs


class InfoSkillConditioningParityTests(unittest.TestCase):
    def test_gate_requires_controlled_settings_exact_trace_and_speedup(self) -> None:
        baseline = Path("/runs/baseline")
        optimized = Path("/runs/optimized")
        record = {
            "task_id": "task-1",
            "rollout_id": 0,
            "steps": [{"old_token_logprobs": [-0.5]}],
        }

        with (
            patch(
                "scripts.compare_infoskill_conditioning_runs._read_json",
                side_effect=(
                    _provenance(grouped=False),
                    _provenance(grouped=True),
                    _summary(100.0),
                    _summary(80.0),
                ),
            ),
            patch(
                "scripts.compare_infoskill_conditioning_runs._read_evaluation_traces",
                side_effect=([record], [record]),
            ),
        ):
            report = compare_runs(baseline, optimized)

        self.assertTrue(report["settings_valid"])
        self.assertTrue(report["semantic_exact"])
        self.assertTrue(report["logprobs_close"])
        self.assertTrue(report["performance_valid"])
        self.assertTrue(report["passed"])

    def test_gate_rejects_candidate_without_explicit_grouping(self) -> None:
        with (
            patch(
                "scripts.compare_infoskill_conditioning_runs._read_json",
                side_effect=(
                    _provenance(grouped=False),
                    _provenance(grouped=False),
                    _summary(100.0),
                    _summary(80.0),
                ),
            ),
            patch(
                "scripts.compare_infoskill_conditioning_runs._read_evaluation_traces",
                side_effect=([], []),
            ),
        ):
            report = compare_runs(Path("/runs/baseline"), Path("/runs/optimized"))

        self.assertFalse(report["settings_valid"])
        self.assertFalse(report["passed"])


def _provenance(*, grouped: bool) -> dict[str, object]:
    return {
        "mode": "infoskill",
        "evaluation_manifest": {"sha256": "manifest", "task_count": 140},
        "policy_model": {"sha256": "model"},
        "evaluation_runtime": {
            "checkpoint_step": 0,
            "policy_checkpoint": None,
            "num_gpus": 3,
            "environment_backend": "native_batch",
            "persistent_rollout_session": True,
            "grouped_infoskill_conditioning": grouped,
        },
        "skill_conditioning": {"retrieval_plan_sha256": "retrieval"},
    }


def _summary(rollout_seconds: float) -> dict[str, object]:
    return {
        "timing_seconds": {"rollout_seconds": rollout_seconds},
        "rollout_performance": {},
    }


if __name__ == "__main__":
    unittest.main()
