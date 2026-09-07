from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from infoskill.evaluation.artifacts import (
    checkpoint_load_payload,
    write_evaluation_provenance,
    write_evaluation_timing,
)


class EvaluationArtifactTests(unittest.TestCase):
    def test_evaluation_provenance_is_written_with_checkpoint_identity(self) -> None:
        run_directory = Path("/runs/evaluation")
        with patch(
            "infoskill.evaluation.artifacts._write_json_atomic"
        ) as write_json:
            payload = write_evaluation_provenance(
                run_directory,
                mode="raw_skill_prompt",
                evaluation_runtime={
                    "backend": "verl",
                    "checkpoint_step": 1,
                    "policy_checkpoint": "/runs/raw/checkpoints/step-000001",
                },
                evaluation_manifest={
                    "split": "valid_seen",
                    "task_count": 140,
                    "sha256": "manifest-sha",
                },
                policy_model={"model_id": "alfworld-7b-sft"},
                skill_conditioning={"retrieval_mode": "embedding"},
                checkpoint_provenance_sha256="checkpoint-provenance-sha",
            )

        write_json.assert_called_once_with(
            run_directory / "provenance.json",
            payload,
        )
        self.assertEqual(payload["artifact_kind"], "valid_seen_evaluation")
        self.assertEqual(payload["mode"], "raw_skill_prompt")
        self.assertEqual(payload["evaluation_manifest"]["task_count"], 140)
        self.assertEqual(
            payload["checkpoint_provenance_sha256"],
            "checkpoint-provenance-sha",
        )

    def test_checkpoint_load_payload_distinguishes_loaded_and_not_requested(self) -> None:
        not_requested = checkpoint_load_payload(
            backend="verl",
            checkpoint=None,
            checkpoint_step=0,
            status="not_requested",
        )
        self.assertFalse(not_requested["requested"])
        self.assertFalse(not_requested["loaded"])
        self.assertEqual(not_requested["status"], "not_requested")

        loaded = checkpoint_load_payload(
            backend="verl",
            checkpoint="/runs/raw/checkpoints/step-000001",
            checkpoint_step=1,
            status="loaded",
            duration_seconds=2.5,
            worker_reports=(
                {
                    "rank": 0,
                    "base_sync_done_after": True,
                    "warmup_performed": True,
                },
            ),
        )
        self.assertTrue(loaded["requested"])
        self.assertTrue(loaded["loaded"])
        self.assertEqual(loaded["worker_reports"][0]["rank"], 0)
        self.assertEqual(loaded["duration_seconds"], 2.5)

    def test_failed_checkpoint_load_records_error_without_claiming_success(self) -> None:
        payload = checkpoint_load_payload(
            backend="verl",
            checkpoint="/runs/raw/checkpoints/step-000001",
            checkpoint_step=1,
            status="failed",
            duration_seconds=1.25,
            error=RuntimeError("state mismatch"),
        )

        self.assertFalse(payload["loaded"])
        self.assertEqual(payload["error_type"], "RuntimeError")
        self.assertEqual(payload["error_message"], "state mismatch")

    def test_evaluation_timing_rejects_negative_durations(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            write_evaluation_timing(
                Path("/runs/evaluation"),
                {"rollout_seconds": -0.1},
            )


if __name__ == "__main__":
    unittest.main()
