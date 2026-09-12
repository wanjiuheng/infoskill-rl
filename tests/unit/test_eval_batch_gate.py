from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infoskill.diagnostics import load_pressure_task_manifest
from infoskill.episode import TaskSpec
from scripts.compare_infoskill_eval_batch_runs import compare_runs


class EvalBatchPressureManifestTests(unittest.TestCase):
    def test_manifest_resolves_exactly_two_tasks_from_each_type(self) -> None:
        task_types = tuple(f"type-{index}" for index in range(6))
        tasks = tuple(
            TaskSpec(
                task_id=f"{task_type}/task-{index}",
                split="valid_seen",
                task_type=task_type,
                goal="goal",
            )
            for task_type in task_types
            for index in range(2)
        )
        payload = {
            "schema_version": 1,
            "split": "valid_seen",
            "full_task_manifest_sha256": "full",
            "tasks": [
                {"task_id": task.task_id, "task_type": task.task_type}
                for task in tasks
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pressure.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            selected, metadata = load_pressure_task_manifest(
                path,
                available_tasks=tasks,
                full_task_manifest_sha256="full",
            )

        self.assertEqual(selected, tasks)
        self.assertEqual(metadata["task_count"], 12)
        self.assertFalse(metadata["reportable_as_valid_seen"])


class EvalBatchGateTests(unittest.TestCase):
    def test_gate_requires_exact_checkpointed_speedy_memory_safe_candidate(
        self,
    ) -> None:
        baseline = Path("/runs/batch8")
        candidate = Path("/runs/batch12")
        record = {
            "task_id": "task-1",
            "rollout_id": 0,
            "steps": [
                {
                    "response_token_ids": [1, 2],
                    "old_token_logprobs": [-0.5, -0.25],
                }
            ],
        }
        with (
            patch(
                "scripts.compare_infoskill_eval_batch_runs._read_json",
                side_effect=(
                    _provenance(batch_size=8),
                    _provenance(batch_size=12),
                    _summary(rollout=120.0, generation=100.0, free=10.0),
                    _summary(rollout=100.0, generation=80.0, free=9.0),
                    _checkpoint_load(),
                    _checkpoint_load(),
                ),
            ),
            patch(
                "scripts.compare_infoskill_eval_batch_runs._read_diagnostic_trace",
                side_effect=([record], [record]),
            ),
        ):
            report = compare_runs(baseline, candidate)

        self.assertTrue(report["settings_valid"])
        self.assertTrue(report["checkpoint_load_valid"])
        self.assertTrue(report["tokens_exact"])
        self.assertTrue(report["physical_memory_valid"])
        self.assertTrue(report["full_140_evaluation_recommended"])
        self.assertTrue(report["passed"])

    def test_gate_rejects_candidate_below_physical_memory_floor(self) -> None:
        with (
            patch(
                "scripts.compare_infoskill_eval_batch_runs._read_json",
                side_effect=(
                    _provenance(batch_size=8),
                    _provenance(batch_size=12),
                    _summary(rollout=120.0, generation=100.0, free=10.0),
                    _summary(rollout=90.0, generation=75.0, free=7.9),
                    _checkpoint_load(),
                    _checkpoint_load(),
                ),
            ),
            patch(
                "scripts.compare_infoskill_eval_batch_runs._read_diagnostic_trace",
                side_effect=([], []),
            ),
        ):
            report = compare_runs(Path("/runs/a"), Path("/runs/b"))

        self.assertFalse(report["physical_memory_valid"])
        self.assertFalse(report["passed"])


def _provenance(*, batch_size: int) -> dict[str, object]:
    return {
        "artifact_kind": "infoskill_eval_batch_pressure_diagnostic",
        "mode": "infoskill",
        "evaluation_manifest": {
            "sha256": "selected",
            "pressure_manifest_sha256": "pressure",
        },
        "policy_model": {"sha256": "model"},
        "skill_conditioning": {"retrieval_plan_sha256": "retrieval"},
        "evaluation_runtime": {
            "num_gpus": 1,
            "environment_backend": "native_batch",
            "persistent_rollout_session": True,
            "grouped_infoskill_conditioning": False,
            "policy_checkpoint": "/checkpoint",
            "checkpoint_step": 2,
            "cuda_memory_poll_interval_ms": 200,
            "eval_batch_size": batch_size,
        },
    }


def _summary(*, rollout: float, generation: float, free: float) -> dict[str, object]:
    return {
        "is_complete": True,
        "evaluated": 12,
        "reportable_as_valid_seen": False,
        "timing_seconds": {"rollout_seconds": rollout},
        "rollout_performance": {
            "perf/rollout_backend_generate_seconds": generation,
            "perf/cuda/rollout_physical_min_free_gb_min": free,
        },
    }


def _checkpoint_load() -> dict[str, object]:
    return {
        "status": "loaded",
        "loaded": True,
        "worker_reports": [
            {
                "rank": 0,
                "lora_state_loaded": True,
                "infoskill_state_loaded": True,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
