from __future__ import annotations

import unittest

from infoskill.integrations.alfworld import (
    ExpertReplayResult,
    GroundingShardReport,
    build_grounding_parity_report,
)


def _lifecycle(*, concurrency: int, peak: int) -> GroundingShardReport:
    return GroundingShardReport(
        schema_version=2,
        expert_type="planner",
        worker_batch_size=2,
        worker_concurrency=concurrency,
        peak_worker_processes=peak,
        worker_processes_started=2,
        processed_tasks=2,
        minimum_free_disk_bytes=20 * 1024**3,
        parallel_worker_disk_reserve_bytes=3 * 1024**3 if concurrency > 1 else 0,
        temporary_directories_cleaned=True,
        shards=(),
        peak_environment_slots=peak,
    )


class GroundingParityTests(unittest.TestCase):
    def test_exact_results_and_observed_parallelism_pass(self) -> None:
        results = [
            (
                "pick_and_place_simple",
                ExpertReplayResult(f"task-{index}", True, (), index + 1, None),
            )
            for index in range(2)
        ]
        report = build_grounding_parity_report(
            serial_results=results,
            parallel_results=list(results),
            serial_lifecycle=_lifecycle(concurrency=1, peak=1),
            parallel_lifecycle=_lifecycle(concurrency=2, peak=2),
            serial_seconds=10.0,
            parallel_seconds=6.0,
            tasks_per_type=2,
            selection_seed=0,
            train_task_manifest_sha256="manifest",
            code_revision="revision",
            expert_binding={
                "requested_expert_type": "planner",
                "effective_expert_type": "planner",
                "compatibility_guard_active": True,
                "positional_binding_corrected": True,
            },
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["mismatch_count"], 0)
        self.assertAlmostEqual(report["speedup"], 10.0 / 6.0)

    def test_different_trajectory_result_fails_with_field_name(self) -> None:
        serial = [
            (
                "pick_and_place_simple",
                ExpertReplayResult("task-0", True, (), 5, None),
            ),
            (
                "pick_and_place_simple",
                ExpertReplayResult("task-1", True, (), 5, None),
            ),
        ]
        parallel = [
            serial[0],
            (
                "pick_and_place_simple",
                ExpertReplayResult("task-1", True, (), 6, None),
            ),
        ]
        report = build_grounding_parity_report(
            serial_results=serial,
            parallel_results=parallel,
            serial_lifecycle=_lifecycle(concurrency=1, peak=1),
            parallel_lifecycle=_lifecycle(concurrency=2, peak=2),
            serial_seconds=10.0,
            parallel_seconds=6.0,
            tasks_per_type=2,
            selection_seed=0,
            train_task_manifest_sha256="manifest",
            code_revision="revision",
            expert_binding={
                "requested_expert_type": "planner",
                "effective_expert_type": "planner",
                "compatibility_guard_active": True,
                "positional_binding_corrected": True,
            },
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["field_checks"]["total_steps"])
        self.assertEqual(report["mismatch_count"], 1)
        self.assertEqual(
            report["mismatches"][0]["different_fields"],
            ["total_steps"],
        )

    def test_exact_candidate_still_fails_when_speedup_gate_is_missed(self) -> None:
        results = [
            (
                "pick_and_place_simple",
                ExpertReplayResult("task-0", True, (), 5, None),
            ),
            (
                "pick_and_place_simple",
                ExpertReplayResult("task-1", True, (), 5, None),
            ),
        ]
        report = build_grounding_parity_report(
            serial_results=results,
            parallel_results=list(results),
            serial_lifecycle=_lifecycle(concurrency=1, peak=1),
            parallel_lifecycle=_lifecycle(concurrency=1, peak=4),
            serial_seconds=10.0,
            parallel_seconds=9.8,
            tasks_per_type=2,
            selection_seed=0,
            train_task_manifest_sha256="manifest",
            code_revision="revision",
            expert_binding={
                "requested_expert_type": "planner",
                "effective_expert_type": "planner",
                "compatibility_guard_active": True,
                "positional_binding_corrected": True,
            },
            minimum_speedup=1.05,
        )

        self.assertTrue(all(report["field_checks"].values()))
        self.assertFalse(report["performance_passed"])
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
