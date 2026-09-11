from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from .expert_replay import ExpertReplayResult
from .grounding_io import grounding_result_payload
from .grounding_shards import GroundingShardReport


_RESULT_FIELDS = (
    "task_type",
    "task_id",
    "succeeded",
    "samples",
    "total_steps",
    "quarantine_reason",
    "exception_stage",
    "exception_type",
    "exception_message",
    "action_mismatch",
)


def serialize_grounding_results(
    results: Sequence[tuple[str, ExpertReplayResult]],
) -> tuple[dict[str, object], ...]:
    return tuple(
        grounding_result_payload(task_type, result)
        for task_type, result in results
    )


def build_grounding_parity_report(
    *,
    serial_results: Sequence[tuple[str, ExpertReplayResult]],
    parallel_results: Sequence[tuple[str, ExpertReplayResult]],
    serial_lifecycle: GroundingShardReport,
    parallel_lifecycle: GroundingShardReport,
    serial_seconds: float,
    parallel_seconds: float,
    tasks_per_type: int,
    selection_seed: int,
    train_task_manifest_sha256: str,
    code_revision: str,
    expert_binding: Mapping[str, object],
    minimum_speedup: float = 0.0,
) -> dict[str, object]:
    serial_rows = serialize_grounding_results(serial_results)
    parallel_rows = serialize_grounding_results(parallel_results)
    checks = {
        field: len(serial_rows) == len(parallel_rows)
        and all(
            serial.get(field) == parallel.get(field)
            for serial, parallel in zip(serial_rows, parallel_rows)
        )
        for field in _RESULT_FIELDS
    }
    task_order_exact = [row["task_id"] for row in serial_rows] == [
        row["task_id"] for row in parallel_rows
    ]
    checks["task_order"] = task_order_exact
    mismatches = []
    for index, (serial, parallel) in enumerate(zip(serial_rows, parallel_rows)):
        different = [
            field for field in _RESULT_FIELDS if serial.get(field) != parallel.get(field)
        ]
        if different:
            mismatches.append(
                {
                    "index": index,
                    "serial_task_id": serial.get("task_id"),
                    "parallel_task_id": parallel.get("task_id"),
                    "different_fields": different,
                }
            )
    if len(serial_rows) != len(parallel_rows):
        mismatches.append(
            {
                "serial_result_count": len(serial_rows),
                "parallel_result_count": len(parallel_rows),
                "different_fields": ["result_count"],
            }
        )

    lifecycle_checks = {
        "serial_processed_all_tasks": (
            serial_lifecycle.processed_tasks == len(serial_rows)
        ),
        "parallel_processed_all_tasks": (
            parallel_lifecycle.processed_tasks == len(parallel_rows)
        ),
        "serial_temporary_directories_cleaned": (
            serial_lifecycle.temporary_directories_cleaned
        ),
        "parallel_temporary_directories_cleaned": (
            parallel_lifecycle.temporary_directories_cleaned
        ),
        "serial_expert_is_planner": serial_lifecycle.expert_type == "planner",
        "parallel_expert_is_planner": parallel_lifecycle.expert_type == "planner",
        "parallelism_observed": parallel_lifecycle.peak_environment_slots > 1,
    }
    identity_checks = {
        "requested_expert_is_planner": (
            expert_binding.get("requested_expert_type") == "planner"
        ),
        "effective_expert_is_planner": (
            expert_binding.get("effective_expert_type") == "planner"
        ),
        "compatibility_guard_active": (
            expert_binding.get("compatibility_guard_active") is True
        ),
        "positional_binding_corrected": (
            expert_binding.get("positional_binding_corrected") is True
        ),
    }
    speedup = serial_seconds / parallel_seconds if parallel_seconds > 0 else None
    performance_passed = speedup is not None and speedup >= minimum_speedup
    passed = (
        all(checks.values())
        and all(lifecycle_checks.values())
        and all(identity_checks.values())
        and performance_passed
    )
    return {
        "schema_version": 2,
        "passed": passed,
        "comparison_scope": (
            "full serialized replay rows, including every persisted state, "
            "admissible command list, candidate skill ID list, expert action, "
            "terminal result, exception, and action mismatch"
        ),
        "selection": {
            "method": "deterministic_stratified_sha256",
            "seed": selection_seed,
            "tasks_per_type": tasks_per_type,
            "selected_games": len(serial_rows),
        },
        "field_checks": checks,
        "lifecycle_checks": lifecycle_checks,
        "identity_checks": identity_checks,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
        "minimum_speedup": minimum_speedup,
        "performance_passed": performance_passed,
        "serial": {
            "seconds": serial_seconds,
            "worker_concurrency": serial_lifecycle.worker_concurrency,
            "peak_worker_processes": serial_lifecycle.peak_worker_processes,
            "minimum_free_disk_bytes": serial_lifecycle.minimum_free_disk_bytes,
            "replay_backend": serial_lifecycle.replay_backend,
            "native_batch_size": serial_lifecycle.native_batch_size,
            "peak_environment_slots": serial_lifecycle.peak_environment_slots,
        },
        "parallel": {
            "seconds": parallel_seconds,
            "worker_concurrency": parallel_lifecycle.worker_concurrency,
            "peak_worker_processes": parallel_lifecycle.peak_worker_processes,
            "minimum_free_disk_bytes": parallel_lifecycle.minimum_free_disk_bytes,
            "replay_backend": parallel_lifecycle.replay_backend,
            "native_batch_size": parallel_lifecycle.native_batch_size,
            "peak_environment_slots": parallel_lifecycle.peak_environment_slots,
        },
        "speedup": speedup,
        "expert_binding": dict(expert_binding),
        "source_checksums": {
            "train_task_manifest": train_task_manifest_sha256,
            "infoskill_source": code_revision,
        },
        "code_revision": code_revision[:16],
    }


def write_serialized_grounding_results(
    path: str | Path,
    results: Sequence[tuple[str, ExpertReplayResult]],
) -> None:
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in serialize_grounding_results(results)
        ),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
