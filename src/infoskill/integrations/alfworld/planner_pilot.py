from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from infoskill.episode import TaskSpec

from .expert_replay import ExpertReplayResult
from .tasks import ALFWORLD_TASK_TYPES


def select_stratified_tasks(
    *,
    tasks: Sequence[TaskSpec],
    tasks_per_type: int,
    selection_seed: int,
) -> tuple[TaskSpec, ...]:
    """Select an order-independent deterministic sample from every task type."""

    if tasks_per_type <= 0:
        raise ValueError("tasks_per_type must be positive")
    by_type: dict[str, list[TaskSpec]] = defaultdict(list)
    for task in tasks:
        if task.split != "train":
            raise ValueError("planner pilot accepts train tasks only")
        by_type[task.task_type].append(task)

    selected: list[TaskSpec] = []
    for task_type in ALFWORLD_TASK_TYPES:
        available = by_type[task_type]
        if len(available) < tasks_per_type:
            raise ValueError(
                f"task type {task_type} requires {tasks_per_type} tasks, "
                f"but only {len(available)} are available"
            )
        ordered = sorted(
            available,
            key=lambda task: (
                hashlib.sha256(
                    f"planner-pilot|{selection_seed}|{task.task_id}".encode()
                ).digest(),
                task.task_id,
            ),
        )
        selected.extend(ordered[:tasks_per_type])
    return tuple(selected)


def build_planner_pilot_report(
    *,
    results: Sequence[tuple[str, ExpertReplayResult]],
    tasks_per_type: int,
    selection_seed: int,
    train_task_manifest_sha256: str,
    code_revision: str,
    max_replay_steps: int,
    persist_horizon: int,
    expert_binding: Mapping[str, object],
    minimum_success_coverage: float = 0.99,
    maximum_over_horizon_rate: float = 0.01,
) -> dict[str, object]:
    if not results:
        raise ValueError("cannot build a planner pilot report without results")
    lengths = [result.total_steps for _, result in results]
    success_count = sum(result.succeeded for _, result in results)
    over_horizon = sum(length > persist_horizon for length in lengths)
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: Counter[str] = Counter()
    for task_type, result in results:
        type_counts[task_type]["total"] += 1
        if result.succeeded:
            type_counts[task_type]["successful"] += 1
        else:
            type_counts[task_type]["quarantined"] += 1
            reasons[result.quarantine_reason or "unknown"] += 1
        if result.total_steps > persist_horizon:
            type_counts[task_type]["over_persist_horizon"] += 1

    observed_counts = Counter(task_type for task_type, _ in results)
    expected_counts = Counter(
        {task_type: tasks_per_type for task_type in ALFWORLD_TASK_TYPES}
    )
    if observed_counts != expected_counts:
        raise ValueError(
            "planner pilot results are not the requested balanced sample: "
            f"expected {dict(expected_counts)!r}, got {dict(observed_counts)!r}"
        )

    total = len(results)
    coverage = success_count / total
    over_rate = over_horizon / total
    handcoded_timeout_signature_count = sum(
        not result.succeeded
        and result.exception_stage == "environment_step"
        and result.exception_type == "Exception"
        and result.exception_message == "Timeout"
        for _, result in results
    )
    identity_failures: list[str] = []
    if expert_binding.get("requested_expert_type") != "planner":
        identity_failures.append("requested_expert_type_is_not_planner")
    if expert_binding.get("effective_expert_type") != "planner":
        identity_failures.append("effective_expert_type_is_not_planner")
    if expert_binding.get("compatibility_guard_active") is not True:
        identity_failures.append("compatibility_guard_is_not_active")
    if expert_binding.get("positional_binding_corrected") is not True:
        identity_failures.append("legacy_positional_binding_was_not_corrected")
    if handcoded_timeout_signature_count:
        identity_failures.append("handcoded_timeout_signature_detected")

    gate_failures = list(identity_failures)
    if coverage < minimum_success_coverage:
        gate_failures.append("success_coverage_below_threshold")
    if over_rate > maximum_over_horizon_rate:
        gate_failures.append("over_horizon_rate_above_threshold")

    per_type: dict[str, dict[str, int | float]] = {}
    for task_type in ALFWORLD_TASK_TYPES:
        counts = type_counts[task_type]
        type_total = counts["total"]
        per_type[task_type] = {
            "total": type_total,
            "successful": counts["successful"],
            "quarantined": counts["quarantined"],
            "success_rate": counts["successful"] / type_total if type_total else 0.0,
            "over_persist_horizon": counts["over_persist_horizon"],
            "over_persist_horizon_rate": (
                counts["over_persist_horizon"] / type_total if type_total else 0.0
            ),
        }

    return {
        "schema_version": 2,
        "pilot_only": True,
        "formal_run_required": True,
        "source_split": "train",
        "selection": {
            "method": "deterministic_stratified_sha256",
            "seed": selection_seed,
            "tasks_per_type": tasks_per_type,
        },
        "selected_games": total,
        "successful_games": success_count,
        "quarantined_games": total - success_count,
        "success_coverage": coverage,
        "over_persist_horizon": over_horizon,
        "over_persist_horizon_rate": over_rate,
        "task_type_counts": per_type,
        "quarantine_reasons": dict(sorted(reasons.items())),
        "expert_binding": dict(expert_binding),
        "expert_identity_gate_passed": not identity_failures,
        "expert_identity_gate_failures": identity_failures,
        "handcoded_timeout_signature_count": handcoded_timeout_signature_count,
        "trajectory_lengths": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / total,
            "median": _percentile(lengths, 0.5),
            "p90": _percentile(lengths, 0.9),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
        },
        "source_checksums": {
            "train_task_manifest": train_task_manifest_sha256,
            "infoskill_source": code_revision,
        },
        "code_revision": code_revision[:16],
        "expert_name": (
            "ALFWorld planner (verified positional binding, strict "
            "admissibility, no fallback)"
        ),
        "max_replay_steps": max_replay_steps,
        "persist_horizon": persist_horizon,
        "pilot_gate_thresholds": {
            "minimum_success_coverage": minimum_success_coverage,
            "maximum_over_horizon_rate": maximum_over_horizon_rate,
        },
        "pilot_gate_passed": not gate_failures,
        "pilot_gate_failures": gate_failures,
    }


def compact_result_payload(
    *,
    task: TaskSpec,
    seed: int,
    result: ExpertReplayResult,
) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "goal": task.goal,
        "seed": seed,
        "succeeded": result.succeeded,
        "total_steps": result.total_steps,
        "reason": result.quarantine_reason,
        "exception_stage": result.exception_stage,
        "exception_type": result.exception_type,
        "exception_message": result.exception_message,
        "action_mismatch": (
            asdict(result.action_mismatch)
            if result.action_mismatch is not None
            else None
        ),
    }


def write_planner_pilot_results(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _percentile(values: Sequence[int], fraction: float) -> float | int:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    value = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return int(value) if value.is_integer() else value
