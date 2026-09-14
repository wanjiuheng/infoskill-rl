#!/usr/bin/env python3
"""Compare behavior-changing INFO-SKILL candidates on a fixed evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALLOWED_EXECUTION_RUNTIME_DIFFERENCES = (
    "hybrid_prefix_cuda_graph",
    "hybrid_prefix_cuda_graph_custom_kernels",
    "hybrid_prefix_cuda_graph_use_inductor",
)


def compare_evaluations(
    baseline: Path,
    candidate: Path,
    *,
    expected_task_count: int = 140,
    equality_tolerance: float = 1e-12,
    allowed_runtime_differences: tuple[str, ...] = (),
    allow_eval_batch_size_difference: bool = False,
    require_same_checkpoint: bool = False,
) -> dict[str, object]:
    baseline_summary = _read_json(baseline / "valid_seen_summary.json")
    candidate_summary = _read_json(candidate / "valid_seen_summary.json")
    baseline_resolved = _read_json(baseline / "resolved_config.json")
    candidate_resolved = _read_json(candidate / "resolved_config.json")
    baseline_load = _read_json(baseline / "checkpoint-load.json")
    candidate_load = _read_json(candidate / "checkpoint-load.json")

    baseline_per_type = _per_type(baseline_summary)
    candidate_per_type = _per_type(candidate_summary)
    controls = {
        "both_complete": (
            baseline_summary.get("is_complete") is True
            and candidate_summary.get("is_complete") is True
        ),
        "expected_task_count": (
            baseline_summary.get("evaluated") == expected_task_count
            and candidate_summary.get("evaluated") == expected_task_count
        ),
        "same_task_manifest": (
            baseline_summary.get("task_manifest_sha256")
            == candidate_summary.get("task_manifest_sha256")
            and bool(baseline_summary.get("task_manifest_sha256"))
        ),
        "same_task_types": set(baseline_per_type) == set(candidate_per_type),
        "same_evaluation_protocol": (
            _without_checkpoint(
                baseline_resolved,
                allowed_runtime_differences=allowed_runtime_differences,
                allow_eval_batch_size_difference=(
                    allow_eval_batch_size_difference
                ),
            )
            == _without_checkpoint(
                candidate_resolved,
                allowed_runtime_differences=allowed_runtime_differences,
                allow_eval_batch_size_difference=(
                    allow_eval_batch_size_difference
                ),
            )
        ),
        "both_checkpoints_loaded": (
            _checkpoint_loaded(baseline_load)
            and _checkpoint_loaded(candidate_load)
        ),
        "same_checkpoint_step": (
            baseline_load.get("checkpoint_step")
            == candidate_load.get("checkpoint_step")
            and isinstance(baseline_load.get("checkpoint_step"), int)
        ),
        "same_checkpoint": (
            not require_same_checkpoint
            or (
                baseline_load.get("checkpoint")
                == candidate_load.get("checkpoint")
                and isinstance(baseline_load.get("checkpoint"), str)
                and bool(baseline_load.get("checkpoint"))
            )
        ),
    }
    controls_valid = all(controls.values())

    baseline_macro = _finite_float(baseline_summary.get("macro_success"))
    candidate_macro = _finite_float(candidate_summary.get("macro_success"))
    baseline_overall = _finite_float(baseline_summary.get("overall_success"))
    candidate_overall = _finite_float(candidate_summary.get("overall_success"))
    macro_delta = candidate_macro - baseline_macro
    overall_delta = candidate_overall - baseline_overall

    if macro_delta > equality_tolerance:
        ordering = "candidate_better_on_macro"
        efficacy_noninferior = True
    elif macro_delta < -equality_tolerance:
        ordering = "candidate_worse_on_macro"
        efficacy_noninferior = False
    elif overall_delta > equality_tolerance:
        ordering = "macro_tied_candidate_better_on_overall"
        efficacy_noninferior = True
    elif overall_delta < -equality_tolerance:
        ordering = "macro_tied_candidate_worse_on_overall"
        efficacy_noninferior = False
    else:
        ordering = "macro_and_overall_tied"
        efficacy_noninferior = True

    per_task_type = {
        task_type: {
            "baseline": baseline_per_type[task_type],
            "candidate": candidate_per_type[task_type],
            "delta": candidate_per_type[task_type]
            - baseline_per_type[task_type],
        }
        for task_type in sorted(set(baseline_per_type) & set(candidate_per_type))
    }
    return {
        "schema_version": 1,
        "decision_rule": "macro_success_primary_overall_success_secondary",
        "allowed_runtime_differences": list(allowed_runtime_differences),
        "allow_eval_batch_size_difference": allow_eval_batch_size_difference,
        "require_same_checkpoint": require_same_checkpoint,
        "control_checks": controls,
        "controls_valid": controls_valid,
        "baseline_checkpoint": baseline_load.get("checkpoint"),
        "candidate_checkpoint": candidate_load.get("checkpoint"),
        "baseline": {
            "run": str(baseline.resolve()),
            "macro_success": baseline_macro,
            "overall_success": baseline_overall,
            "success_count": round(baseline_overall * expected_task_count),
            "invalid_action_rate": baseline_summary.get("invalid_action_rate"),
            "zero_success_task_types": sum(
                value == 0.0 for value in baseline_per_type.values()
            ),
        },
        "candidate": {
            "run": str(candidate.resolve()),
            "macro_success": candidate_macro,
            "overall_success": candidate_overall,
            "success_count": round(candidate_overall * expected_task_count),
            "invalid_action_rate": candidate_summary.get("invalid_action_rate"),
            "zero_success_task_types": sum(
                value == 0.0 for value in candidate_per_type.values()
            ),
        },
        "candidate_minus_baseline": {
            "macro_success": macro_delta,
            "overall_success": overall_delta,
        },
        "per_task_type": per_task_type,
        "ordering": ordering,
        "efficacy_noninferior": efficacy_noninferior,
        "passed": controls_valid and efficacy_noninferior,
    }


def _without_checkpoint(
    payload: dict[str, object],
    *,
    allowed_runtime_differences: tuple[str, ...] = (),
    allow_eval_batch_size_difference: bool = False,
) -> dict[str, object]:
    normalized = json.loads(json.dumps(payload))
    if allow_eval_batch_size_difference:
        normalized.pop("eval_batch_size", None)
        app_config = normalized.get("app_config")
        if isinstance(app_config, dict):
            app_config.pop("eval_batch_size", None)
        evaluation_manifest = normalized.get("evaluation_manifest")
        if isinstance(evaluation_manifest, dict):
            evaluation_manifest.pop("eval_batch_size", None)
    runtime = normalized.get("evaluation_runtime")
    if isinstance(runtime, dict):
        runtime.pop("checkpoint_step", None)
        runtime.pop("policy_checkpoint", None)
        if allow_eval_batch_size_difference:
            runtime.pop("eval_batch_size", None)
        for field in allowed_runtime_differences:
            runtime.pop(field, None)
    return normalized


def _checkpoint_loaded(payload: dict[str, object]) -> bool:
    reports = payload.get("worker_reports")
    return (
        payload.get("requested") is True
        and payload.get("loaded") is True
        and payload.get("status") == "loaded"
        and isinstance(payload.get("checkpoint"), str)
        and bool(payload.get("checkpoint"))
        and isinstance(reports, list)
        and bool(reports)
        and all(
            isinstance(report, dict)
            and report.get("lora_state_loaded") is True
            and report.get("infoskill_state_loaded") is True
            for report in reports
        )
    )


def _per_type(payload: dict[str, object]) -> dict[str, float]:
    raw = payload.get("per_task_type_success")
    if not isinstance(raw, dict):
        raise RuntimeError("evaluation summary lacks per-task-type success")
    return {str(key): _finite_float(value) for key, value in raw.items()}


def _finite_float(value: object) -> float:
    import math

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"expected a numeric evaluation metric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"evaluation metric is not finite: {value!r}")
    return result


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--expected-task-count", type=int, default=140)
    parser.add_argument(
        "--allowed-runtime-difference",
        action="append",
        choices=ALLOWED_EXECUTION_RUNTIME_DIFFERENCES,
        default=[],
        help=(
            "evaluation_runtime field that may differ; repeat for each explicitly "
            "controlled execution-only difference"
        ),
    )
    parser.add_argument(
        "--allow-eval-batch-size-difference",
        action="store_true",
        help=(
            "allow only the recorded evaluation batch size to differ while "
            "keeping every other evaluation protocol field identical"
        ),
    )
    parser.add_argument(
        "--require-same-checkpoint",
        action="store_true",
        help="require both evaluations to load the exact same checkpoint path",
    )
    args = parser.parse_args()
    report = compare_evaluations(
        args.baseline,
        args.candidate,
        expected_task_count=args.expected_task_count,
        allowed_runtime_differences=tuple(args.allowed_runtime_difference),
        allow_eval_batch_size_difference=(
            args.allow_eval_batch_size_difference
        ),
        require_same_checkpoint=args.require_same_checkpoint,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
