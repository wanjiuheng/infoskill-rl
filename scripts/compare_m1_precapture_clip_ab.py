#!/usr/bin/env python3
"""Compare paired M1 joint/separate clipping forks after ten updates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.compare_infoskill_optimization_efficacy import compare_evaluations
    from scripts.compare_infoskill_training_optimization_runs import (
        _compare_training_trace_workload,
        _read_training_trace,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from compare_infoskill_optimization_efficacy import compare_evaluations
    from compare_infoskill_training_optimization_runs import (
        _compare_training_trace_workload,
        _read_training_trace,
    )


def _read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def _train_metrics(run: Path) -> dict[int, dict]:
    rows = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train = [row for row in rows if row.get("phase") == "train"]
    by_step = {int(row["step"]): row for row in train}
    if len(by_step) != len(train):
        raise ValueError(f"duplicate training update in {run}")
    return by_step


def _normalized_training_config(config: dict) -> dict:
    normalized = json.loads(json.dumps(config))
    normalized["runtime_options"].pop("policy_gradient_clip_mode", None)
    return normalized


def _checkpoint_committed(run: Path, step: int) -> bool:
    marker = run / "checkpoints" / f"step-{step:06d}" / "checkpoint.complete.json"
    if not marker.is_file():
        return False
    payload = _read_json(marker)
    runtime = payload.get("runtime_manifest")
    return (
        payload.get("global_update") == step
        and payload.get("portable") is True
        and payload.get("emergency") is False
        and isinstance(runtime, dict)
        and runtime.get("infoskill_modules_included") is True
    )


def _evaluation_trace(run: Path, step: int) -> dict[str, dict]:
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("trace comparison requires zstandard") from error

    path = run / "traces" / f"valid-seen-{step:06d}-rank-000.jsonl.zst"
    with path.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            import io

            rows = [
                json.loads(line)
                for line in io.TextIOWrapper(reader, encoding="utf-8")
                if line.strip()
            ]
    tasks = {str(row["task_id"]): row for row in rows}
    if len(tasks) != len(rows):
        raise ValueError(f"duplicate task ID in {path}")
    return tasks


def _exact_internal_external_trace(train: Path, evaluation: Path) -> bool:
    name = "valid-seen-000225-rank-000.jsonl.zst"
    internal = train / "traces" / name
    external = evaluation / "traces" / name
    if not internal.is_file() or not external.is_file():
        return False
    return hashlib.sha256(internal.read_bytes()).digest() == hashlib.sha256(
        external.read_bytes()
    ).digest()


def compare_clip_ab(
    source: Path,
    control_train: Path,
    control_eval: Path,
    candidate_train: Path,
    candidate_eval: Path,
) -> dict:
    expected_steps = set(range(216, 226))
    configs = [_read_json(run / "resolved_config.json") for run in (control_train, candidate_train)]
    provenance = [_read_json(run / "provenance.json") for run in (control_train, candidate_train)]
    summaries = [_read_json(run / "training_summary.json") for run in (control_train, candidate_train)]
    metrics = [_train_metrics(run) for run in (control_train, candidate_train)]
    evaluation = compare_evaluations(control_eval, candidate_eval, expected_task_count=140)
    loads = [_read_json(run / "checkpoint-load.json") for run in (control_eval, candidate_eval)]
    eval_runtime = [
        _read_json(run / "provenance.json")["evaluation_runtime"]
        for run in (control_eval, candidate_eval)
    ]

    expected_invocation = {"segment_start_update": 215, "segment_end_update": 225}
    controls = {
        "same_source_checkpoint": all(
            item.get("resume_source_checkpoint") == str(source.resolve())
            and item.get("resume_forked") is True
            and item.get("invocation") == expected_invocation
            for item in provenance
        ),
        "same_training_config_except_clip_mode": (
            _normalized_training_config(configs[0])
            == _normalized_training_config(configs[1])
            and configs[0]["runtime_options"]["policy_gradient_clip_mode"] == "joint"
            and configs[1]["runtime_options"]["policy_gradient_clip_mode"] == "separate"
        ),
        "both_paused_at_225": all(
            item.get("status") == "paused"
            and item.get("global_update") == 225
            and item.get("max_updates") == 445
            for item in summaries
        ),
        "both_committed_portable_225": all(
            _checkpoint_committed(run, 225)
            for run in (control_train, candidate_train)
        ),
        "both_have_exactly_ten_training_updates": all(
            set(item) == expected_steps for item in metrics
        ),
        "clip_modes_executed_on_every_update": all(
            float(metrics[0][step].get("policy/separate_gradient_clipping", -1)) == 0.0
            and float(metrics[1][step].get("policy/separate_gradient_clipping", -1)) == 1.0
            for step in expected_steps
            if step in metrics[0] and step in metrics[1]
        ),
        "all_policy_and_aux_optimizer_steps_applied": all(
            row.get("policy/optimizer_step_applied") == 1.0
            and row.get("aux/optimizer_step_applied") == 1.0
            and row.get("policy/optimizer_skip_nonfinite", 0.0) == 0.0
            for item in metrics for row in item.values()
        ),
        "both_graph_split_k_one_on_every_update": all(
            row.get("perf/hybrid_prefix_cuda_graph") == 1.0
            and row.get("perf/lora_shrink_split_k_one_verified") == 1.0
            for item in metrics for row in item.values()
        ),
        "both_evaluations_same_protocol": evaluation["controls_valid"],
        "both_evaluations_load_own_225_checkpoint": all(
            Path(load.get("checkpoint", "")).resolve()
            == (run / "checkpoints" / "step-000225").resolve()
            and load.get("checkpoint_step") == 225
            for load, run in zip(loads, (control_train, candidate_train), strict=True)
        ),
        "both_evaluations_graph_split_k_one_batch_64": all(
            runtime.get("hybrid_prefix_cuda_graph") is True
            and runtime.get("lora_shrink_split_k_one") is True
            and runtime.get("eval_batch_size") == 64
            and runtime.get("num_gpus") == 3
            for runtime in eval_runtime
        ),
        "control_internal_external_trace_exact": _exact_internal_external_trace(control_train, control_eval),
        "candidate_internal_external_trace_exact": _exact_internal_external_trace(candidate_train, candidate_eval),
    }

    workload = {}
    for step in sorted(expected_steps):
        if step not in metrics[0] or step not in metrics[1]:
            workload[str(step)] = {"passed": False, "reason": "missing training metric"}
            continue
        workload[str(step)] = _compare_training_trace_workload(
            _read_training_trace(control_train, step),
            _read_training_trace(candidate_train, step),
        )
    controls["same_task_and_rollout_identity_every_update"] = all(
        item["passed"] and item["baseline_trajectory_count"] == 64
        for item in workload.values()
    )

    control_tasks = _evaluation_trace(control_eval, 225)
    candidate_tasks = _evaluation_trace(candidate_eval, 225)
    controls["same_140_evaluation_task_ids"] = (
        len(control_tasks) == len(candidate_tasks) == 140
        and set(control_tasks) == set(candidate_tasks)
    )
    gains = [
        task_id for task_id in control_tasks.keys() & candidate_tasks.keys()
        if not control_tasks[task_id]["won"] and candidate_tasks[task_id]["won"]
    ]
    losses = [
        task_id for task_id in control_tasks.keys() & candidate_tasks.keys()
        if control_tasks[task_id]["won"] and not candidate_tasks[task_id]["won"]
    ]

    macro_delta = evaluation["candidate_minus_baseline"]["macro_success"]
    overall_delta = evaluation["candidate_minus_baseline"]["overall_success"]
    controls_valid = all(controls.values())
    if not controls_valid:
        classification = "invalid_controls"
    elif macro_delta >= 0.03 and overall_delta >= 0.0:
        classification = "candidate_preliminary_gain_needs_confirmation"
    elif macro_delta <= -0.03 and overall_delta <= 0.0:
        classification = "candidate_preliminary_loss"
    else:
        classification = "inconclusive_short_window"

    def gradient_summary(item: dict[int, dict]) -> dict[str, float]:
        rows = list(item.values())
        return {
            "mean_actor_clip_coefficient": sum(float(row["policy/actor_clip_coefficient"]) for row in rows) / len(rows),
            "mean_projector_clip_coefficient": sum(float(row["policy/projector_clip_coefficient"]) for row in rows) / len(rows),
            "mean_projector_to_actor_grad_norm_ratio": sum(float(row["policy/projector_to_actor_grad_norm_ratio"]) for row in rows) / len(rows),
            "mean_mixed_outcome_groups": sum(float(row["grpo_signal/mixed_outcome_group_count"]) for row in rows) / len(rows),
            "mean_training_success": sum(float(row["rollout/success_rate"]) for row in rows) / len(rows),
        }

    return {
        "schema_version": 1,
        "decision_scope": "paired_short_window_not_formal_improvement_proof",
        "source_checkpoint": str(source),
        "control_train": str(control_train),
        "candidate_train": str(candidate_train),
        "control_eval": str(control_eval),
        "candidate_eval": str(candidate_eval),
        "controls": controls,
        "controls_valid": controls_valid,
        "training_workload_by_update": workload,
        "evaluation": evaluation,
        "paired_task_outcomes": {
            "gain_count": len(gains),
            "loss_count": len(losses),
            "unchanged_count": len(control_tasks.keys() & candidate_tasks.keys()) - len(gains) - len(losses),
            "gain_task_ids": sorted(gains),
            "loss_task_ids": sorted(losses),
        },
        "gradient_summary": {
            "joint": gradient_summary(metrics[0]) if metrics[0] else None,
            "separate": gradient_summary(metrics[1]) if metrics[1] else None,
        },
        "classification": classification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_checkpoint", type=Path)
    parser.add_argument("control_train", type=Path)
    parser.add_argument("control_eval", type=Path)
    parser.add_argument("candidate_train", type=Path)
    parser.add_argument("candidate_eval", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare_clip_ab(
        args.source_checkpoint,
        args.control_train,
        args.control_eval,
        args.candidate_train,
        args.candidate_eval,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": report["classification"],
        "controls_valid": report["controls_valid"],
        "success_delta": report["evaluation"]["candidate"]["success_count"] - report["evaluation"]["baseline"]["success_count"],
        "macro_delta": report["evaluation"]["candidate_minus_baseline"]["macro_success"],
        "overall_delta": report["evaluation"]["candidate_minus_baseline"]["overall_success"],
        "paired_task_outcomes": {
            key: value for key, value in report["paired_task_outcomes"].items()
            if key in {"gain_count", "loss_count", "unchanged_count"}
        },
        "report": str(args.output),
    }, indent=2))
    return 0 if report["controls_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
