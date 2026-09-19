#!/usr/bin/env python3
"""Compare three paired M0 LoRA learning-rate forks and fixed evaluations."""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from statistics import mean

try:
    from scripts.compare_infoskill_training_optimization_runs import (
        _compare_training_trace_workload,
        _read_training_trace,
    )
except ModuleNotFoundError:
    from compare_infoskill_training_optimization_runs import (
        _compare_training_trace_workload,
        _read_training_trace,
    )


EXPECTED_RATES = (1e-6, 3e-6, 1e-5)


def _read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def _train_rows(run: Path, start: int, end: int) -> dict[int, dict]:
    rows = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {
        int(row["step"]): row
        for row in rows
        if row.get("phase") == "train" and start < int(row["step"]) <= end
    }
    if set(result) != set(range(start + 1, end + 1)):
        raise ValueError(f"training updates are incomplete in {run}: {sorted(result)}")
    return result


def _evaluation_tasks(run: Path, step: int) -> dict[str, bool]:
    import zstandard

    path = run / "traces" / f"valid-seen-{step:06d}-rank-000.jsonl.zst"
    with path.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            rows = [
                json.loads(line)
                for line in io.TextIOWrapper(reader, encoding="utf-8")
                if line.strip()
            ]
    result = {str(row["task_id"]): bool(row["won"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate evaluation task in {path}")
    return result


def _normalized_config(config: dict) -> dict:
    result = json.loads(json.dumps(config))
    result["runtime_options"].pop("actor_learning_rate", None)
    return result


def _mean(rows: dict[int, dict], key: str) -> float:
    values = [float(row[key]) for row in rows.values()]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"missing or non-finite metric: {key}")
    return mean(values)


def compare(
    source: Path,
    train_runs: list[Path],
    eval_runs: list[Path],
    *,
    expected_task_count: int,
) -> dict:
    source_payload = _read_json(source / "checkpoint.complete.json")
    start = int(source_payload["global_update"])
    configs = [_read_json(run / "resolved_config.json") for run in train_runs]
    provenances = [_read_json(run / "provenance.json") for run in train_runs]
    summaries = [_read_json(run / "training_summary.json") for run in train_runs]
    end = int(summaries[0]["global_update"])
    rows = [_train_rows(run, start, end) for run in train_runs]
    eval_summaries = [_read_json(run / "valid_seen_summary.json") for run in eval_runs]
    eval_loads = [_read_json(run / "checkpoint-load.json") for run in eval_runs]
    eval_provenances = [_read_json(run / "provenance.json") for run in eval_runs]
    checkpoint_markers = [
        _read_json(run / "checkpoints" / f"step-{end:06d}" / "checkpoint.complete.json")
        for run in train_runs
    ]

    rates = [float(config["runtime_options"]["actor_learning_rate"]) for config in configs]
    controls = {
        "expected_learning_rates": rates == list(EXPECTED_RATES),
        "all_m0_named_forks_from_same_checkpoint": all(
            config.get("mode") == "no_skill"
            and provenance.get("resume_forked") is True
            and Path(provenance.get("resume_source_checkpoint", "")).resolve()
            == source.resolve()
            for config, provenance in zip(configs, provenances, strict=True)
        ),
        "same_training_config_except_learning_rate": all(
            _normalized_config(configs[0]) == _normalized_config(config)
            for config in configs[1:]
        ),
        "same_short_update_window": all(
            summary.get("status") == "paused"
            and int(summary.get("global_update", -1)) == end
            and provenance.get("invocation", {}).get("segment_start_update") == start
            and provenance.get("invocation", {}).get("segment_end_update") == end
            for summary, provenance in zip(summaries, provenances, strict=True)
        ),
        "effective_learning_rate_verified_every_update": all(
            math.isclose(float(row.get("actor/lr", -1)), rate, rel_tol=1e-9)
            for rate, branch in zip(EXPECTED_RATES, rows, strict=True)
            for row in branch.values()
        ),
        "all_updates_finite": all(
            all(
                math.isfinite(float(value))
                for key, value in row.items()
                if isinstance(value, (int, float)) and key != "step"
            )
            for branch in rows for row in branch.values()
        ),
        "all_branch_checkpoints_committed_and_portable": all(
            marker.get("portable") is True
            and marker.get("emergency") is not True
            and int(marker.get("global_update", -1)) == end
            for marker in checkpoint_markers
        ),
        "same_complete_140_task_evaluation": all(
            summary.get("is_complete") is True
            and int(summary.get("evaluated", -1)) == expected_task_count
            and summary.get("task_manifest_sha256")
            == eval_summaries[0].get("task_manifest_sha256")
            for summary in eval_summaries
        ),
        "same_fixed_evaluation_runtime": all(
            provenance.get("evaluation_runtime", {}).get("backend") == "verl"
            and provenance.get("evaluation_runtime", {}).get("num_gpus") == 3
            and provenance.get("evaluation_runtime", {}).get("eval_batch_size") == 64
            and provenance.get("evaluation_runtime", {}).get("environment_backend")
            == "native_batch"
            and provenance.get("evaluation_runtime", {}).get(
                "persistent_rollout_session"
            )
            is True
            and provenance.get("evaluation_runtime", {}).get(
                "hybrid_prefix_cuda_graph"
            )
            is False
            for provenance in eval_provenances
        ),
        "each_evaluation_loaded_own_checkpoint": all(
            load.get("status") == "loaded"
            and int(load.get("checkpoint_step", -1)) == end
            and Path(load.get("checkpoint", "")).resolve()
            == (train / "checkpoints" / f"step-{end:06d}").resolve()
            for train, load in zip(train_runs, eval_loads, strict=True)
        ),
    }

    workload: dict[str, dict] = {}
    for step in range(start + 1, end + 1):
        control_trace = _read_training_trace(train_runs[0], step)
        workload[str(step)] = {}
        for rate, run in zip(EXPECTED_RATES[1:], train_runs[1:], strict=True):
            workload[str(step)][f"{rate:.0e}"] = _compare_training_trace_workload(
                control_trace,
                _read_training_trace(run, step),
            )
    controls["same_training_tasks_and_rollout_ids"] = all(
        comparison["passed"] and comparison["baseline_trajectory_count"] == 64
        for step in workload.values() for comparison in step.values()
    )

    task_results = [_evaluation_tasks(run, end) for run in eval_runs]
    controls["same_evaluation_task_ids"] = (
        len(task_results[0]) == expected_task_count
        and all(set(task_results[0]) == set(item) for item in task_results[1:])
    )

    branches: dict[str, dict] = {}
    control_tasks = task_results[0]
    for rate, train, evaluation, training, tasks in zip(
        EXPECTED_RATES,
        train_runs,
        eval_runs,
        rows,
        task_results,
        strict=True,
    ):
        gains = sorted(task for task in tasks if tasks[task] and not control_tasks[task])
        losses = sorted(task for task in tasks if not tasks[task] and control_tasks[task])
        summary = eval_summaries[len(branches)]
        branches[f"{rate:.0e}"] = {
            "learning_rate": rate,
            "train_run": str(train),
            "eval_run": str(evaluation),
            "training": {
                "mean_success": _mean(training, "rollout/success_rate"),
                "mean_reward": _mean(training, "rollout/mean_reward"),
                "mean_invalid_action_rate": _mean(training, "rollout/invalid_action_rate"),
                "mean_ppo_kl": _mean(training, "actor/ppo_kl"),
                "max_ppo_kl": max(float(row["actor/ppo_kl"]) for row in training.values()),
                "mean_clipfrac": _mean(training, "actor/pg_clipfrac"),
                "max_clipfrac": max(float(row["actor/pg_clipfrac"]) for row in training.values()),
                "mean_grad_norm": _mean(training, "actor/grad_norm"),
                "mean_core_update_seconds": _mean(training, "perf/core_update_seconds"),
            },
            "evaluation": {
                "success_count": int(
                    summary.get(
                        "success_count",
                        round(float(summary["overall_success"]) * expected_task_count),
                    )
                ),
                "macro_success": float(summary["macro_success"]),
                "overall_success": float(summary["overall_success"]),
                "invalid_action_rate": float(summary["invalid_action_rate"]),
                "mean_steps": float(summary["mean_steps"]),
                "per_task_type": summary["per_task_type_success"],
            },
            "paired_vs_1e-6": {
                "gain_count": len(gains),
                "loss_count": len(losses),
                "net_gain": len(gains) - len(losses),
                "gain_task_ids": gains,
                "loss_task_ids": losses,
            },
        }

    controls_valid = all(controls.values())
    ranking = sorted(
        branches,
        key=lambda name: (
            branches[name]["evaluation"]["macro_success"],
            branches[name]["evaluation"]["overall_success"],
            -branches[name]["evaluation"]["invalid_action_rate"],
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "decision_scope": "five_update_paired_screen_not_final_hyperparameter_proof",
        "source_checkpoint": str(source),
        "update_window": {"start_exclusive": start, "end_inclusive": end},
        "controls": controls,
        "controls_valid": controls_valid,
        "training_workload": workload,
        "branches": branches,
        "ranking_macro_then_overall": ranking,
        "classification": (
            "paired_lr_screen_complete" if controls_valid else "invalid_controls"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_checkpoint", type=Path)
    parser.add_argument("--train-runs", nargs=3, required=True, type=Path)
    parser.add_argument("--eval-runs", nargs=3, required=True, type=Path)
    parser.add_argument("--expected-task-count", type=int, default=140)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare(
        args.source_checkpoint,
        args.train_runs,
        args.eval_runs,
        expected_task_count=args.expected_task_count,
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "classification": report["classification"],
        "controls_valid": report["controls_valid"],
        "ranking": report["ranking_macro_then_overall"],
        "branches": {
            name: {
                "evaluation": value["evaluation"],
                "paired_vs_1e-6": {
                    key: metric for key, metric in value["paired_vs_1e-6"].items()
                    if key in {"gain_count", "loss_count", "net_gain"}
                },
            }
            for name, value in report["branches"].items()
        },
        "report": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0 if report["controls_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
