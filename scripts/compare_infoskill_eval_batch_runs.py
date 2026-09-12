#!/usr/bin/env python3
"""Gate an INFO-SKILL evaluation batch-size candidate on one fixed subset."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

try:
    from scripts.compare_rollout_session_runs import compare_records
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from compare_rollout_session_runs import compare_records


def compare_runs(
    baseline: Path,
    candidate: Path,
    *,
    baseline_batch_size: int = 8,
    candidate_batch_size: int = 12,
    logprob_tolerance: float = 1e-3,
    minimum_rollout_speedup: float = 1.10,
    minimum_physical_free_gb: float = 8.0,
) -> dict[str, object]:
    baseline_provenance = _read_json(baseline / "provenance.json")
    candidate_provenance = _read_json(candidate / "provenance.json")
    baseline_summary = _read_json(baseline / "diagnostic_summary.json")
    candidate_summary = _read_json(candidate / "diagnostic_summary.json")
    baseline_load = _read_json(baseline / "checkpoint-load.json")
    candidate_load = _read_json(candidate / "checkpoint-load.json")
    baseline_runtime = dict(baseline_provenance.get("evaluation_runtime", {}))
    candidate_runtime = dict(candidate_provenance.get("evaluation_runtime", {}))

    controlled_runtime_fields = (
        "num_gpus",
        "environment_backend",
        "persistent_rollout_session",
        "grouped_infoskill_conditioning",
        "policy_checkpoint",
        "checkpoint_step",
        "cuda_memory_poll_interval_ms",
    )
    control_checks = {
        "diagnostic_artifacts": (
            baseline_provenance.get("artifact_kind")
            == "infoskill_eval_batch_pressure_diagnostic"
            and candidate_provenance.get("artifact_kind")
            == "infoskill_eval_batch_pressure_diagnostic"
        ),
        "both_infoskill": (
            baseline_provenance.get("mode") == "infoskill"
            and candidate_provenance.get("mode") == "infoskill"
        ),
        "same_pressure_manifest": (
            baseline_provenance.get("evaluation_manifest")
            == candidate_provenance.get("evaluation_manifest")
        ),
        "same_policy_model": (
            _policy_sha256(baseline_provenance)
            == _policy_sha256(candidate_provenance)
        ),
        "same_skill_conditioning": (
            baseline_provenance.get("skill_conditioning")
            == candidate_provenance.get("skill_conditioning")
        ),
        "same_runtime_controls": all(
            baseline_runtime.get(field) == candidate_runtime.get(field)
            for field in controlled_runtime_fields
        ),
        "baseline_batch_size": (
            baseline_runtime.get("eval_batch_size") == baseline_batch_size
        ),
        "candidate_batch_size": (
            candidate_runtime.get("eval_batch_size") == candidate_batch_size
        ),
        "both_complete": (
            baseline_summary.get("is_complete") is True
            and candidate_summary.get("is_complete") is True
            and baseline_summary.get("evaluated") == 12
            and candidate_summary.get("evaluated") == 12
        ),
        "both_non_reportable": (
            baseline_summary.get("reportable_as_valid_seen") is False
            and candidate_summary.get("reportable_as_valid_seen") is False
        ),
    }
    baseline_records = _read_diagnostic_trace(baseline)
    candidate_records = _read_diagnostic_trace(candidate)
    parity = compare_records(
        baseline_records,
        candidate_records,
        logprob_tolerance=logprob_tolerance,
    )
    tokens_exact = _token_sequences_exact(baseline_records, candidate_records)
    logprob_comparison_valid = (
        tokens_exact
        and _logprob_shapes_match(baseline_records, candidate_records)
        and int(parity["logprob_count"]) > 0
    )
    logprobs_close = (
        logprob_comparison_valid and bool(parity["logprobs_close"])
    )
    parity["logprobs_close"] = logprobs_close
    parity["passed"] = bool(parity["semantic_exact"]) and logprobs_close
    baseline_rollout = _metric(baseline_summary, "timing_seconds", "rollout_seconds")
    candidate_rollout = _metric(candidate_summary, "timing_seconds", "rollout_seconds")
    baseline_generation = _metric(
        baseline_summary,
        "rollout_performance",
        "perf/rollout_backend_generate_seconds",
    )
    candidate_generation = _metric(
        candidate_summary,
        "rollout_performance",
        "perf/rollout_backend_generate_seconds",
    )
    candidate_free = _metric(
        candidate_summary,
        "rollout_performance",
        "perf/cuda/rollout_physical_min_free_gb_min",
    )
    rollout_speedup = _speedup(baseline_rollout, candidate_rollout)
    generation_speedup = _speedup(baseline_generation, candidate_generation)
    performance_valid = rollout_speedup >= minimum_rollout_speedup
    physical_memory_valid = candidate_free >= minimum_physical_free_gb
    checkpoint_load_checks = {
        "baseline": _checkpoint_loaded_on_every_rank(
            baseline_load,
            expected_ranks=int(baseline_runtime.get("num_gpus", 0)),
        ),
        "candidate": _checkpoint_loaded_on_every_rank(
            candidate_load,
            expected_ranks=int(candidate_runtime.get("num_gpus", 0)),
        ),
    }
    settings_valid = all(control_checks.values())
    checkpoint_load_valid = all(checkpoint_load_checks.values())
    passed = (
        settings_valid
        and checkpoint_load_valid
        and bool(parity["passed"])
        and performance_valid
        and physical_memory_valid
    )
    return {
        "schema_version": 1,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "diagnostic_only": True,
        "reportable_as_valid_seen": False,
        "control_checks": control_checks,
        "settings_valid": settings_valid,
        "checkpoint_load_checks": checkpoint_load_checks,
        "checkpoint_load_valid": checkpoint_load_valid,
        **parity,
        "tokens_exact": tokens_exact,
        "logprob_comparison_valid": logprob_comparison_valid,
        "logprob_comparison_status": (
            "compared"
            if logprob_comparison_valid
            else "not_comparable_due_to_token_or_length_drift"
        ),
        "baseline_performance": {
            "rollout_seconds": baseline_rollout,
            "generation_seconds": baseline_generation,
        },
        "candidate_performance": {
            "rollout_seconds": candidate_rollout,
            "generation_seconds": candidate_generation,
            "physical_min_free_gb": candidate_free,
        },
        "rollout_speedup": rollout_speedup,
        "generation_speedup": generation_speedup,
        "minimum_rollout_speedup": minimum_rollout_speedup,
        "performance_valid": performance_valid,
        "minimum_physical_free_gb": minimum_physical_free_gb,
        "physical_memory_valid": physical_memory_valid,
        "full_140_evaluation_recommended": passed,
        "passed": passed,
    }


def _checkpoint_loaded_on_every_rank(
    payload: dict[str, object],
    *,
    expected_ranks: int,
) -> bool:
    reports = payload.get("worker_reports")
    if (
        payload.get("status") != "loaded"
        or payload.get("loaded") is not True
        or expected_ranks <= 0
        or not isinstance(reports, list)
        or len(reports) != expected_ranks
    ):
        return False
    ranks = set()
    for report in reports:
        if not isinstance(report, dict):
            return False
        if not report.get("lora_state_loaded") or not report.get(
            "infoskill_state_loaded"
        ):
            return False
        rank = report.get("rank")
        if not isinstance(rank, int):
            return False
        ranks.add(rank)
    return ranks == set(range(expected_ranks))


def _policy_sha256(provenance: dict[str, object]) -> object:
    policy = provenance.get("policy_model", {})
    return policy.get("sha256") if isinstance(policy, dict) else None


def _metric(payload: dict[str, object], section: str, key: str) -> float:
    values = payload.get(section)
    if not isinstance(values, dict) or values.get(key) is None:
        return 0.0
    return float(values[key])


def _speedup(baseline: float, candidate: float) -> float:
    return baseline / candidate if baseline > 0.0 and candidate > 0.0 else 0.0


def _token_sequences_exact(baseline: list[dict], candidate: list[dict]) -> bool:
    pairs = _aligned_record_pairs(baseline, candidate)
    if pairs is None:
        return False
    for left, right in pairs:
        left_steps = left.get("steps", [])
        right_steps = right.get("steps", [])
        if len(left_steps) != len(right_steps):
            return False
        for left_step, right_step in zip(left_steps, right_steps, strict=True):
            if left_step.get("response_token_ids", []) != right_step.get(
                "response_token_ids", []
            ):
                return False
    return True


def _logprob_shapes_match(baseline: list[dict], candidate: list[dict]) -> bool:
    pairs = _aligned_record_pairs(baseline, candidate)
    if pairs is None:
        return False
    for left, right in pairs:
        left_steps = left.get("steps", [])
        right_steps = right.get("steps", [])
        if len(left_steps) != len(right_steps):
            return False
        for left_step, right_step in zip(left_steps, right_steps, strict=True):
            if len(left_step.get("old_token_logprobs", [])) != len(
                right_step.get("old_token_logprobs", [])
            ):
                return False
    return True


def _aligned_record_pairs(
    baseline: list[dict], candidate: list[dict]
) -> list[tuple[dict, dict]] | None:
    def identity(record: dict) -> tuple[str, int]:
        return str(record.get("task_id", "")), int(record.get("rollout_id", 0))

    left = sorted(baseline, key=identity)
    right = sorted(candidate, key=identity)
    if len(left) != len(right):
        return None
    if [identity(record) for record in left] != [identity(record) for record in right]:
        return None
    return list(zip(left, right, strict=True))


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _read_diagnostic_trace(run_directory: Path) -> list[dict]:
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("trace comparison requires zstandard") from error
    paths = sorted(
        (run_directory / "traces").glob(
            "diagnostic-valid-seen-*-rank-*.jsonl.zst"
        )
    )
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one diagnostic trace under {run_directory}, found {len(paths)}"
        )
    records = []
    with paths[0].open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                record = json.loads(line)
                if record.get("record_type") != "infrastructure_failure":
                    records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-batch-size", type=int, default=8)
    parser.add_argument("--candidate-batch-size", type=int, default=12)
    parser.add_argument("--logprob-tolerance", type=float, default=1e-3)
    parser.add_argument("--minimum-rollout-speedup", type=float, default=1.10)
    parser.add_argument("--minimum-physical-free-gb", type=float, default=8.0)
    args = parser.parse_args()
    report = compare_runs(
        args.baseline,
        args.candidate,
        baseline_batch_size=args.baseline_batch_size,
        candidate_batch_size=args.candidate_batch_size,
        logprob_tolerance=args.logprob_tolerance,
        minimum_rollout_speedup=args.minimum_rollout_speedup,
        minimum_physical_free_gb=args.minimum_physical_free_gb,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
