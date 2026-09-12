#!/usr/bin/env python3
"""Gate grouped INFO-SKILL evaluation conditioning against the legacy path."""

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
    optimized: Path,
    *,
    logprob_tolerance: float = 1e-3,
    minimum_rollout_speedup: float = 1.05,
) -> dict[str, object]:
    baseline_provenance = _read_json(baseline / "provenance.json")
    optimized_provenance = _read_json(optimized / "provenance.json")
    baseline_summary = _read_json(baseline / "valid_seen_summary.json")
    optimized_summary = _read_json(optimized / "valid_seen_summary.json")
    baseline_runtime = dict(baseline_provenance.get("evaluation_runtime", {}))
    optimized_runtime = dict(optimized_provenance.get("evaluation_runtime", {}))
    baseline_conditioning = dict(
        baseline_provenance.get("skill_conditioning", {})
    )
    optimized_conditioning = dict(
        optimized_provenance.get("skill_conditioning", {})
    )

    control_checks = {
        "both_infoskill": (
            baseline_provenance.get("mode") == "infoskill"
            and optimized_provenance.get("mode") == "infoskill"
        ),
        "same_task_manifest": (
            baseline_provenance.get("evaluation_manifest")
            == optimized_provenance.get("evaluation_manifest")
        ),
        "same_policy_model": (
            _policy_sha256(baseline_provenance)
            == _policy_sha256(optimized_provenance)
        ),
        "same_checkpoint_step": (
            baseline_runtime.get("checkpoint_step")
            == optimized_runtime.get("checkpoint_step")
        ),
        "same_checkpoint": (
            baseline_runtime.get("policy_checkpoint")
            == optimized_runtime.get("policy_checkpoint")
        ),
        "same_gpu_count": (
            baseline_runtime.get("num_gpus")
            == optimized_runtime.get("num_gpus")
        ),
        "same_environment_backend": (
            baseline_runtime.get("environment_backend")
            == optimized_runtime.get("environment_backend")
        ),
        "same_persistent_session": (
            baseline_runtime.get("persistent_rollout_session")
            == optimized_runtime.get("persistent_rollout_session")
        ),
        "same_retrieval_plan": (
            baseline_conditioning.get("retrieval_plan_sha256")
            == optimized_conditioning.get("retrieval_plan_sha256")
        ),
        "baseline_grouping_disabled": not bool(
            baseline_runtime.get("grouped_infoskill_conditioning", False)
        ),
        "optimized_grouping_enabled": bool(
            optimized_runtime.get("grouped_infoskill_conditioning", False)
        ),
    }
    parity = compare_records(
        _read_evaluation_traces(baseline),
        _read_evaluation_traces(optimized),
        logprob_tolerance=logprob_tolerance,
    )
    baseline_rollout = _rollout_seconds(baseline_summary)
    optimized_rollout = _rollout_seconds(optimized_summary)
    rollout_speedup = (
        baseline_rollout / optimized_rollout
        if optimized_rollout > 0.0
        else 0.0
    )
    performance_valid = rollout_speedup >= minimum_rollout_speedup
    settings_valid = all(control_checks.values())
    return {
        "baseline": str(baseline),
        "optimized": str(optimized),
        "control_checks": control_checks,
        "settings_valid": settings_valid,
        **parity,
        "baseline_performance": _performance_summary(baseline_summary),
        "optimized_performance": _performance_summary(optimized_summary),
        "rollout_speedup": rollout_speedup,
        "minimum_rollout_speedup": minimum_rollout_speedup,
        "performance_valid": performance_valid,
        "passed": settings_valid and bool(parity["passed"]) and performance_valid,
    }


def _policy_sha256(provenance: dict[str, object]) -> object:
    policy = provenance.get("policy_model", {})
    return policy.get("sha256") if isinstance(policy, dict) else None


def _rollout_seconds(summary: dict[str, object]) -> float:
    timing = summary.get("timing_seconds", {})
    if not isinstance(timing, dict):
        return 0.0
    return float(timing.get("rollout_seconds", 0.0))


def _performance_summary(summary: dict[str, object]) -> dict[str, object]:
    timing = summary.get("timing_seconds", {})
    performance = summary.get("rollout_performance", {})
    return {
        "rollout_seconds": (
            timing.get("rollout_seconds") if isinstance(timing, dict) else None
        ),
        "conditioning_seconds": (
            performance.get("perf/rollout_conditioning_seconds")
            if isinstance(performance, dict)
            else None
        ),
        "generation_seconds": (
            performance.get("perf/rollout_backend_generate_seconds")
            if isinstance(performance, dict)
            else None
        ),
        "conditioning_batch_calls": (
            performance.get("perf/rollout_conditioning_batch_calls")
            if isinstance(performance, dict)
            else None
        ),
    }


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _read_evaluation_traces(run_directory: Path) -> list[dict]:
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("trace comparison requires zstandard") from error
    paths = sorted((run_directory / "traces").glob("valid-seen-*.jsonl.zst"))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one valid_seen trace under {run_directory}, found {len(paths)}"
        )
    records: list[dict] = []
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
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--logprob-tolerance", type=float, default=1e-3)
    parser.add_argument("--minimum-rollout-speedup", type=float, default=1.05)
    args = parser.parse_args()
    report = compare_runs(
        args.baseline,
        args.optimized,
        logprob_tolerance=args.logprob_tolerance,
        minimum_rollout_speedup=args.minimum_rollout_speedup,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
