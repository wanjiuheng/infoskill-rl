#!/usr/bin/env python3
"""Compare legacy and persistent-session rollout traces for semantic parity."""

from __future__ import annotations

import argparse
import io
import json
from copy import deepcopy
from pathlib import Path
from typing import Sequence


def compare_records(
    baseline: Sequence[dict],
    optimized: Sequence[dict],
    *,
    logprob_tolerance: float,
) -> dict[str, object]:
    baseline_ordered = sorted(baseline, key=_record_key)
    optimized_ordered = sorted(optimized, key=_record_key)
    semantic_mismatches: list[str] = []
    logprob_count = 0
    max_logprob_error = 0.0

    if len(baseline_ordered) != len(optimized_ordered):
        semantic_mismatches.append(
            f"trajectory_count: {len(baseline_ordered)} != {len(optimized_ordered)}"
        )
    for index, (left, right) in enumerate(zip(baseline_ordered, optimized_ordered)):
        key = _record_key(left)
        if key != _record_key(right):
            semantic_mismatches.append(
                f"trajectory[{index}] key: {key!r} != {_record_key(right)!r}"
            )
            continue
        left_semantic = _without_logprobs(left)
        right_semantic = _without_logprobs(right)
        if left_semantic != right_semantic:
            semantic_mismatches.append(f"trajectory {key}: semantic trace differs")
        left_logprobs = _logprobs(left)
        right_logprobs = _logprobs(right)
        if len(left_logprobs) != len(right_logprobs):
            semantic_mismatches.append(
                f"trajectory {key}: logprob_count {len(left_logprobs)} != "
                f"{len(right_logprobs)}"
            )
            continue
        logprob_count += len(left_logprobs)
        if left_logprobs:
            max_logprob_error = max(
                max_logprob_error,
                max(abs(a - b) for a, b in zip(left_logprobs, right_logprobs)),
            )

    semantic_exact = not semantic_mismatches
    logprobs_close = max_logprob_error <= logprob_tolerance
    return {
        "passed": semantic_exact and logprobs_close,
        "semantic_exact": semantic_exact,
        "logprobs_close": logprobs_close,
        "trajectory_count": len(baseline_ordered),
        "logprob_count": logprob_count,
        "max_logprob_abs_error": max_logprob_error,
        "logprob_tolerance": logprob_tolerance,
        "semantic_mismatches": semantic_mismatches[:20],
    }


def _record_key(record: dict) -> tuple[str, int]:
    return str(record.get("task_id")), int(record.get("rollout_id", -1))


def _without_logprobs(record: dict) -> dict:
    copied = deepcopy(record)
    for step in copied.get("steps", []):
        step.pop("old_token_logprobs", None)
        environment = step.get("environment_raw_output", {})
        info = environment.get("info", {})
        facts = info.get("facts")
        if isinstance(facts, list):
            # TextWorld exposes facts as a set-like collection whose JSON list
            # ordering can differ between otherwise identical processes.
            info["facts"] = sorted(facts)
    return copied


def _logprobs(record: dict) -> list[float]:
    return [
        float(value)
        for step in record.get("steps", [])
        for value in step.get("old_token_logprobs", [])
    ]


def _read_traces(run_directory: Path) -> list[dict]:
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("trace comparison requires zstandard") from error
    paths = sorted((run_directory / "traces").glob("train-update-*.jsonl.zst"))
    if not paths:
        raise RuntimeError(f"no training traces found under {run_directory}")
    records: list[dict] = []
    for path in paths:
        with path.open("rb") as source:
            with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                for line in io.TextIOWrapper(reader, encoding="utf-8"):
                    records.append(json.loads(line))
    return records


def _session_setting(run_directory: Path) -> bool | None:
    payload = json.loads((run_directory / "resolved_config.json").read_text(encoding="utf-8"))
    value = payload.get("runtime_options", {}).get("persistent_rollout_session")
    return value if isinstance(value, bool) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--logprob-tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    baseline_setting = _session_setting(args.baseline)
    optimized_setting = _session_setting(args.optimized)
    report = compare_records(
        _read_traces(args.baseline),
        _read_traces(args.optimized),
        logprob_tolerance=args.logprob_tolerance,
    )
    settings_valid = baseline_setting is False and optimized_setting is True
    report.update(
        {
            "baseline": str(args.baseline.resolve()),
            "optimized": str(args.optimized.resolve()),
            "baseline_persistent_rollout_session": baseline_setting,
            "optimized_persistent_rollout_session": optimized_setting,
            "settings_valid": settings_valid,
        }
    )
    report["passed"] = bool(report["passed"] and settings_valid)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
