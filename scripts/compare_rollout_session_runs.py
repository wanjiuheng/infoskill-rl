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
            differences = _first_differences(left_semantic, right_semantic, limit=3)
            if differences:
                semantic_mismatches.extend(
                    f"trajectory {key}: {difference}" for difference in differences
                )
            else:
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


def _first_differences(
    left: object,
    right: object,
    *,
    path: str = "$",
    limit: int = 3,
) -> list[str]:
    """Return compact, deterministic paths for the first JSON differences."""
    differences: list[str] = []

    def visit(left_value: object, right_value: object, current_path: str) -> None:
        if len(differences) >= limit:
            return
        if type(left_value) is not type(right_value):
            differences.append(
                f"{current_path}: type {type(left_value).__name__} != "
                f"{type(right_value).__name__} "
                f"({_abbreviate(left_value)} != {_abbreviate(right_value)})"
            )
            return
        if isinstance(left_value, dict):
            left_keys = set(left_value)
            right_keys = set(right_value)
            for key in sorted(left_keys | right_keys, key=str):
                child_path = _json_path(current_path, key)
                if key not in left_value:
                    differences.append(
                        f"{child_path}: <missing> != {_abbreviate(right_value[key])}"
                    )
                elif key not in right_value:
                    differences.append(
                        f"{child_path}: {_abbreviate(left_value[key])} != <missing>"
                    )
                else:
                    visit(left_value[key], right_value[key], child_path)
                if len(differences) >= limit:
                    return
            return
        if isinstance(left_value, list):
            if len(left_value) != len(right_value):
                differences.append(
                    f"{current_path}.length: {len(left_value)} != {len(right_value)}"
                )
                if len(differences) >= limit:
                    return
            for index, (left_item, right_item) in enumerate(
                zip(left_value, right_value)
            ):
                visit(left_item, right_item, f"{current_path}[{index}]")
                if len(differences) >= limit:
                    return
            return
        if left_value != right_value:
            differences.append(
                f"{current_path}: {_abbreviate(left_value)} != "
                f"{_abbreviate(right_value)}"
            )

    visit(left, right, path)
    return differences


def _json_path(parent: str, key: object) -> str:
    text = str(key)
    if text.isidentifier():
        return f"{parent}.{text}"
    return f"{parent}[{text!r}]"


def _abbreviate(value: object, *, maximum: int = 160) -> str:
    rendered = repr(value)
    if len(rendered) <= maximum:
        return rendered
    return f"{rendered[: maximum - 3]}..."


def _without_logprobs(record: dict) -> dict:
    copied = deepcopy(record)
    for step in copied.get("steps", []):
        step.pop("old_token_logprobs", None)
        environment = step.get("environment_raw_output", {})
        info = environment.get("info", {})
        # ALFWorld computes this hand-coded expert suggestion after each step,
        # but INFO-SKILL does not feed it to the policy, action resolver,
        # reward, state checksum, or policy update.  Its choice among equivalent
        # object instances is therefore diagnostic metadata, not rollout
        # semantics for this parity gate.
        info.pop("extra.expert_plan", None)
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


def _runtime_options(run_directory: Path) -> dict[str, object]:
    payload = json.loads(
        (run_directory / "resolved_config.json").read_text(encoding="utf-8")
    )
    options = payload.get("runtime_options", {})
    return dict(options) if isinstance(options, dict) else {}


def _session_setting(options: dict[str, object]) -> bool | None:
    value = options.get("persistent_rollout_session")
    return value if isinstance(value, bool) else None


def _environment_worker_setting(options: dict[str, object]) -> int | None:
    value = options.get("environment_workers", 1)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _environment_backend_setting(options: dict[str, object]) -> str | None:
    value = options.get("environment_backend", "individual")
    return value if value in {"individual", "native_batch"} else None


def _settings_are_valid(
    comparison_mode: str,
    *,
    baseline_session: bool | None,
    optimized_session: bool | None,
    baseline_environment_workers: int | None,
    optimized_environment_workers: int | None,
    baseline_environment_backend: str | None = "individual",
    optimized_environment_backend: str | None = "individual",
) -> bool:
    if comparison_mode == "persistent-session":
        return baseline_session is False and optimized_session is True
    if comparison_mode == "environment-workers":
        return (
            baseline_session is True
            and optimized_session is True
            and baseline_environment_workers == 1
            and optimized_environment_workers is not None
            and optimized_environment_workers > 1
            and baseline_environment_backend == "individual"
            and optimized_environment_backend == "individual"
        )
    if comparison_mode == "native-batch":
        return (
            baseline_session is True
            and optimized_session is True
            and baseline_environment_workers == 1
            and optimized_environment_workers == 1
            and baseline_environment_backend == "individual"
            and optimized_environment_backend == "native_batch"
        )
    raise ValueError(f"unsupported comparison mode: {comparison_mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--logprob-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--comparison-mode",
        choices=("persistent-session", "environment-workers", "native-batch"),
        default="persistent-session",
    )
    args = parser.parse_args()
    baseline_options = _runtime_options(args.baseline)
    optimized_options = _runtime_options(args.optimized)
    baseline_setting = _session_setting(baseline_options)
    optimized_setting = _session_setting(optimized_options)
    baseline_environment_workers = _environment_worker_setting(baseline_options)
    optimized_environment_workers = _environment_worker_setting(optimized_options)
    baseline_environment_backend = _environment_backend_setting(baseline_options)
    optimized_environment_backend = _environment_backend_setting(optimized_options)
    report = compare_records(
        _read_traces(args.baseline),
        _read_traces(args.optimized),
        logprob_tolerance=args.logprob_tolerance,
    )
    settings_valid = _settings_are_valid(
        args.comparison_mode,
        baseline_session=baseline_setting,
        optimized_session=optimized_setting,
        baseline_environment_workers=baseline_environment_workers,
        optimized_environment_workers=optimized_environment_workers,
        baseline_environment_backend=baseline_environment_backend,
        optimized_environment_backend=optimized_environment_backend,
    )
    report.update(
        {
            "baseline": str(args.baseline.resolve()),
            "optimized": str(args.optimized.resolve()),
            "comparison_mode": args.comparison_mode,
            "baseline_persistent_rollout_session": baseline_setting,
            "optimized_persistent_rollout_session": optimized_setting,
            "baseline_environment_workers": baseline_environment_workers,
            "optimized_environment_workers": optimized_environment_workers,
            "baseline_environment_backend": baseline_environment_backend,
            "optimized_environment_backend": optimized_environment_backend,
            "settings_valid": settings_valid,
        }
    )
    report["passed"] = bool(report["passed"] and settings_valid)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
