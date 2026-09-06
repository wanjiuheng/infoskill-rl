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


def _training_performance(run_directory: Path) -> dict[str, float | None]:
    records = [
        json.loads(line)
        for line in (run_directory / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    training = [record for record in records if record.get("phase") == "train"]
    core = [float(record["perf/core_update_seconds"]) for record in training]
    allocated = [
        float(record["perf/max_memory_allocated_gb"])
        for record in training
        if record.get("perf/max_memory_allocated_gb") is not None
    ]
    reserved = [
        float(record["perf/max_memory_reserved_gb"])
        for record in training
        if record.get("perf/max_memory_reserved_gb") is not None
    ]
    return {
        "core_seconds": sum(core) if core else None,
        "max_memory_allocated_gb": max(allocated) if allocated else None,
        "max_memory_reserved_gb": max(reserved) if reserved else None,
    }


def _session_setting(options: dict[str, object]) -> bool | None:
    value = options.get("persistent_rollout_session")
    return value if isinstance(value, bool) else None


def _environment_worker_setting(options: dict[str, object]) -> int | None:
    value = options.get("environment_workers", 1)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _environment_backend_setting(options: dict[str, object]) -> str | None:
    value = options.get("environment_backend", "individual")
    return value if value in {"individual", "native_batch"} else None


def _gradient_checkpointing_setting(options: dict[str, object]) -> bool | None:
    value = options.get("actor_gradient_checkpointing", True)
    return value if isinstance(value, bool) else None


def _settings_are_valid(
    comparison_mode: str,
    *,
    baseline_session: bool | None,
    optimized_session: bool | None,
    baseline_environment_workers: int | None,
    optimized_environment_workers: int | None,
    baseline_environment_backend: str | None = "individual",
    optimized_environment_backend: str | None = "individual",
    baseline_gradient_checkpointing: bool | None = True,
    optimized_gradient_checkpointing: bool | None = True,
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
    if comparison_mode == "actor-gradient-checkpointing":
        return (
            baseline_session is True
            and optimized_session is True
            and baseline_environment_workers == 1
            and optimized_environment_workers == 1
            and baseline_environment_backend == "native_batch"
            and optimized_environment_backend == "native_batch"
            and baseline_gradient_checkpointing is True
            and optimized_gradient_checkpointing is False
        )
    raise ValueError(f"unsupported comparison mode: {comparison_mode}")


def _performance_is_valid(
    comparison_mode: str,
    *,
    core_speedup: float | None,
    minimum_core_speedup: float,
) -> bool:
    if comparison_mode != "actor-gradient-checkpointing":
        return True
    return core_speedup is not None and core_speedup >= minimum_core_speedup


def _latest_actor_checkpoint(run_directory: Path) -> Path:
    checkpoints = sorted(
        path / "runtime" / "actor"
        for path in (run_directory / "checkpoints").glob("step-*")
        if (path / "checkpoint.complete.json").is_file()
    )
    if not checkpoints:
        raise RuntimeError(f"no complete actor checkpoint under {run_directory}")
    return checkpoints[-1]


def _compare_nested_values(
    baseline: object,
    optimized: object,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    import torch

    mismatches: list[str] = []
    tensor_count = 0
    max_abs_error = 0.0

    def visit(left: object, right: object, path: str) -> None:
        nonlocal tensor_count, max_abs_error
        if len(mismatches) >= 20:
            return
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            tensor_count += 1
            if left.shape != right.shape or left.dtype != right.dtype:
                mismatches.append(
                    f"{path}: tensor {tuple(left.shape)}/{left.dtype} != "
                    f"{tuple(right.shape)}/{right.dtype}"
                )
                return
            if left.numel():
                difference = (
                    left.detach().cpu().float()
                    - right.detach().cpu().float()
                ).abs()
                max_abs_error = max(max_abs_error, float(difference.max().item()))
            if not torch.allclose(
                left.detach().cpu(),
                right.detach().cpu(),
                atol=atol,
                rtol=rtol,
            ):
                mismatches.append(f"{path}: tensor values exceed tolerance")
            return
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                mismatches.append(f"{path}: mapping keys differ")
                return
            for key in sorted(left, key=str):
                visit(left[key], right[key], f"{path}[{key!r}]")
            return
        if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
            if len(left) != len(right):
                mismatches.append(f"{path}: sequence lengths differ")
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{path}[{index}]")
            return
        if type(left) is not type(right) or left != right:
            mismatches.append(f"{path}: non-tensor values differ")

    visit(baseline, optimized, "$")
    return {
        "passed": not mismatches,
        "tensor_count": tensor_count,
        "max_abs_error": max_abs_error,
        "atol": atol,
        "rtol": rtol,
        "mismatches": mismatches,
    }


def _compare_actor_checkpoints(
    baseline_run: Path,
    optimized_run: Path,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    import torch
    from safetensors.torch import load_file

    baseline = _latest_actor_checkpoint(baseline_run)
    optimized = _latest_actor_checkpoint(optimized_run)
    adapter = _compare_nested_values(
        load_file(str(baseline / "adapter_model.safetensors"), device="cpu"),
        load_file(str(optimized / "adapter_model.safetensors"), device="cpu"),
        atol=atol,
        rtol=rtol,
    )
    optimizer = _compare_nested_values(
        torch.load(
            baseline / "lora_optimizer_full.pt",
            map_location="cpu",
            weights_only=True,
        ),
        torch.load(
            optimized / "lora_optimizer_full.pt",
            map_location="cpu",
            weights_only=True,
        ),
        atol=atol,
        rtol=rtol,
    )
    return {
        "passed": bool(adapter["passed"] and optimizer["passed"]),
        "baseline_checkpoint": str(baseline),
        "optimized_checkpoint": str(optimized),
        "adapter": adapter,
        "optimizer": optimizer,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--logprob-tolerance", type=float, default=1e-3)
    parser.add_argument("--minimum-core-speedup", type=float, default=1.03)
    parser.add_argument("--checkpoint-atol", type=float, default=1e-6)
    parser.add_argument("--checkpoint-rtol", type=float, default=1e-5)
    parser.add_argument(
        "--comparison-mode",
        choices=(
            "persistent-session",
            "environment-workers",
            "native-batch",
            "actor-gradient-checkpointing",
        ),
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
    baseline_gradient_checkpointing = _gradient_checkpointing_setting(
        baseline_options
    )
    optimized_gradient_checkpointing = _gradient_checkpointing_setting(
        optimized_options
    )
    baseline_performance = _training_performance(args.baseline)
    optimized_performance = _training_performance(args.optimized)
    baseline_core = baseline_performance["core_seconds"]
    optimized_core = optimized_performance["core_seconds"]
    core_speedup = (
        baseline_core / optimized_core
        if baseline_core is not None
        and optimized_core is not None
        and optimized_core > 0.0
        else None
    )
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
        baseline_gradient_checkpointing=baseline_gradient_checkpointing,
        optimized_gradient_checkpointing=optimized_gradient_checkpointing,
    )
    performance_valid = _performance_is_valid(
        args.comparison_mode,
        core_speedup=core_speedup,
        minimum_core_speedup=args.minimum_core_speedup,
    )
    checkpoint_comparison = (
        _compare_actor_checkpoints(
            args.baseline,
            args.optimized,
            atol=args.checkpoint_atol,
            rtol=args.checkpoint_rtol,
        )
        if args.comparison_mode == "actor-gradient-checkpointing"
        else {"passed": True, "required": False}
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
            "baseline_actor_gradient_checkpointing": (
                baseline_gradient_checkpointing
            ),
            "optimized_actor_gradient_checkpointing": (
                optimized_gradient_checkpointing
            ),
            "baseline_performance": baseline_performance,
            "optimized_performance": optimized_performance,
            "core_speedup": core_speedup,
            "minimum_core_speedup": args.minimum_core_speedup,
            "performance_valid": performance_valid,
            "checkpoint_comparison": checkpoint_comparison,
            "settings_valid": settings_valid,
        }
    )
    report["passed"] = bool(
        report["passed"]
        and settings_valid
        and performance_valid
        and checkpoint_comparison["passed"]
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
