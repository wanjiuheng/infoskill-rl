#!/usr/bin/env python3
"""Gate a one-update INFO-SKILL throughput candidate against a control fork."""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from scripts.compare_rollout_session_runs import compare_records
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from compare_rollout_session_runs import compare_records


_CANDIDATE_OPTIONS = {
    "fuse_kl_ppo_forward",
    "hybrid_prefix_cuda_graph",
    "skip_unused_old_logprob_entropy",
    "rollout_max_batched_tokens",
    "policy_gradient_clip_mode",
}
_CANDIDATE_MODES = (
    "combined",
    "entropy-only",
    "cuda-graph",
    "fused-kl-ppo",
    "deep-combined",
    "separate-grad-clip",
)
_ALGORITHM_CANDIDATE_MODES = {
    "fused-kl-ppo",
    "deep-combined",
    "separate-grad-clip",
}
_FUSED_KL_PPO_MODES = {"fused-kl-ppo", "deep-combined"}


def compare_runs(
    baseline: Path,
    candidate: Path,
    *,
    candidate_mode: str = "combined",
    logprob_tolerance: float = 1e-3,
    checkpoint_atol: float = 1e-7,
    checkpoint_rtol: float = 1e-6,
    minimum_core_speedup: float = 1.05,
    minimum_physical_free_gb: float = 8.0,
) -> dict[str, object]:
    if candidate_mode not in _CANDIDATE_MODES:
        raise ValueError(f"unsupported candidate mode: {candidate_mode}")
    baseline_resolved = _read_json(baseline / "resolved_config.json")
    candidate_resolved = _read_json(candidate / "resolved_config.json")
    baseline_provenance = _read_json(baseline / "provenance.json")
    candidate_provenance = _read_json(candidate / "provenance.json")
    baseline_summary = _read_json(baseline / "training_summary.json")
    candidate_summary = _read_json(candidate / "training_summary.json")
    baseline_metric = _single_training_metric(baseline)
    candidate_metric = _single_training_metric(candidate)
    baseline_options = _runtime_options(baseline_resolved)
    candidate_options = _runtime_options(candidate_resolved)
    baseline_update = _positive_int(baseline_summary.get("global_update"))
    candidate_update = _positive_int(candidate_summary.get("global_update"))

    baseline_invocation = baseline_provenance.get("invocation")
    candidate_invocation = candidate_provenance.get("invocation")
    source_checkpoint = baseline_provenance.get("resume_source_checkpoint")
    source_update = _checkpoint_step(source_checkpoint)
    control_checks = {
        "both_infoskill": (
            baseline_resolved.get("mode") == "infoskill"
            and candidate_resolved.get("mode") == "infoskill"
        ),
        "same_controlled_config": (
            _without_candidate_options(baseline_resolved)
            == _without_candidate_options(candidate_resolved)
        ),
        "same_source_checkpoint": (
            isinstance(source_checkpoint, str)
            and source_checkpoint
            == candidate_provenance.get("resume_source_checkpoint")
        ),
        "both_named_forks": (
            baseline_provenance.get("resume_forked") is True
            and candidate_provenance.get("resume_forked") is True
        ),
        "same_one_update_segment": (
            isinstance(baseline_invocation, dict)
            and baseline_invocation == candidate_invocation
            and baseline_invocation.get("segment_start_update") == source_update
            and baseline_invocation.get("segment_end_update") == source_update + 1
        ),
        "both_paused_after_one_update": (
            baseline_summary.get("status") == "paused"
            and candidate_summary.get("status") == "paused"
            and baseline_update == source_update + 1
            and candidate_update == source_update + 1
            and baseline_metric.get("step") == source_update + 1
            and candidate_metric.get("step") == source_update + 1
        ),
        "both_checkpoints_committed_and_portable": (
            _checkpoint_committed(baseline, baseline_update)
            and _checkpoint_committed(candidate, candidate_update)
        ),
        "baseline_registered_defaults": (
            baseline_options.get("skip_unused_old_logprob_entropy", False)
            is False
            and baseline_options.get("rollout_max_batched_tokens", 16_384)
            == 16_384
            and baseline_options.get("hybrid_prefix_cuda_graph", False)
            is False
            and baseline_options.get("fuse_kl_ppo_forward", False) is False
            and baseline_options.get("policy_gradient_clip_mode", "joint")
            == "joint"
        ),
        "physical_memory_sampling_enabled": (
            _positive_int(baseline_options.get("cuda_memory_poll_interval_ms")) > 0
            and baseline_options.get("cuda_memory_poll_interval_ms")
            == candidate_options.get("cuda_memory_poll_interval_ms")
        ),
        "runtime_reports_expected_entropy_paths": (
            float(baseline_metric.get("perf/old_logprob_entropy_skipped", 0.0))
            == 0.0
            and float(candidate_metric.get("perf/old_logprob_entropy_skipped", 0.0))
            == float(
                candidate_mode in {"combined", "entropy-only", "deep-combined"}
            )
        ),
        "runtime_reports_expected_cuda_graph_path": (
            float(
                baseline_metric.get("perf/hybrid_prefix_cuda_graph", 0.0)
            )
            == 0.0
            and float(
                candidate_metric.get("perf/hybrid_prefix_cuda_graph", 0.0)
            )
            == float(candidate_mode in {"cuda-graph", "deep-combined"})
        ),
        "runtime_reports_expected_kl_ppo_path": (
            float(
                baseline_metric.get("perf/fuse_kl_ppo_forward", 0.0)
            )
            == 0.0
            and float(
                candidate_metric.get("perf/fuse_kl_ppo_forward", 0.0)
            )
            == float(candidate_mode in _FUSED_KL_PPO_MODES)
        ),
        "runtime_reports_expected_gradient_clip_path": (
            float(
                baseline_metric.get("policy/separate_gradient_clipping", 0.0)
            )
            == 0.0
            and float(
                candidate_metric.get("policy/separate_gradient_clipping", 0.0)
            )
            == float(candidate_mode == "separate-grad-clip")
        ),
    }
    control_checks.update(
        _candidate_option_checks(
            baseline_options,
            candidate_options,
            candidate_mode=candidate_mode,
        )
    )

    trace_comparison = compare_records(
        _read_training_trace(baseline, baseline_update),
        _read_training_trace(candidate, candidate_update),
        logprob_tolerance=logprob_tolerance,
    )
    checkpoint_comparison = _compare_checkpoints(
        baseline / "checkpoints" / f"step-{baseline_update:06d}",
        candidate / "checkpoints" / f"step-{candidate_update:06d}",
        atol=checkpoint_atol,
        rtol=checkpoint_rtol,
    )
    baseline_performance = _performance(baseline_metric)
    candidate_performance = _performance(candidate_metric)
    core_speedup = _speedup(
        baseline_performance["core_seconds"],
        candidate_performance["core_seconds"],
    )
    rollout_speedup = _speedup(
        baseline_performance["rollout_seconds"],
        candidate_performance["rollout_seconds"],
    )
    policy_speedup = _speedup(
        baseline_performance["policy_update_seconds"],
        candidate_performance["policy_update_seconds"],
    )
    old_logprob_speedup = _speedup(
        baseline_performance["old_logprob_seconds"],
        candidate_performance["old_logprob_seconds"],
    )
    normalized_speedups = {
        name: _throughput_speedup(
            baseline_performance[metric],
            candidate_performance[metric],
            baseline_performance["training_sample_count"],
            candidate_performance["training_sample_count"],
        )
        for name, metric in {
            "core": "core_seconds",
            "rollout": "rollout_seconds",
            "generation": "generation_seconds",
            "policy": "policy_update_seconds",
            "old_logprob": "old_logprob_seconds",
        }.items()
    }
    candidate_free_values = [
        candidate_performance["policy_physical_min_free_gb"],
        candidate_performance["rollout_physical_min_free_gb"],
    ]
    candidate_physical_free = (
        min(float(value) for value in candidate_free_values if value is not None)
        if all(value is not None for value in candidate_free_values)
        else None
    )
    settings_valid = all(control_checks.values())
    performance_valid = (
        core_speedup is not None and core_speedup >= minimum_core_speedup
    )
    physical_memory_valid = (
        candidate_physical_free is not None
        and candidate_physical_free >= minimum_physical_free_gb
    )
    trace_exact = bool(trace_comparison["passed"])
    checkpoint_equivalent = bool(checkpoint_comparison["passed"])
    performance_comparison_valid = trace_exact
    algorithm_change_requested = candidate_mode in _ALGORITHM_CANDIDATE_MODES
    equivalent_optimization_passed = (
        settings_valid
        and not algorithm_change_requested
        and trace_exact
        and checkpoint_equivalent
        and performance_valid
        and physical_memory_valid
    )
    behavior_change_detected = not trace_exact
    efficacy_gate_required = settings_valid and (
        behavior_change_detected or algorithm_change_requested
    )
    return {
        "schema_version": 1,
        "baseline": str(baseline.resolve()),
        "candidate": str(candidate.resolve()),
        "source_checkpoint": source_checkpoint,
        "source_update": source_update,
        "target_update": baseline_update,
        "candidate_mode": candidate_mode,
        "diagnostic_only": True,
        "control_checks": control_checks,
        "settings_valid": settings_valid,
        "trace_comparison": trace_comparison,
        "checkpoint_comparison": checkpoint_comparison,
        "baseline_performance": baseline_performance,
        "candidate_performance": candidate_performance,
        "core_speedup": core_speedup,
        "rollout_speedup": rollout_speedup,
        "policy_speedup": policy_speedup,
        "old_logprob_speedup": old_logprob_speedup,
        "training_sample_normalized_speedups": normalized_speedups,
        "minimum_core_speedup": minimum_core_speedup,
        "performance_valid": performance_valid,
        "performance_comparison_valid": performance_comparison_valid,
        "candidate_physical_min_free_gb": candidate_physical_free,
        "minimum_physical_free_gb": minimum_physical_free_gb,
        "physical_memory_valid": physical_memory_valid,
        "algorithm_change_requested": algorithm_change_requested,
        "behavior_change_detected": behavior_change_detected,
        "efficacy_gate_required": efficacy_gate_required,
        "equivalent_optimization_passed": equivalent_optimization_passed,
        "safe_to_continue_candidate": equivalent_optimization_passed,
        "passed": equivalent_optimization_passed,
    }


def _candidate_option_checks(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    candidate_mode: str,
) -> dict[str, bool]:
    baseline_capacity = _positive_int(
        baseline.get("rollout_max_batched_tokens", 16_384)
    )
    candidate_capacity = _positive_int(candidate.get("rollout_max_batched_tokens"))
    expected_entropy = candidate_mode in {
        "combined",
        "entropy-only",
        "deep-combined",
    }
    expected_cuda_graph = candidate_mode in {"cuda-graph", "deep-combined"}
    expected_fused_kl_ppo = candidate_mode in _FUSED_KL_PPO_MODES
    expected_gradient_clip_mode = (
        "separate" if candidate_mode == "separate-grad-clip" else "joint"
    )
    if candidate_mode == "combined":
        capacity_matches_mode = candidate_capacity > baseline_capacity
    elif candidate_mode in {
        "entropy-only",
        "cuda-graph",
        "fused-kl-ppo",
        "deep-combined",
        "separate-grad-clip",
    }:
        capacity_matches_mode = candidate_capacity == baseline_capacity
    else:
        raise ValueError(f"unsupported candidate mode: {candidate_mode}")
    return {
        "candidate_entropy_skip_matches_mode": (
            candidate.get("skip_unused_old_logprob_entropy", False)
            is expected_entropy
        ),
        "candidate_rollout_capacity_matches_mode": capacity_matches_mode,
        "candidate_cuda_graph_matches_mode": (
            candidate.get("hybrid_prefix_cuda_graph", False)
            is expected_cuda_graph
        ),
        "candidate_fused_kl_ppo_matches_mode": (
            candidate.get("fuse_kl_ppo_forward", False)
            is expected_fused_kl_ppo
        ),
        "candidate_gradient_clip_mode_matches_mode": (
            candidate.get("policy_gradient_clip_mode", "joint")
            == expected_gradient_clip_mode
        ),
    }


def _without_candidate_options(payload: dict[str, object]) -> dict[str, object]:
    normalized = json.loads(json.dumps(payload))
    options = normalized.get("runtime_options")
    if isinstance(options, dict):
        for name in _CANDIDATE_OPTIONS:
            options.pop(name, None)
    return normalized


def _runtime_options(payload: dict[str, object]) -> dict[str, object]:
    options = payload.get("runtime_options")
    return dict(options) if isinstance(options, dict) else {}


def _single_training_metric(run: Path) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    training = [record for record in records if record.get("phase") == "train"]
    if len(training) != 1:
        raise RuntimeError(
            f"expected exactly one training metric in {run}, found {len(training)}"
        )
    return training[0]


def _read_training_trace(run: Path, update: int) -> list[dict]:
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("trace comparison requires zstandard") from error
    path = run / "traces" / f"train-update-{update:06d}-rank-000.jsonl.zst"
    if not path.is_file():
        raise RuntimeError(f"training trace is missing: {path}")
    records: list[dict] = []
    with path.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                records.append(json.loads(line))
    return records


def _performance(metric: dict[str, object]) -> dict[str, float | None]:
    def value(name: str) -> float | None:
        raw = metric.get(name)
        return float(raw) if raw is not None else None

    return {
        "core_seconds": value("perf/core_update_seconds"),
        "rollout_seconds": value("perf/rollout_seconds"),
        "generation_seconds": value("perf/rollout_backend_generate_seconds"),
        "policy_update_seconds": value("perf/policy_update_seconds"),
        "old_logprob_seconds": value("perf/old_logprob_seconds"),
        "reference_logprob_seconds": value("perf/reference_logprob_seconds"),
        "actor_update_seconds": value("perf/actor_update_seconds"),
        "auxiliary_update_seconds": value("perf/auxiliary_update_seconds"),
        "training_sample_count": value("runtime/training_sample_count"),
        "rollout_mean_steps": value("rollout/mean_steps"),
        "policy_physical_min_free_gb": value(
            "perf/cuda/policy_physical_min_free_gb_min"
        ),
        "rollout_physical_min_free_gb": value(
            "perf/cuda/rollout_physical_min_free_gb_min"
        ),
    }


def _compare_checkpoints(
    baseline: Path,
    candidate: Path,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError(
            "checkpoint comparison requires torch and safetensors"
        ) from error

    files = (
        "runtime/actor/adapter_model.safetensors",
        "runtime/actor/lora_optimizer_full.pt",
        "runtime/actor/lora_scheduler.pt",
        "runtime/actor/infoskill/infoskill_modules.pt",
        "runtime/actor/infoskill/infoskill_optimizers.pt",
        "runtime/actor/infoskill/infoskill_schedulers.pt",
        "runtime/actor/infoskill/infoskill_rng_state.pt",
        "trainer_state.json",
    )
    mismatches: list[str] = []
    compared_tensors = 0
    exactly_equal_tensors = 0
    max_abs_error = 0.0

    for relative in files:
        left_path = baseline / relative
        right_path = candidate / relative
        if not left_path.is_file() or not right_path.is_file():
            mismatches.append(f"{relative}: missing from one or both checkpoints")
            continue
        if relative.endswith(".json"):
            left: Any = _read_json(left_path)
            right: Any = _read_json(right_path)
        elif relative.endswith(".safetensors"):
            left = load_file(str(left_path), device="cpu")
            right = load_file(str(right_path), device="cpu")
        else:
            left = torch.load(left_path, map_location="cpu", weights_only=False)
            right = torch.load(right_path, map_location="cpu", weights_only=False)
        result = _compare_values(
            left,
            right,
            path=relative,
            atol=atol,
            rtol=rtol,
            mismatch_limit=max(0, 20 - len(mismatches)),
        )
        mismatches.extend(result["mismatches"])
        compared_tensors += result["compared_tensors"]
        exactly_equal_tensors += result["exactly_equal_tensors"]
        max_abs_error = max(max_abs_error, result["max_abs_error"])

    return {
        "passed": not mismatches,
        "baseline_checkpoint": str(baseline.resolve()),
        "candidate_checkpoint": str(candidate.resolve()),
        "files_compared": list(files),
        "compared_tensors": compared_tensors,
        "exactly_equal_tensors": exactly_equal_tensors,
        "all_tensors_exact": compared_tensors == exactly_equal_tensors,
        "max_tensor_abs_error": max_abs_error,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "mismatches": mismatches[:20],
    }


def _compare_values(
    left: Any,
    right: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
    mismatch_limit: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    mismatches: list[str] = []
    compared_tensors = 0
    exactly_equal_tensors = 0
    max_abs_error = 0.0

    def mismatch(message: str) -> None:
        if len(mismatches) < mismatch_limit:
            mismatches.append(message)

    def visit(a: Any, b: Any, current: str) -> None:
        nonlocal compared_tensors, exactly_equal_tensors, max_abs_error
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                mismatch(f"{current}: tensor/non-tensor type mismatch")
                return
            compared_tensors += 1
            if a.shape != b.shape or a.dtype != b.dtype:
                mismatch(
                    f"{current}: tensor geometry {tuple(a.shape)}/{a.dtype} != "
                    f"{tuple(b.shape)}/{b.dtype}"
                )
                return
            exact = torch.equal(a, b)
            if exact:
                exactly_equal_tensors += 1
                return
            if a.is_floating_point() or a.is_complex():
                difference = float((a - b).abs().max().item()) if a.numel() else 0.0
                max_abs_error = max(max_abs_error, difference)
                if torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
                    return
                mismatch(f"{current}: tensor max_abs_error={difference:.9g}")
                return
            mismatch(f"{current}: non-floating tensor values differ")
            return
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
                mismatch(f"{current}: ndarray/non-ndarray type mismatch")
            elif a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
                mismatch(f"{current}: ndarray values differ")
            return
        if isinstance(a, Mapping) or isinstance(b, Mapping):
            if not isinstance(a, Mapping) or not isinstance(b, Mapping):
                mismatch(f"{current}: mapping/non-mapping type mismatch")
                return
            if set(a) != set(b):
                mismatch(f"{current}: mapping keys differ")
                return
            for key in sorted(a, key=str):
                visit(a[key], b[key], f"{current}.{key}")
            return
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
                mismatch(f"{current}: sequence/non-sequence type mismatch")
                return
            if len(a) != len(b):
                mismatch(f"{current}: sequence lengths {len(a)} != {len(b)}")
                return
            for index, (left_item, right_item) in enumerate(zip(a, b, strict=True)):
                visit(left_item, right_item, f"{current}[{index}]")
            return
        if type(a) is not type(b) or a != b:
            mismatch(f"{current}: values differ ({a!r} != {b!r})")

    visit(left, right, path)
    return {
        "mismatches": mismatches,
        "compared_tensors": compared_tensors,
        "exactly_equal_tensors": exactly_equal_tensors,
        "max_abs_error": max_abs_error,
    }


def _checkpoint_step(value: object) -> int:
    if not isinstance(value, str):
        return -1
    name = Path(value).name
    if not name.startswith("step-"):
        return -1
    try:
        return int(name.removeprefix("step-"))
    except ValueError:
        return -1


def _checkpoint_committed(run: Path, update: int) -> bool:
    marker = (
        run
        / "checkpoints"
        / f"step-{update:06d}"
        / "checkpoint.complete.json"
    )
    if update <= 0 or not marker.is_file():
        return False
    payload = _read_json(marker)
    runtime = payload.get("runtime_manifest")
    files = payload.get("files")
    required = {
        "runtime/actor/adapter_model.safetensors",
        "runtime/actor/lora_optimizer_full.pt",
        "runtime/actor/lora_scheduler.pt",
        "runtime/actor/infoskill/infoskill_modules.pt",
        "runtime/actor/infoskill/infoskill_optimizers.pt",
        "runtime/actor/infoskill/infoskill_schedulers.pt",
        "runtime/actor/infoskill/infoskill_rng_state.pt",
        "trainer_state.json",
    }
    return (
        payload.get("global_update") == update
        and payload.get("portable") is True
        and payload.get("emergency") is False
        and isinstance(runtime, dict)
        and runtime.get("portable") is True
        and runtime.get("infoskill_modules_included") is True
        and isinstance(files, list)
        and required.issubset(files)
    )


def _positive_int(value: object) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else -1
    )


def _speedup(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or candidate <= 0.0:
        return None
    return baseline / candidate


def _throughput_speedup(
    baseline_seconds: float | None,
    candidate_seconds: float | None,
    baseline_work: float | None,
    candidate_work: float | None,
) -> float | None:
    if (
        baseline_seconds is None
        or candidate_seconds is None
        or baseline_work is None
        or candidate_work is None
        or baseline_seconds <= 0.0
        or candidate_seconds <= 0.0
        or baseline_work <= 0.0
        or candidate_work <= 0.0
    ):
        return None
    return (candidate_work / candidate_seconds) / (baseline_work / baseline_seconds)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--candidate-mode",
        choices=_CANDIDATE_MODES,
        default="combined",
    )
    parser.add_argument("--logprob-tolerance", type=float, default=1e-3)
    parser.add_argument("--checkpoint-atol", type=float, default=1e-7)
    parser.add_argument("--checkpoint-rtol", type=float, default=1e-6)
    parser.add_argument("--minimum-core-speedup", type=float, default=1.05)
    parser.add_argument("--minimum-physical-free-gb", type=float, default=8.0)
    args = parser.parse_args()
    report = compare_runs(
        args.baseline,
        args.candidate,
        candidate_mode=args.candidate_mode,
        logprob_tolerance=args.logprob_tolerance,
        checkpoint_atol=args.checkpoint_atol,
        checkpoint_rtol=args.checkpoint_rtol,
        minimum_core_speedup=args.minimum_core_speedup,
        minimum_physical_free_gb=args.minimum_physical_free_gb,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
