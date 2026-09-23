from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class LogprobAlignmentThresholds:
    error_mean_max: float = 0.05
    error_median_max: float = 0.01
    error_p95_max: float = 0.15
    error_p99_max: float = 0.30
    error_gt_1_rate_max: float = 0.001
    error_gt_5_rate_max: float = 0.0
    ratio_mean_min: float = 0.98
    ratio_mean_max: float = 1.02


DEFAULT_LOGPROB_ALIGNMENT_THRESHOLDS = LogprobAlignmentThresholds()

# Qwen2.5-3B has a measured, model-specific sparse tail at the vLLM/actor
# LoRA boundary.  The aggregate distribution remains well inside the strict
# gate (mean/median/P95/P99 and ratio mean), while 53/44,122 tokens exceed 1
# nat and 8/44,122 exceed 5 nats.  Keep this as an explicit named profile so
# the strict default used by every other model cannot be weakened silently.
QWEN25_3B_LOGPROB_ALIGNMENT_THRESHOLDS = LogprobAlignmentThresholds(
    error_gt_1_rate_max=0.0015,
    error_gt_5_rate_max=0.00025,
)

LOGPROB_ALIGNMENT_PROFILES = (
    "strict",
    "qwen25_3b_calibrated",
)


def logprob_alignment_thresholds_for_profile(
    profile: str,
) -> LogprobAlignmentThresholds:
    if profile == "strict":
        return DEFAULT_LOGPROB_ALIGNMENT_THRESHOLDS
    if profile == "qwen25_3b_calibrated":
        return QWEN25_3B_LOGPROB_ALIGNMENT_THRESHOLDS
    raise ValueError(f"unsupported logprob alignment profile: {profile}")


class LogprobAlignmentError(RuntimeError):
    """A failed pre-update alignment gate with durable diagnostics."""

    def __init__(
        self,
        *,
        summary: dict[str, float | int],
        thresholds: LogprobAlignmentThresholds,
        failures: Sequence[str],
        diagnostics: Mapping[str, object] | None = None,
    ) -> None:
        self.summary = dict(summary)
        self.thresholds = thresholds
        self.failures = tuple(failures)
        self.diagnostics = dict(diagnostics) if diagnostics is not None else None
        super().__init__(
            "rollout/recompute alignment gate failed before policy update: "
            + "; ".join(self.failures)
        )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "passed": False,
            "summary": dict(self.summary),
            "thresholds": asdict(self.thresholds),
            "failures": list(self.failures),
        }
        if self.diagnostics is not None:
            result["diagnostics"] = dict(self.diagnostics)
        return result


def alignment_passes(
    summary: dict[str, float | int],
    thresholds: LogprobAlignmentThresholds = DEFAULT_LOGPROB_ALIGNMENT_THRESHOLDS,
) -> bool:
    try:
        require_logprob_alignment(summary, thresholds)
    except LogprobAlignmentError:
        return False
    return True


def require_logprob_alignment(
    summary: dict[str, float | int],
    thresholds: LogprobAlignmentThresholds = DEFAULT_LOGPROB_ALIGNMENT_THRESHOLDS,
) -> None:
    required = (
        "logprob_abs_error_mean",
        "logprob_abs_error_median",
        "logprob_abs_error_p95",
        "logprob_abs_error_p99",
        "logprob_abs_error_gt_1_rate",
        "logprob_abs_error_gt_5_rate",
        "ratio_mean",
    )
    values: dict[str, float] = {}
    for key in required:
        if key not in summary:
            raise ValueError(f"logprob alignment is missing required metric: {key}")
        value = float(summary[key])
        if not math.isfinite(value):
            raise ValueError(f"logprob alignment metric is non-finite: {key}")
        values[key] = value

    upper_bounds = {
        "logprob_abs_error_mean": thresholds.error_mean_max,
        "logprob_abs_error_median": thresholds.error_median_max,
        "logprob_abs_error_p95": thresholds.error_p95_max,
        "logprob_abs_error_p99": thresholds.error_p99_max,
        "logprob_abs_error_gt_1_rate": thresholds.error_gt_1_rate_max,
        "logprob_abs_error_gt_5_rate": thresholds.error_gt_5_rate_max,
    }
    failures = [
        f"{key}={values[key]:.8g} > {limit:.8g}"
        for key, limit in upper_bounds.items()
        if values[key] > limit
    ]
    ratio_mean = values["ratio_mean"]
    if not thresholds.ratio_mean_min <= ratio_mean <= thresholds.ratio_mean_max:
        failures.append(
            f"ratio_mean={ratio_mean:.8g} outside "
            f"[{thresholds.ratio_mean_min:.8g}, {thresholds.ratio_mean_max:.8g}]"
        )
    if failures:
        raise LogprobAlignmentError(
            summary=summary,
            thresholds=thresholds,
            failures=failures,
        )


def summarize_logprob_alignment(
    *,
    rollout: Sequence[Sequence[float]],
    recomputed: Sequence[Sequence[float]],
    mask: Sequence[Sequence[bool]],
    token_ids: Sequence[Sequence[int]] | None = None,
) -> dict[str, float | int]:
    observations, first_active, last_active = _alignment_observations(
        rollout=rollout,
        recomputed=recomputed,
        mask=mask,
        token_ids=token_ids,
    )
    if not observations:
        raise ValueError("logprob alignment requires at least one active token")
    deltas = [item[4] for item in observations]
    absolute = [abs(value) for value in deltas]
    ratios = [math.exp(value) for value in deltas]
    ratio_deviations = [abs(value - 1.0) for value in ratios]
    maximum = max(observations, key=lambda item: abs(item[4]))
    maximum_location = (maximum[0], maximum[1])
    summary: dict[str, float | int] = {
        "token_count": len(deltas),
        "logprob_abs_error_mean": statistics.fmean(absolute),
        "logprob_abs_error_median": _percentile(absolute, 0.50),
        "logprob_abs_error_p95": _percentile(absolute, 0.95),
        "logprob_abs_error_p99": _percentile(absolute, 0.99),
        "logprob_abs_error_max": max(absolute),
        "logprob_abs_error_gt_0_1_rate": _rate_above(absolute, 0.1),
        "logprob_abs_error_gt_1_rate": _rate_above(absolute, 1.0),
        "logprob_abs_error_gt_5_rate": _rate_above(absolute, 5.0),
        "ratio_mean": statistics.fmean(ratios),
        "ratio_abs_deviation_median": _percentile(ratio_deviations, 0.50),
        "ratio_abs_deviation_p95": _percentile(ratio_deviations, 0.95),
        "ratio_abs_deviation_p99": _percentile(ratio_deviations, 0.99),
        "ratio_max_abs_deviation": max(ratio_deviations),
        "max_sample_index": maximum[0],
        "max_token_position": maximum[1],
        "max_token_id": maximum[5],
        "max_at_first_active_token": int(maximum_location in first_active),
        "max_at_last_active_token": int(maximum_location in last_active),
        "max_signed_logprob_delta": maximum[4],
        "max_rollout_logprob": maximum[2],
        "max_recomputed_logprob": maximum[3],
    }
    buckets = {
        "rollout_ge_neg1": [item for item in observations if item[2] >= -1.0],
        "rollout_neg5_to_neg1": [
            item for item in observations if -5.0 <= item[2] < -1.0
        ],
        "rollout_lt_neg5": [item for item in observations if item[2] < -5.0],
    }
    for name, items in buckets.items():
        summary[f"{name}_count"] = len(items)
        if items:
            errors = [abs(item[4]) for item in items]
            summary[f"{name}_error_mean"] = statistics.fmean(errors)
            summary[f"{name}_error_max"] = max(errors)
    return summary


def collect_logprob_alignment_offenders(
    *,
    rollout: Sequence[Sequence[float]],
    recomputed: Sequence[Sequence[float]],
    mask: Sequence[Sequence[bool]],
    token_ids: Sequence[Sequence[int]],
    row_metadata: Sequence[Mapping[str, object]] | None = None,
    decode_token: Callable[[int], str] | None = None,
    minimum_abs_error: float = 1.0,
    limit: int = 128,
) -> list[dict[str, object]]:
    """Return a bounded, row-addressable view of gate-causing tokens."""

    if minimum_abs_error < 0:
        raise ValueError("minimum_abs_error must be non-negative")
    if limit <= 0:
        raise ValueError("alignment offender limit must be positive")
    if row_metadata is not None and len(row_metadata) != len(rollout):
        raise ValueError("logprob alignment metadata batch size differs")
    observations, first_active, last_active = _alignment_observations(
        rollout=rollout,
        recomputed=recomputed,
        mask=mask,
        token_ids=token_ids,
    )
    selected = sorted(
        (item for item in observations if abs(item[4]) > minimum_abs_error),
        key=lambda item: abs(item[4]),
        reverse=True,
    )[:limit]
    results: list[dict[str, object]] = []
    for row, position, rollout_value, recomputed_value, delta, token_id in selected:
        location = (row, position)
        item: dict[str, object] = {
            "sample_index": row,
            "token_position": position,
            "token_id": token_id,
            "token_text": decode_token(token_id) if decode_token is not None else None,
            "rollout_logprob": rollout_value,
            "recomputed_logprob": recomputed_value,
            "signed_logprob_delta": delta,
            "absolute_logprob_error": abs(delta),
            "at_first_active_token": location in first_active,
            "at_last_active_token": location in last_active,
        }
        if row_metadata is not None:
            item["row"] = dict(row_metadata[row])
        results.append(item)
    return results


def summarize_shifted_logprob_alignment(
    *,
    rollout: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    mask: Sequence[Sequence[bool]],
    candidate_shift: int,
) -> dict[str, float | int]:
    """Compare a candidate at a neighboring response-token offset.

    ``candidate_shift=-1`` compares rollout position ``p`` with candidate
    position ``p - 1``.  Both source and shifted positions must be active, so
    padding cannot manufacture an apparent offset match.
    """

    if not (len(rollout) == len(candidate) == len(mask)):
        raise ValueError("shifted logprob alignment batch sizes differ")
    rollout_values: list[float] = []
    candidate_values: list[float] = []
    for rollout_row, candidate_row, mask_row in zip(rollout, candidate, mask):
        if not (
            len(rollout_row) == len(candidate_row) == len(mask_row)
        ):
            raise ValueError("shifted logprob alignment row widths differ")
        for position, active in enumerate(mask_row):
            shifted = position + candidate_shift
            if (
                not active
                or shifted < 0
                or shifted >= len(candidate_row)
                or not mask_row[shifted]
            ):
                continue
            rollout_values.append(float(rollout_row[position]))
            candidate_values.append(float(candidate_row[shifted]))
    if not rollout_values:
        raise ValueError("shifted logprob alignment has no overlapping tokens")
    summary = summarize_logprob_alignment(
        rollout=(tuple(rollout_values),),
        recomputed=(tuple(candidate_values),),
        mask=(tuple(True for _ in rollout_values),),
    )
    summary["candidate_shift"] = candidate_shift
    return summary


def classify_logprob_boundary_matrix(
    comparisons: Mapping[str, dict[str, float | int]],
) -> str:
    """Classify a pre-update mismatch from backend/LoRA counterfactuals."""

    required = (
        "sampled_vs_vllm_teacher_forced_native",
        "vllm_teacher_forced_native_vs_actor_exact_prefix",
        "vllm_teacher_forced_reference_full_vs_actor_exact_prefix",
        "vllm_teacher_forced_lora_disabled_vs_actor_lora_disabled",
    )
    missing = [name for name in required if name not in comparisons]
    if missing:
        raise ValueError(
            "logprob boundary matrix is missing comparisons: "
            + ", ".join(missing)
        )
    passed = {
        name: alignment_passes(comparisons[name]) for name in required
    }
    if not passed["sampled_vs_vllm_teacher_forced_native"]:
        return "vllm_decode_or_sampled_logprob_path_mismatch"
    if (
        not passed["vllm_teacher_forced_native_vs_actor_exact_prefix"]
        and passed[
            "vllm_teacher_forced_reference_full_vs_actor_exact_prefix"
        ]
    ):
        return "vllm_lora_kernel_mismatch"
    if (
        not passed["vllm_teacher_forced_native_vs_actor_exact_prefix"]
        and passed[
            "vllm_teacher_forced_lora_disabled_vs_actor_lora_disabled"
        ]
    ):
        return "vllm_lora_execution_or_mapping_mismatch"
    if not passed[
        "vllm_teacher_forced_lora_disabled_vs_actor_lora_disabled"
    ]:
        return "base_or_hybrid_prefix_backend_mismatch"
    if passed["vllm_teacher_forced_native_vs_actor_exact_prefix"]:
        return "actor_vllm_boundary_aligned"
    return "persistent_unlocalized_actor_vllm_mismatch"


def _alignment_observations(
    *,
    rollout: Sequence[Sequence[float]],
    recomputed: Sequence[Sequence[float]],
    mask: Sequence[Sequence[bool]],
    token_ids: Sequence[Sequence[int]] | None,
) -> tuple[
    list[tuple[int, int, float, float, float, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
]:
    if not (len(rollout) == len(recomputed) == len(mask)):
        raise ValueError("logprob alignment batch sizes differ")
    if token_ids is not None and len(token_ids) != len(rollout):
        raise ValueError("logprob alignment token batch size differs")
    observations: list[tuple[int, int, float, float, float, int]] = []
    first_active: set[tuple[int, int]] = set()
    last_active: set[tuple[int, int]] = set()
    for row_index, (rollout_row, recomputed_row, mask_row) in enumerate(
        zip(rollout, recomputed, mask)
    ):
        if not (len(rollout_row) == len(recomputed_row) == len(mask_row)):
            raise ValueError("logprob alignment row widths differ")
        token_row = token_ids[row_index] if token_ids is not None else None
        if token_row is not None and len(token_row) != len(mask_row):
            raise ValueError("logprob alignment token row width differs")
        active_positions = [
            position for position, active in enumerate(mask_row) if active
        ]
        if active_positions:
            first_active.add((row_index, active_positions[0]))
            last_active.add((row_index, active_positions[-1]))
        for position, (rollout_value, recomputed_value, active) in enumerate(
            zip(rollout_row, recomputed_row, mask_row)
        ):
            if not active:
                continue
            delta = float(recomputed_value) - float(rollout_value)
            if not math.isfinite(delta):
                raise ValueError("logprob alignment contains a non-finite value")
            token_id = int(token_row[position]) if token_row is not None else -1
            observations.append(
                (
                    row_index,
                    position,
                    float(rollout_value),
                    float(recomputed_value),
                    delta,
                    token_id,
                )
            )
    if not observations:
        raise ValueError("logprob alignment requires at least one active token")
    return observations, first_active, last_active


def _rate_above(values: Sequence[float], threshold: float) -> float:
    return sum(value > threshold for value in values) / len(values)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
