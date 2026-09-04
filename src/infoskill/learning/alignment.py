from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def summarize_logprob_alignment(
    *,
    rollout: Sequence[Sequence[float]],
    recomputed: Sequence[Sequence[float]],
    mask: Sequence[Sequence[bool]],
) -> dict[str, float | int]:
    if not (len(rollout) == len(recomputed) == len(mask)):
        raise ValueError("logprob alignment batch sizes differ")
    deltas: list[float] = []
    for rollout_row, recomputed_row, mask_row in zip(rollout, recomputed, mask):
        if not (len(rollout_row) == len(recomputed_row) == len(mask_row)):
            raise ValueError("logprob alignment row widths differ")
        for rollout_value, recomputed_value, active in zip(
            rollout_row, recomputed_row, mask_row
        ):
            if not active:
                continue
            delta = float(recomputed_value) - float(rollout_value)
            if not math.isfinite(delta):
                raise ValueError("logprob alignment contains a non-finite value")
            deltas.append(delta)
    if not deltas:
        raise ValueError("logprob alignment requires at least one active token")
    absolute = [abs(value) for value in deltas]
    ratios = [math.exp(value) for value in deltas]
    return {
        "token_count": len(deltas),
        "logprob_abs_error_mean": statistics.fmean(absolute),
        "logprob_abs_error_max": max(absolute),
        "ratio_mean": statistics.fmean(ratios),
        "ratio_max_abs_deviation": max(abs(value - 1.0) for value in ratios),
    }
