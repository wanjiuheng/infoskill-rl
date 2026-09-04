from __future__ import annotations

import math
from typing import Sequence


def group_relative_advantages(
    rewards: Sequence[float],
    *,
    epsilon: float = 1e-6,
) -> tuple[float, ...]:
    """Normalize one task group's episodic rewards using sample standard deviation."""

    if len(rewards) < 2:
        raise ValueError("a GRPO group requires at least two trajectory rewards")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    values = tuple(float(reward) for reward in rewards)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("trajectory rewards must be finite")
    mean = math.fsum(values) / len(values)
    squared_error = math.fsum((value - mean) ** 2 for value in values)
    sample_variance = squared_error / (len(values) - 1)
    if sample_variance == 0.0:
        return tuple(0.0 for _ in values)
    denominator = math.sqrt(sample_variance) + epsilon
    return tuple((value - mean) / denominator for value in values)
