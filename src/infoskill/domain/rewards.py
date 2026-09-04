from __future__ import annotations


def trajectory_reward(
    *,
    won: bool,
    invalid_action_count: int,
    invalid_action_penalty: float = 0.01,
) -> float:
    """Return the registered episodic reward without a step penalty."""

    if invalid_action_count < 0:
        raise ValueError("invalid_action_count must be non-negative")
    if invalid_action_penalty < 0:
        raise ValueError("invalid_action_penalty must be non-negative")
    return float(won) - invalid_action_penalty * invalid_action_count
