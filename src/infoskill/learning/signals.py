from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from infoskill.episode import TrajectoryGroup

from .advantages import group_relative_advantages


@dataclass(frozen=True, slots=True)
class GroupAdvantageSignals:
    """Two deliberately different group-relative learning signals."""

    shaped_policy: tuple[float, ...]
    task_success: tuple[float, ...]


def build_group_advantage_signals(
    groups: Sequence[TrajectoryGroup],
) -> tuple[GroupAdvantageSignals, ...]:
    """Build the policy signal from shaped reward and fidelity target from success."""

    return tuple(
        GroupAdvantageSignals(
            shaped_policy=group_relative_advantages(
                [trajectory.reward for trajectory in group.trajectories]
            ),
            task_success=group_relative_advantages(
                [float(trajectory.won) for trajectory in group.trajectories]
            ),
        )
        for group in groups
    )


def summarize_grpo_signals(
    groups: Sequence[TrajectoryGroup],
    signals: Sequence[GroupAdvantageSignals],
) -> dict[str, float]:
    """Summarize whether each group carries task or shaping-only information."""

    if len(groups) != len(signals):
        raise ValueError("groups and signals must have equal length")
    if not groups:
        return {
            "grpo_signal/group_count": 0.0,
            "grpo_signal/mixed_outcome_group_count": 0.0,
            "grpo_signal/mixed_outcome_group_rate": 0.0,
            "grpo_signal/all_failure_group_count": 0.0,
            "grpo_signal/all_failure_group_rate": 0.0,
            "grpo_signal/all_success_group_count": 0.0,
            "grpo_signal/all_success_group_rate": 0.0,
            "grpo_signal/shaping_only_group_count": 0.0,
            "grpo_signal/shaping_only_group_rate": 0.0,
            "grpo_signal/zero_policy_signal_group_count": 0.0,
            "grpo_signal/zero_policy_signal_group_rate": 0.0,
            "grpo_signal/task_success_signal_group_count": 0.0,
            "grpo_signal/task_success_signal_group_rate": 0.0,
        }

    totals = {
        "mixed_outcome": 0,
        "all_failure": 0,
        "all_success": 0,
        "shaping_only": 0,
        "zero_policy_signal": 0,
        "task_success_signal": 0,
    }
    by_task_type: dict[str, dict[str, int]] = {}

    for group, group_signals in zip(groups, signals):
        trajectory_count = len(group.trajectories)
        if trajectory_count != len(group_signals.shaped_policy):
            raise ValueError("policy signal count must match group trajectories")
        if trajectory_count != len(group_signals.task_success):
            raise ValueError("task-success signal count must match group trajectories")

        success_count = sum(trajectory.won for trajectory in group.trajectories)
        mixed_outcome = 0 < success_count < trajectory_count
        all_failure = success_count == 0
        all_success = success_count == trajectory_count
        policy_nonzero = any(value != 0.0 for value in group_signals.shaped_policy)
        task_success_nonzero = any(value != 0.0 for value in group_signals.task_success)
        shaping_only = policy_nonzero and not task_success_nonzero
        zero_policy_signal = not policy_nonzero

        flags = {
            "mixed_outcome": mixed_outcome,
            "all_failure": all_failure,
            "all_success": all_success,
            "shaping_only": shaping_only,
            "zero_policy_signal": zero_policy_signal,
            "task_success_signal": task_success_nonzero,
        }
        task_type = group.task.task_type
        task_counts = by_task_type.setdefault(
            task_type,
            {"groups": 0, **{name: 0 for name in flags}},
        )
        task_counts["groups"] += 1
        for name, enabled in flags.items():
            totals[name] += int(enabled)
            task_counts[name] += int(enabled)

    group_count = len(groups)
    metrics: dict[str, float] = {"grpo_signal/group_count": float(group_count)}
    for name, count in totals.items():
        metrics[f"grpo_signal/{name}_group_count"] = float(count)
        metrics[f"grpo_signal/{name}_group_rate"] = count / group_count

    for task_type, counts in sorted(by_task_type.items()):
        task_group_count = counts["groups"]
        prefix = f"grpo_signal/task_type/{task_type}"
        metrics[f"{prefix}/group_count"] = float(task_group_count)
        for name in (
            "mixed_outcome",
            "all_failure",
            "shaping_only",
            "zero_policy_signal",
            "task_success_signal",
        ):
            metrics[f"{prefix}/{name}_group_count"] = float(counts[name])
            metrics[f"{prefix}/{name}_group_rate"] = counts[name] / task_group_count
    return metrics
