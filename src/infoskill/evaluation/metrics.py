from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from infoskill.config import EvaluationConfig


@dataclass(frozen=True, slots=True)
class EpisodeEvaluation:
    task_id: str
    task_type: str
    won: bool
    steps: int
    invalid_action_count: int
    infrastructure_error: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.task_type:
            raise ValueError("evaluation task identity must not be empty")
        if self.steps < 0 or self.invalid_action_count < 0:
            raise ValueError("evaluation counts must be non-negative")


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    is_complete: bool
    evaluated: int
    overall_success: float | None
    macro_success: float | None
    invalid_action_rate: float | None
    mean_steps: float | None
    per_task_type_success: dict[str, float]
    incomplete_reasons: tuple[str, ...] = ()


def aggregate_valid_seen(
    records: Sequence[EpisodeEvaluation],
    *,
    config: EvaluationConfig,
) -> EvaluationSummary:
    """Aggregate metrics only when the registered 140-task manifest is complete."""

    reasons: list[str] = []
    task_ids = [record.task_id for record in records]
    if len(set(task_ids)) != len(task_ids):
        reasons.append("duplicate_task_id")

    failed = [record.task_id for record in records if record.infrastructure_error is not None]
    if failed:
        reasons.append(f"infrastructure_failures:{len(failed)}")

    expected = {item.task_type: item.count for item in config.denominators}
    actual = Counter(record.task_type for record in records)
    if dict(actual) != expected:
        reasons.append(f"denominator_mismatch:expected={expected},actual={dict(actual)}")

    if reasons:
        return EvaluationSummary(
            is_complete=False,
            evaluated=len(records),
            overall_success=None,
            macro_success=None,
            invalid_action_rate=None,
            mean_steps=None,
            per_task_type_success={},
            incomplete_reasons=tuple(reasons),
        )

    successes = Counter(record.task_type for record in records if record.won)
    per_type = {task_type: successes[task_type] / count for task_type, count in expected.items()}
    overall = sum(successes.values()) / config.total_tasks
    macro = sum(per_type.values()) / len(per_type)
    total_steps = sum(record.steps for record in records)
    invalid_action_rate = (
        sum(record.invalid_action_count for record in records) / total_steps
        if total_steps
        else 0.0
    )
    return EvaluationSummary(
        is_complete=True,
        evaluated=len(records),
        overall_success=overall,
        macro_success=macro,
        invalid_action_rate=invalid_action_rate,
        mean_steps=total_steps / len(records),
        per_task_type_success=per_type,
    )
