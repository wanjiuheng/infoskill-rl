from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class EvaluationCheckpointScore:
    step: int
    macro_success: float
    overall_success: float
    invalid_action_rate: float


def select_best_valid(scores: Sequence[EvaluationCheckpointScore]) -> EvaluationCheckpointScore:
    if not scores:
        raise ValueError("best-valid selection requires at least one complete evaluation")
    return max(
        scores,
        key=lambda score: (
            score.macro_success,
            score.overall_success,
            -score.invalid_action_rate,
            -score.step,
        ),
    )
