from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class EvaluationCheckpointScore:
    step: int
    macro_success: float
    overall_success: float
    invalid_action_rate: float
    checkpoint: str | None = None


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


def checkpoint_score_payload(
    score: EvaluationCheckpointScore,
) -> dict[str, float | int | str]:
    return {
        "step": score.step,
        "macro_success": score.macro_success,
        "overall_success": score.overall_success,
        "invalid_action_rate": score.invalid_action_rate,
        "checkpoint": score.checkpoint or f"checkpoints/step-{score.step:06d}",
    }


def load_checkpoint_scores(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> list[EvaluationCheckpointScore]:
    selection_path = Path(path)
    if not selection_path.is_file():
        return []
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = payload.get("task_manifest_sha256")
    if (
        expected_manifest_sha256 is not None
        and manifest != expected_manifest_sha256
    ):
        raise RuntimeError(
            f"checkpoint selection manifest differs from the registered manifest: {selection_path}"
        )
    records = payload.get("evaluations")
    if not isinstance(records, list):
        raise RuntimeError(f"invalid checkpoint selection history: {selection_path}")
    scores: list[EvaluationCheckpointScore] = []
    seen_steps: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError(f"invalid checkpoint selection history: {selection_path}")
        score = EvaluationCheckpointScore(
            step=int(record["step"]),
            macro_success=float(record["macro_success"]),
            overall_success=float(record["overall_success"]),
            invalid_action_rate=float(record["invalid_action_rate"]),
            checkpoint=(
                str(record["checkpoint"])
                if record.get("checkpoint") is not None
                else None
            ),
        )
        if score.step in seen_steps:
            raise RuntimeError(f"duplicate checkpoint selection step: {selection_path}")
        seen_steps.add(score.step)
        scores.append(score)
    return sorted(scores, key=lambda score: score.step)


def write_checkpoint_selection(
    path: str | Path,
    *,
    scores: Sequence[EvaluationCheckpointScore],
    task_manifest_sha256: str,
) -> dict[str, object]:
    if not scores:
        raise ValueError("checkpoint selection requires at least one evaluation")
    ordered = sorted(scores, key=lambda score: score.step)
    if len({score.step for score in ordered}) != len(ordered):
        raise RuntimeError("checkpoint selection contains duplicate evaluation steps")
    best = select_best_valid(ordered)
    payload: dict[str, object] = {
        "schema_version": 1,
        "disclosure": "validation-selected performance on valid_seen",
        "task_manifest_sha256": task_manifest_sha256,
        "rule": [
            "max_macro_success",
            "max_overall_success",
            "min_invalid_action_rate",
            "earliest_update",
        ],
        "evaluations": [checkpoint_score_payload(score) for score in ordered],
        "last": checkpoint_score_payload(ordered[-1]),
        "best_valid": checkpoint_score_payload(best),
    }
    selection_path = Path(path)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = selection_path.with_name(f".{selection_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(selection_path)
    return payload


def inherit_forked_checkpoint_selection(
    *,
    source_run: str | Path,
    destination_run: str | Path,
    max_source_step: int,
    task_manifest_sha256: str,
) -> dict[str, object]:
    """Merge eligible source evaluations into a fork without losing provenance."""
    source_directory = Path(source_run).expanduser().resolve()
    destination_directory = Path(destination_run).expanduser().resolve()
    source_path = source_directory / "checkpoint_selection.json"
    if not source_path.is_file():
        raise RuntimeError(
            f"forked valid_seen resume requires source checkpoint selection: {source_path}"
        )
    inherited: list[EvaluationCheckpointScore] = []
    for score in load_checkpoint_scores(
        source_path,
        expected_manifest_sha256=task_manifest_sha256,
    ):
        if score.step > max_source_step:
            continue
        checkpoint = Path(
            score.checkpoint or f"checkpoints/step-{score.step:06d}"
        )
        if not checkpoint.is_absolute():
            checkpoint = (source_directory / checkpoint).resolve()
        inherited.append(
            EvaluationCheckpointScore(
                step=score.step,
                macro_success=score.macro_success,
                overall_success=score.overall_success,
                invalid_action_rate=score.invalid_action_rate,
                checkpoint=str(checkpoint),
            )
        )

    destination_path = destination_directory / "checkpoint_selection.json"
    existing = load_checkpoint_scores(
        destination_path,
        expected_manifest_sha256=task_manifest_sha256,
    )
    by_step = {score.step: score for score in inherited}
    for score in existing:
        previous = by_step.get(score.step)
        if previous is not None and (
            previous.macro_success,
            previous.overall_success,
            previous.invalid_action_rate,
        ) != (
            score.macro_success,
            score.overall_success,
            score.invalid_action_rate,
        ):
            raise RuntimeError(
                f"forked checkpoint selection has conflicting evaluation step {score.step}"
            )
        by_step[score.step] = score
    if not by_step:
        raise RuntimeError("forked checkpoint selection has no eligible evaluations")
    return write_checkpoint_selection(
        destination_path,
        scores=tuple(by_step.values()),
        task_manifest_sha256=task_manifest_sha256,
    )
