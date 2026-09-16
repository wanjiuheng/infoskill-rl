from __future__ import annotations

import json
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class EvaluationCheckpointScore:
    step: int
    macro_success: float
    overall_success: float
    invalid_action_rate: float
    checkpoint: str | None = None
    eval_batch_size: int | None = None
    comparison_role: str | None = None
    execution_mode: str | None = None


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
) -> dict[str, float | int | str | None]:
    payload: dict[str, float | int | str | None] = {
        "step": score.step,
        "macro_success": score.macro_success,
        "overall_success": score.overall_success,
        "invalid_action_rate": score.invalid_action_rate,
        "checkpoint": score.checkpoint or f"checkpoints/step-{score.step:06d}",
    }
    if score.eval_batch_size is not None:
        payload["eval_batch_size"] = score.eval_batch_size
    if score.comparison_role is not None:
        payload["comparison_role"] = score.comparison_role
    if score.execution_mode is not None:
        payload["execution_mode"] = score.execution_mode
    return payload


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
    default_batch_size = payload.get("eval_batch_size")
    default_comparison_role = payload.get("comparison_role")
    default_execution_mode = payload.get("execution_mode")
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
            eval_batch_size=(
                int(record.get("eval_batch_size", default_batch_size))
                if record.get("eval_batch_size", default_batch_size) is not None
                else None
            ),
            comparison_role=(
                str(record.get("comparison_role", default_comparison_role))
                if record.get("comparison_role", default_comparison_role)
                is not None
                else None
            ),
            execution_mode=(
                str(record.get("execution_mode", default_execution_mode))
                if record.get("execution_mode", default_execution_mode) is not None
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
    eval_batch_size: int = 8,
    comparison_role: str = "registered_batch8",
    execution_mode: str | None = None,
) -> dict[str, object]:
    if not scores:
        raise ValueError("checkpoint selection requires at least one evaluation")
    ordered = sorted(
        (
            replace(
                score,
                eval_batch_size=(
                    score.eval_batch_size
                    if score.eval_batch_size is not None
                    else eval_batch_size
                ),
                comparison_role=score.comparison_role or comparison_role,
                execution_mode=score.execution_mode or execution_mode,
            )
            for score in scores
        ),
        key=lambda score: score.step,
    )
    if len({score.step for score in ordered}) != len(ordered):
        raise RuntimeError("checkpoint selection contains duplicate evaluation steps")
    best = select_best_valid(ordered)
    payload: dict[str, object] = {
        "schema_version": 2,
        "disclosure": "validation-selected performance on valid_seen",
        "task_manifest_sha256": task_manifest_sha256,
        "eval_batch_size": eval_batch_size,
        "comparison_role": comparison_role,
        "execution_mode": execution_mode,
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


def write_valid_seen_learning_curve(
    path: str | Path,
    *,
    scores: Sequence[EvaluationCheckpointScore],
    eval_batch_size: int,
    monitoring_only: bool,
) -> Path:
    """Atomically refresh a dependency-free SVG of checkpoint success rates."""

    if not scores:
        raise ValueError("valid_seen learning curve requires at least one evaluation")
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    ordered = sorted(scores, key=lambda score: score.step)
    if len({score.step for score in ordered}) != len(ordered):
        raise RuntimeError("valid_seen learning curve contains duplicate steps")

    width, height = max(1_000, 190 + (len(ordered) - 1) * 90), 640
    left, right, top, bottom = 90, width - 50, 110, 535
    plot_width = right - left
    plot_height = bottom - top

    def point(index: int, value: float) -> tuple[float, float]:
        x = (
            left + plot_width * index / (len(ordered) - 1)
            if len(ordered) > 1
            else left + plot_width / 2
        )
        y = bottom - plot_height * min(1.0, max(0.0, value))
        return x, y

    macro_points = [
        point(index, score.macro_success)
        for index, score in enumerate(ordered)
    ]
    overall_points = [
        point(index, score.overall_success)
        for index, score in enumerate(ordered)
    ]

    def coordinates(points: Sequence[tuple[float, float]]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    def protocol(
        score: EvaluationCheckpointScore,
    ) -> tuple[int | None, str | None]:
        return score.eval_batch_size, score.execution_mode

    def protocol_text(value: tuple[int | None, str | None]) -> str:
        batch_size, execution_mode = value
        batch = f"batch {batch_size}" if batch_size is not None else "batch unknown"
        execution = {
            "cuda_graph": "CUDA Graph",
            "eager": "eager",
            "cuda_graph_split_k_one": "CUDA Graph / LoRA SPLIT_K=1",
            "eager_split_k_one": "eager / LoRA SPLIT_K=1",
        }.get(execution_mode, execution_mode or "execution unknown")
        return f"{batch} / {execution}"

    status = "monitoring curve" if monitoring_only else "registered evaluation curve"
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        "text{font-family:Arial,sans-serif;fill:#172033}",
        ".grid{stroke:#d9dfeb;stroke-width:1}",
        ".axis{stroke:#59657a;stroke-width:1.5}",
        ".macro{stroke:#2563eb;fill:none;stroke-width:3}",
        ".overall{stroke:#ea580c;fill:none;stroke-width:3}",
        ".dot-macro{fill:#2563eb}.dot-overall{fill:#ea580c}",
        ".point-label-macro{fill:#1d4ed8;font-size:11px;font-weight:700}",
        ".point-label-overall{fill:#c2410c;font-size:11px}",
        ".protocol-bridge{stroke:#7c3aed;stroke-width:1.5;stroke-dasharray:6 5}",
        ".protocol-label{fill:#6d28d9;font-size:12px;font-weight:700}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="90" y="35" font-size="24" font-weight="700">'
        "valid_seen success curve</text>",
        f'<text x="90" y="60" font-size="14" fill="#59657a">'
        f"batch={eval_batch_size} · {status} · refreshed after every evaluation"
        "</text>",
    ]
    for tick in range(0, 101, 20):
        y = bottom - plot_height * tick / 100
        lines.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                f'x2="{right}" y2="{y:.2f}"/>',
                f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" '
                f'font-size="13">{tick}%</text>',
            ]
        )
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{bottom}" '
            f'x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" '
            f'x2="{left}" y2="{bottom}"/>',
        ]
    )
    for index in range(1, len(ordered)):
        before = protocol(ordered[index - 1])
        after = protocol(ordered[index])
        if before == after or before == (None, None) or after == (None, None):
            continue
        bridge_x = macro_points[index - 1][0]
        label = (
            f"after update {ordered[index - 1].step}: "
            f"{protocol_text(before)} → {protocol_text(after)}"
        )
        lines.extend(
            [
                f'<line class="protocol-bridge" x1="{bridge_x:.2f}" y1="{top}" '
                f'x2="{bridge_x:.2f}" y2="{bottom}"/>',
                f'<text class="protocol-label" x="{bridge_x + 7:.2f}" '
                f'y="{top - 13}">{escape(label)}</text>',
            ]
        )
    lines.extend(
        [
            f'<polyline class="macro" points="{coordinates(macro_points)}"/>',
            f'<polyline class="overall" points="{coordinates(overall_points)}"/>',
        ]
    )
    for score, macro_point, overall_point in zip(
        ordered, macro_points, overall_points, strict=True
    ):
        x, macro_y = macro_point
        _, overall_y = overall_point
        labels_are_close = abs(macro_y - overall_y) < 34
        if labels_are_close:
            macro_label_y = min(macro_y, overall_y) - 12
            overall_label_y = max(macro_y, overall_y) + 20
            label_top = top + 14
            label_bottom = bottom - 8
            if macro_label_y < label_top:
                shift = label_top - macro_label_y
                macro_label_y += shift
                overall_label_y += shift
            if overall_label_y > label_bottom:
                shift = overall_label_y - label_bottom
                macro_label_y -= shift
                overall_label_y -= shift
        else:
            macro_label_y = macro_y - 12
            overall_label_y = overall_y + 20
            macro_label_y = min(bottom - 8, max(top + 14, macro_label_y))
            overall_label_y = min(bottom - 8, max(top + 14, overall_label_y))
        lines.extend(
            [
                f'<circle class="dot-macro" cx="{x:.2f}" cy="{macro_y:.2f}" '
                f'r="4"><title>update {score.step}: Macro success '
                f"{score.macro_success:.2%}</title></circle>",
                f'<circle class="dot-overall" cx="{x:.2f}" cy="{overall_y:.2f}" '
                f'r="4"><title>update {score.step}: Overall success '
                f"{score.overall_success:.2%}</title></circle>",
                f'<text class="point-label-macro" x="{x:.2f}" '
                f'y="{macro_label_y:.2f}" text-anchor="middle">'
                f"M {score.macro_success:.2%}</text>",
                f'<text class="point-label-overall" x="{x:.2f}" '
                f'y="{overall_label_y:.2f}" text-anchor="middle">'
                f"O {score.overall_success:.2%}</text>",
                f'<text x="{x:.2f}" y="{bottom + 24}" text-anchor="middle" '
                f'font-size="12">{score.step}</text>',
            ]
        )
    latest = ordered[-1]
    lines.extend(
        [
            f'<line class="macro" x1="{width - 350}" y1="35" '
            f'x2="{width - 310}" y2="35"/>',
            f'<text x="{width - 300}" y="40" font-size="14">'
            "Macro success</text>",
            f'<line class="overall" x1="{width - 350}" y1="58" '
            f'x2="{width - 310}" y2="58"/>',
            f'<text x="{width - 300}" y="63" font-size="14">'
            "Overall success</text>",
            f'<text x="90" y="595" font-size="15" font-weight="700">'
            f"Latest update {latest.step}: Macro {latest.macro_success:.2%} · "
            f"Overall {latest.overall_success:.2%}</text>",
            f'<text x="{width / 2:.2f}" y="628" text-anchor="middle" '
            'font-size="13">Optimizer update</text>',
            "</svg>",
        ]
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def inherit_forked_checkpoint_selection(
    *,
    source_run: str | Path,
    destination_run: str | Path,
    max_source_step: int,
    task_manifest_sha256: str,
    eval_batch_size: int = 8,
    comparison_role: str = "registered_batch8",
    execution_mode: str | None = None,
) -> dict[str, object]:
    """Merge eligible source evaluations into a fork without losing provenance."""
    source_directory = Path(source_run).expanduser().resolve()
    destination_directory = Path(destination_run).expanduser().resolve()
    source_path = source_directory / "checkpoint_selection.json"
    if not source_path.is_file():
        raise RuntimeError(
            f"forked valid_seen resume requires source checkpoint selection: {source_path}"
        )
    source_execution_mode = _resolved_execution_mode(source_directory)
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
                eval_batch_size=score.eval_batch_size,
                comparison_role=score.comparison_role,
                execution_mode=score.execution_mode or source_execution_mode,
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
        eval_batch_size=eval_batch_size,
        comparison_role=comparison_role,
        execution_mode=execution_mode,
    )


def _resolved_execution_mode(run_directory: Path) -> str | None:
    path = run_directory / "resolved_config.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    runtime_options = payload.get("runtime_options")
    if not isinstance(runtime_options, dict):
        return None
    enabled = runtime_options.get("hybrid_prefix_cuda_graph")
    if enabled is None:
        return None
    return "cuda_graph" if bool(enabled) else "eager"
