from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True, slots=True)
class TrainingRolloutStepScore:
    step: int
    mean_steps: float
    mean_successful_steps: float | None
    mean_failed_steps: float | None
    horizon_exhaustion_rate: float | None


def load_training_rollout_step_scores(
    metric_paths: Sequence[str | Path],
    *,
    max_step: int | None = None,
) -> tuple[TrainingRolloutStepScore, ...]:
    """Load the last committed train metric for every optimizer update."""

    if max_step is not None and max_step < 0:
        raise ValueError("max_step must be nonnegative")
    by_step: dict[int, TrainingRolloutStepScore] = {}
    for value in metric_paths:
        path = Path(value)
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("phase") != "train":
                continue
            if "step" not in record or "rollout/mean_steps" not in record:
                continue
            step = int(record["step"])
            if step < 0:
                raise RuntimeError(f"negative train step in {path}:{line_number}")
            if max_step is not None and step > max_step:
                continue
            mean_steps = _finite_nonnegative(
                record["rollout/mean_steps"],
                location=f"{path}:{line_number}:rollout/mean_steps",
            )
            successful_count = _optional_count(
                record.get("rollout/successful_trajectories"),
                location=(
                    f"{path}:{line_number}:rollout/successful_trajectories"
                ),
            )
            failed_count = _optional_count(
                record.get("rollout/failed_trajectories"),
                location=f"{path}:{line_number}:rollout/failed_trajectories",
            )
            successful_mean = _partition_mean(
                record.get("rollout/mean_steps_successful"),
                successful_count,
                location=(
                    f"{path}:{line_number}:rollout/mean_steps_successful"
                ),
            )
            failed_mean = _partition_mean(
                record.get("rollout/mean_steps_failed"),
                failed_count,
                location=f"{path}:{line_number}:rollout/mean_steps_failed",
            )
            horizon_rate = _optional_rate(
                record.get("rollout/horizon_exhaustion_rate"),
                location=(
                    f"{path}:{line_number}:rollout/horizon_exhaustion_rate"
                ),
            )
            by_step[step] = TrainingRolloutStepScore(
                step=step,
                mean_steps=mean_steps,
                mean_successful_steps=successful_mean,
                mean_failed_steps=failed_mean,
                horizon_exhaustion_rate=horizon_rate,
            )
    return tuple(by_step[step] for step in sorted(by_step))


def write_training_rollout_steps_curve(
    path: str | Path,
    *,
    scores: Sequence[TrainingRolloutStepScore],
) -> Path:
    """Atomically refresh a dependency-free SVG of training rollout lengths."""

    if not scores:
        raise ValueError("training rollout step curve requires at least one update")
    ordered = sorted(scores, key=lambda score: score.step)
    if len({score.step for score in ordered}) != len(ordered):
        raise RuntimeError("training rollout step curve contains duplicate steps")

    width, height = max(1_200, 260 + (len(ordered) - 1) * 4), 750
    left, right = 90, width - 60
    step_top, step_bottom = 115, 475
    horizon_top, horizon_bottom = 555, 650
    plot_width = right - left
    step_plot_height = step_bottom - step_top
    horizon_plot_height = horizon_bottom - horizon_top
    observed_steps = [score.mean_steps for score in ordered]
    observed_steps.extend(
        value
        for score in ordered
        for value in (score.mean_successful_steps, score.mean_failed_steps)
        if value is not None
    )
    step_ceiling = max(5.0, math.ceil(max(observed_steps) / 5.0) * 5.0)

    def x_at(index: int) -> float:
        if len(ordered) == 1:
            return left + plot_width / 2
        return left + plot_width * index / (len(ordered) - 1)

    def step_y(value: float) -> float:
        return step_bottom - step_plot_height * value / step_ceiling

    def horizon_y(value: float) -> float:
        return horizon_bottom - horizon_plot_height * value

    def series_segments(
        getter: Callable[[TrainingRolloutStepScore], float | None],
        y_mapper: Callable[[float], float],
    ) -> list[str]:
        segments: list[str] = []
        current: list[str] = []
        for index, score in enumerate(ordered):
            value = getter(score)
            if value is None:
                if current:
                    segments.append(" ".join(current))
                    current = []
                continue
            current.append(f"{x_at(index):.2f},{y_mapper(value):.2f}")
        if current:
            segments.append(" ".join(current))
        return segments

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        "text{font-family:Arial,sans-serif;fill:#172033}",
        ".grid{stroke:#d9dfeb;stroke-width:1}",
        ".axis{stroke:#59657a;stroke-width:1.5}",
        ".all{stroke:#2563eb;fill:none;stroke-width:2.5}",
        ".success{stroke:#16a34a;fill:none;stroke-width:2.5}",
        ".failure{stroke:#dc2626;fill:none;stroke-width:2.5}",
        ".horizon{stroke:#7c3aed;fill:none;stroke-width:2.5}",
        ".dot-all{fill:#2563eb}",
        ".dot-success{fill:#16a34a}",
        ".dot-failure{fill:#dc2626}",
        ".dot-horizon{fill:#7c3aed}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="90" y="34" font-size="24" font-weight="700">'
        "training rollout step curve</text>",
        '<text x="90" y="61" font-size="14" fill="#59657a">'
        "Per-update training diagnostics · task mix changes across updates"
        "</text>",
    ]

    legend = (
        ("all", "All rollouts", 0),
        ("success", "Successful rollouts", 180),
        ("failure", "Failed rollouts", 410),
        ("horizon", "Horizon exhausted", 590),
    )
    for css_class, label, offset in legend:
        x = width - 780 + offset
        lines.extend(
            [
                f'<line class="{css_class}" x1="{x}" y1="35" '
                f'x2="{x + 32}" y2="35"/>',
                f'<text x="{x + 40}" y="40" font-size="13">{label}</text>',
            ]
        )

    for tick_index in range(6):
        value = step_ceiling * tick_index / 5
        y = step_y(value)
        lines.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                f'x2="{right}" y2="{y:.2f}"/>',
                f'<text x="{left - 12}" y="{y + 5:.2f}" '
                f'text-anchor="end" font-size="12">{value:.0f}</text>',
            ]
        )
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{step_top}" '
            f'x2="{left}" y2="{step_bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{step_bottom}" '
            f'x2="{right}" y2="{step_bottom}"/>',
            '<text x="22" y="300" transform="rotate(-90 22 300)" '
            'text-anchor="middle" font-size="13">Mean environment steps</text>',
        ]
    )

    series = (
        ("all", lambda score: score.mean_steps, step_y),
        ("success", lambda score: score.mean_successful_steps, step_y),
        ("failure", lambda score: score.mean_failed_steps, step_y),
        ("horizon", lambda score: score.horizon_exhaustion_rate, horizon_y),
    )
    for css_class, getter, y_mapper in series:
        for points in series_segments(getter, y_mapper):
            lines.append(f'<polyline class="{css_class}" points="{points}"/>')
        for index, score in enumerate(ordered):
            value = getter(score)
            if value is None:
                continue
            lines.append(
                f'<circle class="dot-{css_class}" cx="{x_at(index):.2f}" '
                f'cy="{y_mapper(value):.2f}" r="2.25">'
                f'<title>update {score.step}: {value:.4f}</title></circle>'
            )

    for rate in (0.0, 0.5, 1.0):
        y = horizon_y(rate)
        lines.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                f'x2="{right}" y2="{y:.2f}"/>',
                f'<text x="{left - 12}" y="{y + 5:.2f}" '
                f'text-anchor="end" font-size="12">{rate:.0%}</text>',
            ]
        )
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{horizon_top}" '
            f'x2="{left}" y2="{horizon_bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{horizon_bottom}" '
            f'x2="{right}" y2="{horizon_bottom}"/>',
            '<text x="22" y="605" transform="rotate(-90 22 605)" '
            'text-anchor="middle" font-size="13">Horizon rate</text>',
        ]
    )

    labeled_steps = {
        score.step for score in ordered if score.step % 25 == 0
    } | {ordered[0].step, ordered[-1].step}
    for index, score in enumerate(ordered):
        if score.step not in labeled_steps:
            continue
        x = x_at(index)
        lines.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{step_top}" '
                f'x2="{x:.2f}" y2="{horizon_bottom}"/>',
                f'<text x="{x:.2f}" y="{horizon_bottom + 22}" '
                f'text-anchor="middle" font-size="11">{score.step}</text>',
            ]
        )

    latest = ordered[-1]
    latest_parts = [
        f"Latest update {latest.step}",
        f"all mean {latest.mean_steps:.2f}",
    ]
    if latest.mean_successful_steps is not None:
        latest_parts.append(
            f"Latest successful mean {latest.mean_successful_steps:.2f}"
        )
    if latest.mean_failed_steps is not None:
        latest_parts.append(f"Latest failed mean {latest.mean_failed_steps:.2f}")
    if latest.horizon_exhaustion_rate is not None:
        latest_parts.append(f"horizon {latest.horizon_exhaustion_rate:.2%}")
    lines.extend(
        [
            '<text x="90" y="706" font-size="13" font-weight="700">'
            f"{' · '.join(latest_parts)}</text>",
            f'<text x="{width / 2:.2f}" y="738" text-anchor="middle" '
            'font-size="13">Optimizer update</text>',
            "</svg>",
        ]
    )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    _fsync_directory(destination.parent)
    return destination


def _finite_nonnegative(value: object, *, location: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise RuntimeError(f"invalid nonnegative metric at {location}: {value}")
    return result


def _optional_count(value: object | None, *, location: str) -> float | None:
    if value is None:
        return None
    return _finite_nonnegative(value, location=location)


def _partition_mean(
    value: object | None,
    count: float | None,
    *,
    location: str,
) -> float | None:
    if value is None or count is None or count == 0:
        return None
    return _finite_nonnegative(value, location=location)


def _optional_rate(value: object | None, *, location: str) -> float | None:
    if value is None:
        return None
    result = _finite_nonnegative(value, location=location)
    if result > 1:
        raise RuntimeError(f"invalid rate metric at {location}: {value}")
    return result


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
