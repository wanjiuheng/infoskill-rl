from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .expert_replay import ExpertReplayResult


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GroundingManifest:
    schema_version: int
    source_split: str
    total_games: int
    successful_games: int
    quarantined_games: int
    success_coverage: float
    over_persist_horizon: int
    over_persist_horizon_rate: float
    task_type_counts: Mapping[str, Mapping[str, int]]
    quarantine_reasons: Mapping[str, int]
    trajectory_lengths: Mapping[str, float | int]
    source_checksums: Mapping[str, str]
    code_revision: str
    expert_name: str
    max_replay_steps: int
    persist_horizon: int
    formal_gate_passed: bool
    formal_gate_failures: tuple[str, ...]


def build_grounding_manifest(
    *,
    results: Sequence[tuple[str, ExpertReplayResult]],
    source_checksums: Mapping[str, str],
    code_revision: str,
    max_replay_steps: int,
    persist_horizon: int,
    minimum_success_coverage: float = 0.99,
    maximum_over_horizon_rate: float = 0.01,
) -> GroundingManifest:
    if not results:
        raise ValueError("cannot build a grounding manifest without replay results")
    success = [result for _, result in results if result.succeeded]
    lengths = [result.total_steps for _, result in results]
    over_horizon = sum(result.total_steps > persist_horizon for _, result in results)
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: Counter[str] = Counter()
    for task_type, result in results:
        type_counts[task_type]["total"] += 1
        if result.succeeded:
            type_counts[task_type]["successful"] += 1
        else:
            type_counts[task_type]["quarantined"] += 1
            reasons[result.quarantine_reason or "unknown"] += 1

    coverage = len(success) / len(results)
    over_rate = over_horizon / len(results)
    failures: list[str] = []
    if coverage < minimum_success_coverage:
        failures.append("success_coverage_below_threshold")
    if over_rate > maximum_over_horizon_rate:
        failures.append("over_horizon_rate_above_threshold")
    ordered = sorted(lengths)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    return GroundingManifest(
        schema_version=1,
        source_split="train",
        total_games=len(results),
        successful_games=len(success),
        quarantined_games=len(results) - len(success),
        success_coverage=coverage,
        over_persist_horizon=over_horizon,
        over_persist_horizon_rate=over_rate,
        task_type_counts={key: dict(value) for key, value in sorted(type_counts.items())},
        quarantine_reasons=dict(sorted(reasons.items())),
        trajectory_lengths={
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "median": median,
        },
        source_checksums=dict(sorted(source_checksums.items())),
        code_revision=code_revision,
        expert_name="ALFWorld HandCodedTWAgent (direct, strict admissibility)",
        max_replay_steps=max_replay_steps,
        persist_horizon=persist_horizon,
        formal_gate_passed=not failures,
        formal_gate_failures=tuple(failures),
    )


def write_grounding_artifacts(
    *,
    output_directory: str | Path,
    results: Iterable[tuple[str, ExpertReplayResult]],
    manifest: GroundingManifest,
) -> None:
    """Atomically persist successful samples, quarantines, and the audit manifest."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    samples_path = destination / "grounding_samples.jsonl"
    quarantine_path = destination / "quarantine.jsonl"
    manifest_path = destination / "manifest.json"
    sample_lines: list[str] = []
    quarantine_lines: list[str] = []
    for task_type, result in results:
        if result.succeeded:
            for sample in result.samples:
                sample_lines.append(
                    json.dumps(
                        {
                            "task_id": result.task_id,
                            "task_type": task_type,
                            "state": asdict(sample.state),
                            "expert_action": sample.expert_action,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
        else:
            quarantine_lines.append(
                json.dumps(
                    {
                        "task_id": result.task_id,
                        "task_type": task_type,
                        "total_steps": result.total_steps,
                        "reason": result.quarantine_reason,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    _atomic_write(samples_path, "\n".join(sample_lines) + ("\n" if sample_lines else ""))
    _atomic_write(quarantine_path, "\n".join(quarantine_lines) + ("\n" if quarantine_lines else ""))
    _atomic_write(
        manifest_path,
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
