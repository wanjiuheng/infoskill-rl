from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Sequence

from infoskill.episode import TrajectoryGroup


_DIFFICULTY_BANDS = ("hard_failed", "partial", "mastered", "incomplete")


class TrainingTaskOutcomeWriter:
    """Write a compact, replay-safe index over the full training traces."""

    def __init__(
        self,
        run_directory: str | Path,
        *,
        expected_rollouts_per_task: int,
        committed_through_update: int | None = None,
    ) -> None:
        if expected_rollouts_per_task <= 0:
            raise ValueError("expected_rollouts_per_task must be positive")
        self.run_directory = Path(run_directory)
        self.expected_rollouts_per_task = expected_rollouts_per_task
        self.outcome_directory = self.run_directory / "task-outcomes"
        self.outcome_directory.mkdir(parents=True, exist_ok=True)
        if committed_through_update is not None:
            if committed_through_update < 0:
                raise ValueError("committed_through_update must be nonnegative")
            self._quarantine_after(committed_through_update)
            self._refresh_summary()

    def write_training_update(
        self,
        *,
        global_update: int,
        groups: Sequence[TrajectoryGroup],
        trace_path: str | Path,
    ) -> dict[str, int]:
        if global_update < 0:
            raise ValueError("global_update must be nonnegative")
        trace = Path(trace_path)
        try:
            trace_reference = trace.resolve().relative_to(
                self.run_directory.resolve()
            ).as_posix()
        except ValueError:
            trace_reference = str(trace.resolve())

        records = [
            self._group_record(
                global_update=global_update,
                group=group,
                trace_reference=trace_reference,
            )
            for group in groups
        ]
        path = self.outcome_directory / f"train-update-{global_update:06d}.jsonl"
        _atomic_write_jsonl(path, records)
        self._refresh_summary()
        return {
            f"{band}_groups": sum(
                record["difficulty_band"] == band for record in records
            )
            for band in _DIFFICULTY_BANDS
        } | {"task_groups": len(records)}

    def _group_record(
        self,
        *,
        global_update: int,
        group: TrajectoryGroup,
        trace_reference: str,
    ) -> dict[str, object]:
        trajectories = group.trajectories
        success_count = sum(trajectory.won for trajectory in trajectories)
        rollout_count = len(trajectories)
        if rollout_count != self.expected_rollouts_per_task:
            difficulty_band = "incomplete"
        elif success_count == 0:
            difficulty_band = "hard_failed"
        elif success_count == rollout_count:
            difficulty_band = "mastered"
        else:
            difficulty_band = "partial"
        return {
            "schema_version": 1,
            "global_update": global_update,
            "task_id": group.task.task_id,
            "task_type": group.task.task_type,
            "split": group.task.split,
            "goal": group.task.goal,
            "environment_path": group.task.environment_path,
            "trajectory_path": group.task.trajectory_path,
            "expected_rollout_count": self.expected_rollouts_per_task,
            "rollout_count": rollout_count,
            "success_count": success_count,
            "failure_count": rollout_count - success_count,
            "failed_rollout_ids": [
                trajectory.rollout_id
                for trajectory in trajectories
                if not trajectory.won
            ],
            "difficulty_band": difficulty_band,
            "trace": trace_reference,
            "rollouts": [
                {
                    "rollout_id": trajectory.rollout_id,
                    "won": trajectory.won,
                    "reward": trajectory.reward,
                    "step_count": len(trajectory.steps),
                    "invalid_action_count": trajectory.invalid_action_count,
                    "environment_done": trajectory.environment_done,
                    "horizon_exhausted": trajectory.horizon_exhausted,
                }
                for trajectory in trajectories
            ],
        }

    def _refresh_summary(self) -> dict[str, object]:
        counts = {band: 0 for band in _DIFFICULTY_BANDS}
        per_task_type: dict[str, dict[str, int]] = {}
        updates: list[int] = []
        total_task_groups = 0
        for path in sorted(self.outcome_directory.glob("train-update-*.jsonl")):
            update_seen: int | None = None
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                band = str(record["difficulty_band"])
                task_type = str(record["task_type"])
                if band not in counts:
                    raise RuntimeError(f"invalid task outcome band in {path}: {band}")
                counts[band] += 1
                type_counts = per_task_type.setdefault(
                    task_type,
                    {item: 0 for item in _DIFFICULTY_BANDS},
                )
                type_counts[band] += 1
                total_task_groups += 1
                update_seen = int(record["global_update"])
            if update_seen is not None:
                updates.append(update_seen)
        summary: dict[str, object] = {
            "schema_version": 1,
            "expected_rollouts_per_task": self.expected_rollouts_per_task,
            "updates": sorted(updates),
            "total_task_groups": total_task_groups,
            "difficulty_counts": counts,
            "per_task_type": dict(sorted(per_task_type.items())),
        }
        _atomic_write_json(
            self.run_directory / "task-outcomes-summary.json",
            summary,
        )
        return summary

    def _quarantine_after(self, committed_through_update: int) -> None:
        stale_paths = []
        active_paths = []
        for path in sorted(self.outcome_directory.glob("train-update-*.jsonl")):
            match = re.fullmatch(r"train-update-(\d{6})\.jsonl", path.name)
            if match is None:
                continue
            if int(match.group(1)) > committed_through_update:
                stale_paths.append(path)
            else:
                active_paths.append(path)
        if not stale_paths:
            return
        stale_directory = (
            self.outcome_directory
            / f"stale-after-resume-{committed_through_update:06d}"
        )
        stale_directory.mkdir(parents=True, exist_ok=True)
        active_traces = {
            trace
            for path in active_paths
            for record in _read_jsonl(path)
            if (
                trace := self._internal_trace_path(record.get("trace"))
            ) is not None
        }
        archived_traces: dict[Path, tuple[Path, str]] = {}
        for path in stale_paths:
            records = _read_jsonl(path)
            source_traces: set[Path] = set()
            for record in records:
                source_trace = self._internal_trace_path(record.get("trace"))
                if source_trace is None or not source_trace.is_file():
                    continue
                source_traces.add(source_trace)
                archived = archived_traces.get(source_trace)
                if archived is None:
                    trace_sha256 = _sha256_file(source_trace)
                    archived_trace = (
                        stale_directory
                        / "traces"
                        / f"{trace_sha256[:16]}-{source_trace.name}"
                    )
                    _durable_copy(
                        source_trace,
                        archived_trace,
                        expected_sha256=trace_sha256,
                    )
                    archived = (archived_trace, trace_sha256)
                    archived_traces[source_trace] = archived
                archived_trace, trace_sha256 = archived
                record["trace"] = archived_trace.relative_to(
                    self.run_directory
                ).as_posix()
                record["trace_sha256"] = trace_sha256
            destination = _matching_jsonl(stale_directory, path.name, records)
            if destination is None:
                destination = _unique_path(stale_directory / path.name)
                _atomic_write_jsonl(destination, records)
            path.unlink()
            for source_trace in source_traces - active_traces:
                if source_trace.exists():
                    source_trace.unlink()
                    _fsync_directory(source_trace.parent)
        _fsync_directory(stale_directory)
        _fsync_directory(self.outcome_directory)

    def _internal_trace_path(self, value: object) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.run_directory / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(self.run_directory.resolve())
        except ValueError:
            return None
        return candidate


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _unique_path(path: Path) -> Path:
    destination = path
    suffix = 1
    while destination.exists():
        destination = path.with_name(f"{path.name}.{suffix:03d}")
        suffix += 1
    return destination


def _matching_jsonl(
    directory: Path,
    name: str,
    records: Sequence[dict[str, object]],
) -> Path | None:
    for candidate in sorted(directory.glob(f"{name}*")):
        if candidate.is_file() and _read_jsonl(candidate) == list(records):
            return candidate
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _durable_copy(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if _sha256_file(destination) != expected_sha256:
            raise RuntimeError(f"archived trace checksum mismatch: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.tmp")
    with source.open("rb") as source_stream, temporary.open("wb") as stream:
        shutil.copyfileobj(source_stream, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    _fsync_directory(destination.parent)


def _atomic_write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
