from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from infoskill.episode import TaskSpec


@dataclass(frozen=True, slots=True)
class TrainMonitorManifest:
    schema_version: int
    master_seed: int
    fraction: float
    source_task_count: int
    monitor_task_ids: tuple[str, ...]
    per_task_type_counts: dict[str, int]
    source_manifest_sha256: str


def build_train_monitor_manifest(
    tasks: Sequence[TaskSpec], *, master_seed: int, fraction: float = 0.10
) -> TrainMonitorManifest:
    if not 0.0 < fraction < 1.0:
        raise ValueError("monitor fraction must be between zero and one")
    grouped: dict[str, list[TaskSpec]] = defaultdict(list)
    for task in tasks:
        if task.split != "train":
            raise ValueError("train monitor can only be derived from train tasks")
        grouped[task.task_type].append(task)
    selected: list[TaskSpec] = []
    for task_type in sorted(grouped):
        ordered = sorted(
            grouped[task_type],
            key=lambda task: _key(master_seed, task.task_id),
        )
        count = max(1, round(len(ordered) * fraction))
        selected.extend(ordered[:count])
    selected.sort(key=lambda task: task.task_id)
    per_type = defaultdict(int)
    for task in selected:
        per_type[task.task_type] += 1
    return TrainMonitorManifest(
        schema_version=1,
        master_seed=master_seed,
        fraction=fraction,
        source_task_count=len(tasks),
        monitor_task_ids=tuple(task.task_id for task in selected),
        per_task_type_counts=dict(sorted(per_type.items())),
        source_manifest_sha256=_source_checksum(tasks),
    )


def write_train_monitor_manifest(path: str | Path, manifest: TrainMonitorManifest) -> None:
    Path(path).write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _key(master_seed: int, task_id: str) -> bytes:
    return hashlib.sha256(f"train_monitor|{master_seed}|{task_id}".encode()).digest()


def _source_checksum(tasks: Sequence[TaskSpec]) -> str:
    digest = hashlib.sha256()
    for task in sorted(tasks, key=lambda item: item.task_id):
        digest.update(f"{task.task_id}\0{task.task_type}\0{task.goal}\n".encode("utf-8"))
    return digest.hexdigest()
