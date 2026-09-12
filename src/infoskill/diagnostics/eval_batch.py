from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from infoskill.config import TaskDenominator
from infoskill.episode import TaskSpec


def load_pressure_task_manifest(
    path: str | Path,
    *,
    available_tasks: Sequence[TaskSpec],
    full_task_manifest_sha256: str,
) -> tuple[tuple[TaskSpec, ...], dict[str, object]]:
    """Resolve a fixed non-reportable pressure subset against valid_seen."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("evaluation pressure manifest must use schema_version=1")
    if payload.get("split") != "valid_seen":
        raise ValueError("evaluation pressure manifest must use valid_seen")
    if payload.get("full_task_manifest_sha256") != full_task_manifest_sha256:
        raise RuntimeError(
            "evaluation pressure manifest does not match the registered "
            "valid_seen task manifest"
        )
    rows = payload.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation pressure manifest requires tasks")

    available = {task.task_id: task for task in available_tasks}
    selected: list[TaskSpec] = []
    task_ids: list[str] = []
    declared_types: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("evaluation pressure task rows must be mappings")
        task_id = row.get("task_id")
        task_type = row.get("task_type")
        if not isinstance(task_id, str) or not isinstance(task_type, str):
            raise TypeError("evaluation pressure task identity must be text")
        task = available.get(task_id)
        if task is None:
            raise RuntimeError(f"evaluation pressure task is unavailable: {task_id}")
        if task.task_type != task_type:
            raise RuntimeError(
                f"evaluation pressure task type differs for {task_id}: "
                f"{task_type} != {task.task_type}"
            )
        selected.append(task)
        task_ids.append(task_id)
        declared_types.append(task_type)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("evaluation pressure task IDs must be unique")
    counts = Counter(declared_types)
    if set(counts.values()) != {2} or len(counts) != 6:
        raise ValueError(
            "evaluation pressure manifest must contain exactly two tasks "
            "from each of six task types"
        )

    manifest_sha256 = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    metadata = {
        **payload,
        "path": str(source.resolve()),
        "sha256": manifest_sha256,
        "task_count": len(selected),
        "denominators": [
            {"task_type": task_type, "count": counts[task_type]}
            for task_type in sorted(counts)
        ],
        "diagnostic_only": True,
        "reportable_as_valid_seen": False,
    }
    return tuple(selected), metadata


def diagnostic_denominators(
    tasks: Sequence[TaskSpec],
) -> tuple[TaskDenominator, ...]:
    counts = Counter(task.task_type for task in tasks)
    return tuple(
        TaskDenominator(task_type, counts[task_type])
        for task_type in sorted(counts)
    )
