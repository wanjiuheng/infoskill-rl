from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol

from infoskill.training import TaskScheduleState


class CheckpointRuntime(Protocol):
    def save_portable_state(self, directory: Path) -> Mapping[str, object]: ...

    def load_portable_state(self, directory: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainerCheckpointState:
    global_update: int
    schedule: TaskScheduleState
    semantic_counters: Mapping[str, int]


class CheckpointManager:
    """Commit rank-0 portable state last; incomplete directories are never resumable."""

    def __init__(
        self,
        root: str | Path,
        *,
        keep_recent: int = 2,
        minimum_free_bytes: int = 10 * 1024**3,
    ) -> None:
        if keep_recent <= 0:
            raise ValueError("keep_recent must be positive")
        if minimum_free_bytes < 0:
            raise ValueError("minimum_free_bytes cannot be negative")
        self.root = Path(root)
        self.keep_recent = keep_recent
        self.minimum_free_bytes = minimum_free_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        state: TrainerCheckpointState,
        runtime: CheckpointRuntime,
        resolved_config: Mapping[str, object],
        provenance: Mapping[str, object],
        permanent: bool = False,
    ) -> Path:
        free = shutil.disk_usage(self.root).free
        emergency = free < self.minimum_free_bytes
        label = f"step-{state.global_update:06d}" + ("-emergency" if emergency else "")
        destination = self.root / label
        temporary = self.root / f".{label}.incomplete"
        if destination.exists():
            raise RuntimeError(
                f"refusing to overwrite an existing committed checkpoint: {destination}"
            )
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        runtime_manifest = dict(runtime.save_portable_state(temporary / "runtime"))
        _write_json(temporary / "trainer_state.json", asdict(state))
        _write_json(temporary / "resolved_config.json", resolved_config)
        _write_json(temporary / "provenance.json", provenance)
        files = sorted(
            str(path.relative_to(temporary)).replace("\\", "/")
            for path in temporary.rglob("*")
            if path.is_file()
        )
        completion = {
            "schema_version": 1,
            "global_update": state.global_update,
            "portable": True,
            "runtime_manifest": runtime_manifest,
            "files": files,
            "permanent": permanent or emergency,
            "emergency": emergency,
        }
        _write_json(temporary / "checkpoint.complete.json", completion)
        os.replace(temporary, destination)
        if not (permanent or emergency):
            self._rotate_recent()
        if emergency:
            raise RuntimeError(
                f"free disk space is below {self.minimum_free_bytes} bytes; "
                f"emergency checkpoint committed at {destination}"
            )
        return destination

    def validate(self, path: str | Path) -> dict[str, object]:
        directory = Path(path)
        completion = directory / "checkpoint.complete.json"
        if not completion.is_file():
            raise RuntimeError(f"checkpoint is incomplete: {directory}")
        payload = json.loads(completion.read_text(encoding="utf-8"))
        for relative in payload.get("files", []):
            if not (directory / relative).is_file():
                raise RuntimeError(f"checkpoint file is missing: {relative}")
        return payload

    def load_trainer_state(self, path: str | Path) -> TrainerCheckpointState:
        directory = Path(path)
        self.validate(directory)
        payload = json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
        schedule = payload["schedule"]
        return TrainerCheckpointState(
            global_update=int(payload["global_update"]),
            schedule=TaskScheduleState(
                cursor=int(schedule["cursor"]),
                ordered_task_ids=tuple(schedule["ordered_task_ids"]),
            ),
            semantic_counters={
                str(key): int(value) for key, value in payload.get("semantic_counters", {}).items()
            },
        )

    def _rotate_recent(self) -> None:
        recent = []
        for directory in self.root.glob("step-[0-9]*"):
            completion = directory / "checkpoint.complete.json"
            if not completion.is_file():
                continue
            payload = json.loads(completion.read_text(encoding="utf-8"))
            if not payload.get("permanent", False):
                recent.append((int(payload["global_update"]), directory))
        for _, directory in sorted(recent)[: -self.keep_recent]:
            shutil.rmtree(directory)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
