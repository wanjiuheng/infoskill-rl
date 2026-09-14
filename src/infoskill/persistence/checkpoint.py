from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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


@dataclass(frozen=True, slots=True)
class PortableCheckpoint:
    directory: Path
    runtime_directory: Path
    global_update: int


def resolve_portable_checkpoint(path: str | Path) -> PortableCheckpoint:
    """Validate a committed checkpoint and locate its portable runtime state."""

    directory = Path(path).expanduser().resolve()
    payload = _validated_checkpoint_payload(directory)
    if payload.get("portable") is not True:
        raise RuntimeError(f"checkpoint is not portable: {directory}")
    runtime_directory = directory / "runtime"
    actor_directory = runtime_directory / "actor"
    if not actor_directory.is_dir():
        raise RuntimeError(f"checkpoint has no portable actor state: {directory}")
    try:
        global_update = int(payload["global_update"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"checkpoint has no valid global update: {directory}") from error
    if global_update < 0:
        raise RuntimeError(f"checkpoint has a negative global update: {directory}")
    actor_manifest_path = actor_directory / "actor_manifest.json"
    if not actor_manifest_path.is_file():
        raise RuntimeError(f"checkpoint has no portable actor manifest: {directory}")
    actor_manifest = json.loads(actor_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(actor_manifest, dict):
        raise RuntimeError(f"portable actor manifest is not an object: {directory}")
    try:
        actor_global_update = int(actor_manifest["global_step"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"portable actor manifest has no valid global step: {directory}"
        ) from error
    if actor_global_update != global_update:
        raise RuntimeError(
            "checkpoint and portable actor global updates differ: "
            f"{global_update} != {actor_global_update}"
        )
    runtime_manifest = payload.get("runtime_manifest")
    if runtime_manifest is not None and not isinstance(runtime_manifest, dict):
        raise RuntimeError(f"checkpoint has invalid runtime manifest: {directory}")
    if isinstance(runtime_manifest, dict) and runtime_manifest.get(
        "infoskill_modules_included"
    ) is True:
        infoskill_manifest = actor_directory / "infoskill" / "infoskill_manifest.json"
        if not infoskill_manifest.is_file():
            raise RuntimeError(
                f"checkpoint has no portable INFO-SKILL state: {directory}"
            )
    return PortableCheckpoint(
        directory=directory,
        runtime_directory=runtime_directory,
        global_update=global_update,
    )


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
        self.root = Path(root).expanduser().resolve()
        self.keep_recent = keep_recent
        self.minimum_free_bytes = minimum_free_bytes
        self._best_checkpoint: Path | None = None
        self.root.mkdir(parents=True, exist_ok=True)

    def set_best_checkpoint(self, path: str | Path | None) -> None:
        """Protect one committed checkpoint in this manager's own directory."""

        if path is None:
            self._best_checkpoint = None
            self._rotate_recent()
            return
        checkpoint = Path(path).expanduser().resolve()
        if checkpoint.parent != self.root:
            raise ValueError(
                "best checkpoint must be a direct child of the managed checkpoint root"
            )
        self.validate(checkpoint)
        self._best_checkpoint = checkpoint
        self._rotate_recent()

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
        if not emergency:
            self._rotate_recent()
        if emergency:
            raise RuntimeError(
                f"free disk space is below {self.minimum_free_bytes} bytes; "
                f"emergency checkpoint committed at {destination}"
            )
        return destination

    def validate(self, path: str | Path) -> dict[str, object]:
        return _validated_checkpoint_payload(Path(path))

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
        recent: list[tuple[int, Path]] = []
        for directory in self.root.glob("step-[0-9]*"):
            resolved = directory.resolve()
            if resolved.parent != self.root:
                raise RuntimeError(
                    f"refusing to rotate a checkpoint outside the managed root: {directory}"
                )
            completion = directory / "checkpoint.complete.json"
            if not completion.is_file():
                continue
            payload = json.loads(completion.read_text(encoding="utf-8"))
            if not payload.get("permanent", False):
                recent.append((int(payload["global_update"]), directory))
        recent.sort(key=lambda item: item[0])
        retained = {
            directory.resolve()
            for _, directory in recent[-self.keep_recent :]
        }
        if self._best_checkpoint is not None:
            if self._best_checkpoint.is_dir():
                retained.add(self._best_checkpoint)
            else:
                self._best_checkpoint = None
        for update, directory in recent:
            if directory.resolve() in retained:
                continue
            self._validate_rotation_target(directory)
            self._append_retention_audit(
                event="checkpoint_delete_intent",
                checkpoint=directory.name,
                global_update=update,
            )
            shutil.rmtree(directory)
            self._append_retention_audit(
                event="checkpoint_deleted",
                checkpoint=directory.name,
                global_update=update,
            )

    def _validate_rotation_target(self, directory: Path) -> None:
        resolved = directory.resolve()
        if resolved.parent != self.root:
            raise RuntimeError(
                f"refusing to delete a checkpoint outside the managed root: {directory}"
            )
        if re.fullmatch(r"step-[0-9]{6}", directory.name) is None:
            raise RuntimeError(
                f"refusing to delete an unsafe checkpoint path: {directory}"
            )
        self.validate(directory)

    def _append_retention_audit(
        self,
        *,
        event: str,
        checkpoint: str,
        global_update: int,
    ) -> None:
        payload = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "checkpoint": checkpoint,
            "global_update": global_update,
            "reason": "outside_recent_and_best",
            "keep_recent": self.keep_recent,
            "best_checkpoint": (
                self._best_checkpoint.name
                if self._best_checkpoint is not None
                else None
            ),
        }
        audit_path = self.root / "retention-audit.jsonl"
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validated_checkpoint_payload(directory: Path) -> dict[str, object]:
    completion = directory / "checkpoint.complete.json"
    if not completion.is_file():
        raise RuntimeError(f"checkpoint is incomplete: {directory}")
    payload = json.loads(completion.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint completion manifest is not an object: {directory}")
    files = payload.get("files")
    if not isinstance(files, list) or not all(
        isinstance(relative, str) for relative in files
    ):
        raise RuntimeError(f"checkpoint completion manifest has invalid files: {directory}")
    for relative in files:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"checkpoint file path is unsafe: {relative}")
        if not (directory / relative_path).is_file():
            raise RuntimeError(f"checkpoint file is missing: {relative}")
    return payload
