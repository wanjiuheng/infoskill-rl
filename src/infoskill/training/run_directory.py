from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def resolve_training_run_directory(
    *,
    output_root: str | Path,
    profile_name: str,
    run_name: str | None,
    resume: str | None,
) -> tuple[Path, Path | None, bool]:
    """Resolve an in-place run or a named fork from an immutable checkpoint."""
    if resume is not None:
        checkpoint = Path(resume).expanduser().resolve()
        if checkpoint.parent.name != "checkpoints":
            raise ValueError("resume path must be a run checkpoints/step-* directory")
        if run_name is None:
            return checkpoint.parent.parent, checkpoint, False
        return _create_run_directory(output_root, run_name), checkpoint, True

    name = run_name or f"m0-{profile_name}"
    return _create_run_directory(output_root, name), None, False


def validate_resume_config(
    checkpoint: str | Path,
    current: Mapping[str, object],
    *,
    allow_gpu_change: bool,
) -> int:
    checkpoint_path = Path(checkpoint)
    path = checkpoint_path / "resolved_config.json"
    if not path.is_file():
        raise RuntimeError(f"resume checkpoint has no resolved config: {path}")
    previous = json.loads(path.read_text(encoding="utf-8"))
    previous_gpus = int(previous.get("num_gpus", 0))
    current_gpus = int(current.get("num_gpus", 0))

    previous_without_gpus = _with_runtime_defaults(previous)
    current_without_gpus = _with_runtime_defaults(current)
    previous_without_gpus.pop("num_gpus", None)
    current_without_gpus.pop("num_gpus", None)
    if previous_without_gpus != current_without_gpus:
        raise RuntimeError("resume configuration differs from the original run")
    if previous_gpus != current_gpus and not allow_gpu_change:
        raise RuntimeError(
            "changing GPU count during resume requires a new run_name"
        )
    return previous_gpus


def _with_runtime_defaults(config: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(config)
    runtime_options = normalized.get("runtime_options")
    if isinstance(runtime_options, Mapping):
        normalized_options = dict(runtime_options)
        normalized_options.setdefault("environment_workers", 1)
        normalized_options.setdefault("environment_backend", "individual")
        normalized_options.setdefault("cuda_memory_poll_interval_ms", 0)
        normalized_options.setdefault("policy_max_tokens_per_gpu", 16_384)
        normalized["runtime_options"] = normalized_options
    return normalized


def _create_run_directory(output_root: str | Path, name: str) -> Path:
    if not _RUN_NAME.fullmatch(name):
        raise ValueError("run_name must contain only letters, digits, '.', '_' or '-'")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output_root).expanduser().resolve() / f"{stamp}-{name}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory
