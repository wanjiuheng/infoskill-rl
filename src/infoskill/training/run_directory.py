from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from infoskill.persistence.model_identity import provenance_matches_pinned_model


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
    matching_model_id = _matching_policy_model_id(previous, current)
    if matching_model_id is not None:
        _validate_checkpoint_policy_provenance(checkpoint_path, matching_model_id)
        previous_without_gpus = _without_policy_model_path(previous_without_gpus)
        current_without_gpus = _without_policy_model_path(current_without_gpus)
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
        normalized_options.setdefault("balance_policy_tokens_across_ranks", False)
        normalized["runtime_options"] = normalized_options
    return normalized


def _matching_policy_model_id(
    previous: Mapping[str, object],
    current: Mapping[str, object],
) -> str | None:
    previous_app = previous.get("app_config")
    current_app = current.get("app_config")
    if not isinstance(previous_app, Mapping) or not isinstance(current_app, Mapping):
        return None
    previous_id = previous_app.get("policy_model_id")
    current_id = current_app.get("policy_model_id")
    if (
        isinstance(previous_id, str)
        and bool(previous_id.strip())
        and previous_id == current_id
    ):
        return previous_id
    return None


def _validate_checkpoint_policy_provenance(
    checkpoint: Path,
    model_id: str,
) -> None:
    path = checkpoint / "provenance.json"
    if not path.is_file():
        raise RuntimeError(f"resume checkpoint has no policy provenance: {path}")
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if not provenance_matches_pinned_model(provenance, model_id=model_id):
        raise RuntimeError(
            "resume checkpoint policy provenance differs from the registered model"
        )


def _without_policy_model_path(config: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(config)
    app_config = normalized.get("app_config")
    if not isinstance(app_config, Mapping):
        return normalized
    normalized_app = dict(app_config)
    paths = normalized_app.get("paths")
    if isinstance(paths, Mapping):
        normalized_paths = dict(paths)
        normalized_paths.pop("policy_model", None)
        normalized_app["paths"] = normalized_paths
    normalized["app_config"] = normalized_app
    return normalized


def _create_run_directory(output_root: str | Path, name: str) -> Path:
    if not _RUN_NAME.fullmatch(name):
        raise ValueError("run_name must contain only letters, digits, '.', '_' or '-'")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output_root).expanduser().resolve() / f"{stamp}-{name}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory
