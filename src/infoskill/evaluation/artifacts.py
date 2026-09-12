from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path


_CHECKPOINT_LOAD_STATUSES = {
    "not_requested",
    "pending",
    "loaded",
    "failed",
}


def write_evaluation_provenance(
    run_directory: Path,
    *,
    mode: str,
    evaluation_runtime: Mapping[str, object],
    evaluation_manifest: Mapping[str, object],
    policy_model: Mapping[str, object] | None,
    skill_conditioning: Mapping[str, object] | None = None,
    checkpoint_provenance_sha256: str | None = None,
    artifact_kind: str = "valid_seen_evaluation",
) -> dict[str, object]:
    """Write the immutable inputs that identify one evaluation run."""

    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": artifact_kind,
        "mode": mode,
        "evaluation_runtime": dict(evaluation_runtime),
        "evaluation_manifest": dict(evaluation_manifest),
    }
    if policy_model is not None:
        payload["policy_model"] = dict(policy_model)
    if skill_conditioning is not None:
        payload["skill_conditioning"] = dict(skill_conditioning)
    if checkpoint_provenance_sha256 is not None:
        payload["checkpoint_provenance_sha256"] = checkpoint_provenance_sha256
    _write_json_atomic(run_directory / "provenance.json", payload)
    return payload


def checkpoint_load_payload(
    *,
    backend: str,
    checkpoint: str | None,
    checkpoint_step: int,
    status: str,
    duration_seconds: float | None = None,
    worker_reports: Sequence[Mapping[str, object]] = (),
    error: BaseException | None = None,
) -> dict[str, object]:
    """Build a structured record for the portable-checkpoint load boundary."""

    if status not in _CHECKPOINT_LOAD_STATUSES:
        raise ValueError(f"unsupported checkpoint load status: {status}")
    requested = checkpoint is not None
    if status == "not_requested" and requested:
        raise ValueError("not_requested checkpoint status requires checkpoint=None")
    if status != "not_requested" and not requested:
        raise ValueError(f"checkpoint status {status!r} requires a checkpoint")
    if duration_seconds is not None and duration_seconds < 0:
        raise ValueError("checkpoint load duration must be non-negative")
    if error is not None and status != "failed":
        raise ValueError("checkpoint load errors require status='failed'")

    payload: dict[str, object] = {
        "schema_version": 1,
        "backend": backend,
        "checkpoint": checkpoint,
        "checkpoint_step": checkpoint_step,
        "requested": requested,
        "loaded": status == "loaded",
        "status": status,
        "worker_reports": [dict(report) for report in worker_reports],
    }
    if duration_seconds is not None:
        payload["duration_seconds"] = duration_seconds
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    return payload


def write_checkpoint_load(
    run_directory: Path,
    *,
    backend: str,
    checkpoint: str | None,
    checkpoint_step: int,
    status: str,
    duration_seconds: float | None = None,
    worker_reports: Sequence[Mapping[str, object]] = (),
    error: BaseException | None = None,
) -> dict[str, object]:
    payload = checkpoint_load_payload(
        backend=backend,
        checkpoint=checkpoint,
        checkpoint_step=checkpoint_step,
        status=status,
        duration_seconds=duration_seconds,
        worker_reports=worker_reports,
        error=error,
    )
    _write_json_atomic(run_directory / "checkpoint-load.json", payload)
    return payload


def write_evaluation_timing(
    run_directory: Path,
    values: Mapping[str, float],
) -> dict[str, object]:
    if any(value < 0 for value in values.values()):
        raise ValueError("evaluation timing values must be non-negative")
    payload: dict[str, object] = {
        "schema_version": 1,
        **{key: float(value) for key, value in values.items()},
    }
    _write_json_atomic(run_directory / "evaluation-timing.json", payload)
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
