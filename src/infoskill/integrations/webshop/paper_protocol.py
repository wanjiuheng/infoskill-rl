"""Frozen task manifests for the public GiGPO WebShop protocol.

The upstream environment samples validation session positions from ``range(500)``
and constructs every worker with a distinct seed.  A position alone therefore
does not identify a task: the synthetic goal list is generated and shuffled
independently inside every worker.  This module freezes the resolved goal for
each slot so later evaluations do not depend on mutable RNG state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from infoskill.config import EvaluationConfig, TaskDenominator
from infoskill.episode import TaskSpec


PAPER128_SCHEMA_VERSION = 1
PAPER128_TASK_COUNT = 128
PAPER128_POOL_SIZE = 500
PAPER128_ENV_SEED = 0
PAPER128_VALIDATION_SEED_OFFSET = 1000
_PAPER128_SOURCE_FILE_NAMES = {
    "products": "items_shuffle_1000.json",
    "attributes": "items_ins_v2_1000.json",
    "human_instructions": "items_human_ins.json",
}

GoalFactory = Callable[[int], Sequence[Mapping[str, Any]]]


def canonical_json_sha256(value: object) -> str:
    """Return a stable SHA256 for JSON-compatible content."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_session_indices(
    *,
    validation_call_index: int = 0,
    task_count: int = PAPER128_TASK_COUNT,
    pool_size: int = PAPER128_POOL_SIZE,
    env_seed: int = PAPER128_ENV_SEED,
) -> tuple[int, ...]:
    """Reproduce the upstream stateful validation sampler for one call.

    ``validation_call_index=0`` is the validation-before-training draw.  Higher
    values deliberately advance the same legacy ``RandomState`` rather than
    deriving a new seed, matching the public environment implementation.
    """

    if validation_call_index < 0:
        raise ValueError("validation_call_index must be non-negative")
    if task_count <= 0 or task_count > pool_size:
        raise ValueError("task_count must be in the interval [1, pool_size]")
    rng = np.random.RandomState(env_seed + PAPER128_VALIDATION_SEED_OFFSET)
    selected: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] | None = None
    for _ in range(validation_call_index + 1):
        selected = rng.choice(range(pool_size), size=task_count, replace=False)
    assert selected is not None
    return tuple(int(value) for value in selected.tolist())


def _goal_payload(goal: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "asin",
        "instruction_text",
        "attributes",
        "price_upper",
        "goal_options",
    )
    missing = [name for name in required if name not in goal]
    if missing:
        raise ValueError(f"WebShop goal is missing required fields: {missing}")
    return {name: goal[name] for name in required}


def build_paper128_manifest(
    *,
    goal_factory: GoalFactory,
    source_files: Mapping[str, Path],
    environment_commit: str,
    validation_call_index: int = 0,
    task_count: int = PAPER128_TASK_COUNT,
    pool_size: int = PAPER128_POOL_SIZE,
    env_seed: int = PAPER128_ENV_SEED,
) -> dict[str, Any]:
    """Resolve and freeze one paper-aligned WebShop validation draw."""

    source_identity: dict[str, dict[str, Any]] = {}
    for name, unresolved in sorted(source_files.items()):
        path = unresolved.expanduser().resolve(strict=True)
        source_identity[name] = {
            # Keep the artifact portable across machines; identity comes from
            # content, not an installation-specific absolute path.
            "file_name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    session_indices = validation_session_indices(
        validation_call_index=validation_call_index,
        task_count=task_count,
        pool_size=pool_size,
        env_seed=env_seed,
    )
    tasks: list[dict[str, Any]] = []
    for slot, session_index in enumerate(session_indices):
        worker_seed = env_seed + PAPER128_VALIDATION_SEED_OFFSET + slot
        goals = goal_factory(worker_seed)
        if len(goals) < pool_size:
            raise ValueError(
                f"worker seed {worker_seed} produced only {len(goals)} goals; "
                f"the protocol requires at least {pool_size}"
            )
        goal = _goal_payload(goals[session_index])
        tasks.append(
            {
                "slot": slot,
                "worker_seed": worker_seed,
                "session_index": session_index,
                "goal": goal,
                "goal_sha256": canonical_json_sha256(goal),
                "instruction_sha256": hashlib.sha256(
                    str(goal["instruction_text"]).encode("utf-8")
                ).hexdigest(),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": PAPER128_SCHEMA_VERSION,
        "artifact": "INFO-SKILL frozen GiGPO-compatible WebShop validation tasks",
        "protocol": {
            "environment": "webshop",
            "source": "GiGPO public implementation",
            "use_small": True,
            "human_goals": False,
            "env_seed": env_seed,
            "validation_seed_offset": PAPER128_VALIDATION_SEED_OFFSET,
            "validation_call_index": validation_call_index,
            "candidate_session_start": 0,
            "candidate_session_stop": pool_size,
            "task_count": task_count,
            "sampling": "numpy-randomstate-choice-without-replacement",
            "worker_seed_rule": "env_seed + validation_seed_offset + slot",
            "max_steps": 15,
            "validation_temperature": 0.4,
            "validation_do_sample": True,
        },
        "environment_commit": environment_commit,
        "numpy_version": np.__version__,
        "source_files": source_identity,
        "tasks": tasks,
    }
    payload["task_sequence_sha256"] = canonical_json_sha256(tasks)
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    validate_paper128_manifest(payload)
    return payload


def validate_paper128_manifest(payload: Mapping[str, Any]) -> None:
    """Fail closed if a frozen manifest is malformed or has been modified."""

    if payload.get("schema_version") != PAPER128_SCHEMA_VERSION:
        raise ValueError("unsupported WebShop paper manifest schema")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("WebShop paper manifest is missing protocol metadata")
    if protocol.get("use_small") is not True:
        raise ValueError("paper-aligned WebShop manifest must use the small catalog")
    if protocol.get("human_goals") is not False:
        raise ValueError("paper-aligned WebShop manifest must use synthetic goals")
    if protocol.get("env_seed") != PAPER128_ENV_SEED:
        raise ValueError("paper-aligned WebShop manifest must use env_seed=0")
    if protocol.get("validation_seed_offset") != PAPER128_VALIDATION_SEED_OFFSET:
        raise ValueError("paper-aligned WebShop validation seed offset differs")
    if protocol.get("candidate_session_stop") != PAPER128_POOL_SIZE:
        raise ValueError("paper-aligned WebShop candidate pool differs from 500")
    if protocol.get("max_steps") != 15:
        raise ValueError("paper-aligned WebShop evaluation requires max_steps=15")
    if protocol.get("validation_temperature") != 0.4:
        raise ValueError(
            "paper-aligned WebShop evaluation requires validation_temperature=0.4"
        )
    if protocol.get("validation_do_sample") is not True:
        raise ValueError("paper-aligned WebShop evaluation requires sampled decoding")
    source_files = payload.get("source_files")
    if not isinstance(source_files, Mapping):
        raise ValueError("WebShop paper manifest has no source identities")
    expected_source_names = _PAPER128_SOURCE_FILE_NAMES
    if set(source_files) != set(expected_source_names):
        raise ValueError("WebShop paper manifest source names differ")
    for name, expected_file_name in expected_source_names.items():
        source = source_files[name]
        if not isinstance(source, Mapping) or source.get("file_name") != expected_file_name:
            raise ValueError(f"WebShop paper manifest source file differs: {name}")
    task_count = protocol.get("task_count")
    tasks = payload.get("tasks")
    if not isinstance(task_count, int) or not isinstance(tasks, list):
        raise ValueError("WebShop paper manifest has invalid task metadata")
    if len(tasks) != task_count:
        raise ValueError("WebShop paper manifest task count differs from protocol")
    if task_count != PAPER128_TASK_COUNT:
        raise ValueError("paper-aligned WebShop manifest must contain 128 tasks")
    slots = [task.get("slot") for task in tasks if isinstance(task, Mapping)]
    if slots != list(range(task_count)):
        raise ValueError("WebShop paper manifest slots are incomplete or reordered")
    validation_call_index = protocol.get("validation_call_index")
    if not isinstance(validation_call_index, int):
        raise ValueError("WebShop paper manifest has no validation call index")
    expected_indices = validation_session_indices(
        validation_call_index=validation_call_index
    )
    for slot, task in enumerate(tasks):
        if not isinstance(task, Mapping) or not isinstance(task.get("goal"), Mapping):
            raise ValueError("WebShop paper manifest contains an invalid task")
        if task.get("worker_seed") != PAPER128_VALIDATION_SEED_OFFSET + slot:
            raise ValueError("WebShop paper manifest worker seed mismatch")
        if task.get("session_index") != expected_indices[slot]:
            raise ValueError("WebShop paper manifest session index mismatch")
        if task.get("goal_sha256") != canonical_json_sha256(task["goal"]):
            raise ValueError("WebShop paper manifest goal checksum mismatch")
        expected_instruction_sha256 = hashlib.sha256(
            str(task["goal"].get("instruction_text")).encode("utf-8")
        ).hexdigest()
        if task.get("instruction_sha256") != expected_instruction_sha256:
            raise ValueError("WebShop paper manifest instruction checksum mismatch")
    if payload.get("task_sequence_sha256") != canonical_json_sha256(tasks):
        raise ValueError("WebShop paper manifest task sequence checksum mismatch")
    expected_manifest_sha256 = payload.get("manifest_sha256")
    without_checksum = dict(payload)
    without_checksum.pop("manifest_sha256", None)
    if expected_manifest_sha256 != canonical_json_sha256(without_checksum):
        raise ValueError("WebShop paper manifest checksum mismatch")


def load_paper128_manifest(
    path: Path,
    *,
    source_files: Mapping[str, Path] | None = None,
    environment_commit: str | None = None,
) -> dict[str, Any]:
    """Load a manifest and optionally bind it to the current runtime assets."""

    payload = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("WebShop paper manifest must be a JSON object")
    validate_paper128_manifest(payload)
    if environment_commit is not None and payload.get("environment_commit") != environment_commit:
        raise ValueError("WebShop paper manifest environment commit mismatch")
    if source_files is not None:
        recorded_sources = payload["source_files"]
        if set(recorded_sources) != set(source_files):
            raise ValueError("WebShop paper manifest source names differ")
        for name, unresolved in source_files.items():
            path = unresolved.expanduser().resolve(strict=True)
            recorded = recorded_sources[name]
            if not isinstance(recorded, Mapping):
                raise ValueError(f"WebShop paper manifest source is invalid: {name}")
            current = {
                "file_name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if dict(recorded) != current:
                raise ValueError(f"WebShop paper manifest source differs: {name}")
    return payload


def paper128_tasks(payload: Mapping[str, Any]) -> tuple[TaskSpec, ...]:
    """Convert one validated frozen manifest into the generic rollout seam."""

    validate_paper128_manifest(payload)
    return tuple(
        TaskSpec(
            task_id=f"webshop-paper128-{int(task['slot']):03d}",
            split="paper128",
            task_type="webshop",
            goal=str(task["goal"]["instruction_text"]).strip(),
        )
        for task in payload["tasks"]
    )


def paper128_evaluation_config(payload: Mapping[str, Any]) -> EvaluationConfig:
    """Build the exact denominator and task-sequence identity for paper128."""

    validate_paper128_manifest(payload)
    return EvaluationConfig(
        split="paper128",
        denominators=(TaskDenominator("webshop", PAPER128_TASK_COUNT),),
        manifest_sha256=str(payload["task_sequence_sha256"]),
    )


def write_paper128_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    validate_paper128_manifest(payload)
    destination = path.expanduser().resolve()
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        validate_paper128_manifest(existing)
        if existing != payload:
            raise FileExistsError(
                f"refusing to replace a different frozen manifest: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
