from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from infoskill.domain.state import CanonicalAgentState, render_policy_message
from infoskill.integrations.alfworld import GroundingDataset
from infoskill.integrations.alfworld.grounding_io import sha256_file

from .providers import DemonstrationProvider, DemonstrationTrajectory


def prepare_alfworld_imitation_data(
    *,
    grounding_directory: str | Path,
    output_directory: str | Path,
    validation_fraction: float = 0.02,
    split_seed: int = 0,
    expected_trajectory_count: int | None = None,
) -> dict[str, object]:
    """Convert successful planner trajectories into step-level SFT pairs.

    The split is made by task/trajectory before expanding steps, which prevents
    near-identical adjacent states from leaking across train and validation.
    """

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if split_seed < 0:
        raise ValueError("split_seed must be non-negative")
    source = GroundingDataset.load(grounding_directory)
    if (
        expected_trajectory_count is not None
        and source.game_count != expected_trajectory_count
    ):
        raise ValueError(
            "planner trajectory count differs from the registered protocol: "
            f"expected {expected_trajectory_count}, got {source.game_count}"
        )
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    task_ids = sorted(source.samples_by_game)
    ranked = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(
            f"{split_seed}:{task_id}".encode("utf-8")
        ).digest(),
    )
    validation_count = max(1, round(len(ranked) * validation_fraction))
    validation_ids = set(ranked[:validation_count])
    rows: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    task_type_counts: dict[str, int] = {}
    for task_id in task_ids:
        split = "validation" if task_id in validation_ids else "train"
        samples = source.samples_by_game[task_id]
        if not samples:
            continue
        task_type = samples[0].state.task_type
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        for sample in samples:
            rows[split].append(_sft_row(sample.state, sample.expert_action))
    if not rows["train"] or not rows["validation"]:
        raise ValueError("trajectory split produced an empty train or validation set")
    for split, payloads in rows.items():
        _atomic_write_jsonl(destination / f"{split}.jsonl", payloads)
    source_checksums = source.manifest.get("source_checksums", {})
    parent_samples_sha256 = (
        source_checksums.get("parent_grounding_samples")
        if isinstance(source_checksums, dict)
        else None
    )
    source_samples_sha256 = sha256_file(source.root / "grounding_samples.jsonl")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "provider": "alfworld_verified_planner_grounding",
        "source_split": "train",
        "source_grounding_root": str(source.root),
        "source_grounding_manifest_sha256": source.manifest_sha256,
        "source_grounding_samples_sha256": source_samples_sha256,
        "source_planner_samples_sha256": (
            parent_samples_sha256
            if isinstance(parent_samples_sha256, str)
            else source_samples_sha256
        ),
        "trajectory_count": source.game_count,
        "expected_trajectory_count": expected_trajectory_count,
        "sample_count": source.sample_count,
        "train_trajectory_count": len(task_ids) - len(validation_ids),
        "validation_trajectory_count": len(validation_ids),
        "train_sample_count": len(rows["train"]),
        "validation_sample_count": len(rows["validation"]),
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "task_type_trajectory_counts": dict(sorted(task_type_counts.items())),
        "prompt_contract": "canonical_policy_message_v1",
        "response_contract": "think_action_v1",
    }
    _atomic_write_json(destination / "manifest.json", manifest)
    return manifest


def prepare_demonstration_imitation_data(
    *,
    provider: DemonstrationProvider,
    output_directory: str | Path,
    validation_fraction: float = 0.02,
    split_seed: int = 0,
    expected_trajectory_count: int | None = None,
    expected_source_checksums: dict[str, str] | None = None,
) -> dict[str, object]:
    """Prepare environment-native demonstrations without trajectory leakage."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if split_seed < 0:
        raise ValueError("split_seed must be non-negative")
    source_files = _provider_source_files(provider)
    source_checksums = {
        name: sha256_file(path) for name, path in sorted(source_files.items())
    }
    _validate_source_checksums(
        source_checksums,
        expected_source_checksums or {},
    )
    trajectories = provider.trajectories()
    if (
        expected_trajectory_count is not None
        and len(trajectories) != expected_trajectory_count
    ):
        raise ValueError(
            "demonstration trajectory count differs from the registered protocol: "
            f"expected {expected_trajectory_count}, got {len(trajectories)}"
        )
    environments = {item.environment for item in trajectories}
    if len(environments) != 1:
        raise ValueError("imitation corpus must contain exactly one environment")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    ranked = sorted(
        trajectories,
        key=lambda item: hashlib.sha256(
            f"{split_seed}:{item.trajectory_id}".encode("utf-8")
        ).digest(),
    )
    validation_count = max(1, round(len(ranked) * validation_fraction))
    if validation_count >= len(ranked):
        raise ValueError("trajectory split produced an empty training set")
    validation_ids = {item.trajectory_id for item in ranked[:validation_count]}
    rows: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    for trajectory in sorted(trajectories, key=lambda item: item.trajectory_id):
        split = "validation" if trajectory.trajectory_id in validation_ids else "train"
        rows[split].extend(_demonstration_rows(trajectory))
    for split, payloads in rows.items():
        if not payloads:
            raise ValueError(f"imitation {split} split is empty")
        _atomic_write_jsonl(destination / f"{split}.jsonl", payloads)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "provider": type(provider).__name__,
        "environment": next(iter(environments)),
        "source_split": getattr(provider, "source_split", "train"),
        "source_files": {name: str(path) for name, path in sorted(source_files.items())},
        "source_checksums": source_checksums,
        "trajectory_count": len(trajectories),
        "expected_trajectory_count": expected_trajectory_count,
        "sample_count": sum(len(item.steps) for item in trajectories),
        "train_trajectory_count": len(trajectories) - len(validation_ids),
        "validation_trajectory_count": len(validation_ids),
        "train_sample_count": len(rows["train"]),
        "validation_sample_count": len(rows["validation"]),
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "trajectory_id_sha256": _trajectory_id_sha256(trajectories),
        "prompt_contract": "environment_online_policy_message_v1",
        "response_contract": "think_action_v1",
    }
    _atomic_write_json(destination / "manifest.json", manifest)
    return manifest


def _validate_source_checksums(
    actual: dict[str, str],
    expected: dict[str, str],
) -> None:
    for name, expected_digest in sorted(expected.items()):
        actual_digest = actual.get(name)
        if actual_digest is None:
            raise ValueError(
                f"expected imitation source checksum has no source file: {name}"
            )
        if actual_digest != expected_digest:
            raise ValueError(
                "imitation source checksum differs from the registered protocol: "
                f"{name} expected {expected_digest}, got {actual_digest}"
            )


def _sft_row(state: CanonicalAgentState, action: str) -> dict[str, object]:
    return {
        "task_id": state.task_id,
        "task_type": state.task_type,
        "step_index": state.step_index,
        "prompt": render_policy_message(state, history_limit=2),
        "response": (
            f"<think>{_rationale(state)}</think>\n"
            f"<action>{action}</action>"
        ),
    }


def _rationale(state: CanonicalAgentState) -> str:
    if state.task_type == "pick_two_obj_and_place":
        delivered = sum(
            entry.executed_action.startswith(("move ", "put "))
            for entry in state.history
        )
        phase = "second object" if delivered else "first object"
        return f"Follow the verified {phase} plan and advance only with an admissible action."
    return (
        f"Follow the verified {state.task_type} plan and advance only with an "
        "admissible action."
    )


def _demonstration_rows(
    trajectory: DemonstrationTrajectory,
) -> list[dict[str, object]]:
    return [
        {
            "task_id": trajectory.trajectory_id,
            "task_type": trajectory.environment,
            "step_index": step_index,
            "prompt": step.prompt,
            "response": (
                f"<think>{_environment_rationale(step.action)}</think>\n"
                f"<action>{step.action}</action>"
            ),
        }
        for step_index, step in enumerate(trajectory.steps)
    ]


def _environment_rationale(action: str) -> str:
    if action.startswith("search["):
        return "Use a focused query containing the instruction's discriminating attributes."
    if action.lower() == "click[buy now]":
        return "The selected product and required options satisfy the instruction, so purchase it."
    if action.startswith("click["):
        return "Inspect or select the demonstrated admissible choice that advances the shopping goal."
    return "Follow the successful environment-native demonstration."


def _provider_source_files(provider: DemonstrationProvider) -> dict[str, Path]:
    method = getattr(provider, "source_files", None)
    if not callable(method):
        return {}
    payload = method()
    if not isinstance(payload, dict):
        raise ValueError("demonstration provider source_files must return a mapping")
    result: dict[str, Path] = {}
    for name, value in payload.items():
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(name)] = path
    return result


def _trajectory_id_sha256(
    trajectories: tuple[DemonstrationTrajectory, ...],
) -> str:
    content = "\n".join(sorted(item.trajectory_id for item in trajectories)) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _atomic_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    content = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
