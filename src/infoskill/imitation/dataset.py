from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from infoskill.domain.state import CanonicalAgentState, render_policy_message
from infoskill.integrations.alfworld import GroundingDataset
from infoskill.integrations.alfworld.grounding_io import sha256_file


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
