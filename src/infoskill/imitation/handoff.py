from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def create_handoff(
    *,
    adapter_directory: str | Path,
    imitation_manifest: str | Path,
    skill_bank: str | Path,
    base_model_id: str,
    output_directory: str | Path,
) -> dict[str, object]:
    source = Path(adapter_directory).expanduser().resolve()
    destination = Path(output_directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"immutable m1-handoff already exists: {destination}")
    required = (source / "adapter_model.safetensors", source / "adapter_config.json")
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("adapter requires adapter_model.safetensors and adapter_config.json")
    training_manifest = source / "imitation-training-manifest.json"
    if not training_manifest.is_file():
        raise FileNotFoundError("adapter requires imitation-training-manifest.json")
    training_payload = json.loads(training_manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(training_payload, dict)
        or training_payload.get("kind") != "actor_imitation_lora_sft"
        or training_payload.get("status") != "complete"
    ):
        raise ValueError("actor imitation training manifest is not complete")
    config = json.loads(required[1].read_text(encoding="utf-8"))
    rank = int(config.get("r", 0))
    alpha = int(config.get("lora_alpha", 0))
    if rank != 16 or alpha != 32:
        raise ValueError("M1 handoff requires LoRA rank=16 and alpha=32")
    expected_targets = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    if set(config.get("target_modules", ())) != expected_targets:
        raise ValueError("M1 handoff LoRA target modules differ from the runtime")
    data_path = Path(imitation_manifest).expanduser().resolve()
    skills_path = Path(skill_bank).expanduser().resolve()
    skills_manifest_path = skills_path.with_suffix(".manifest.json")
    if not data_path.is_file() or not skills_path.is_file():
        raise FileNotFoundError("imitation manifest and planner skill bank are required")
    if not skills_manifest_path.is_file():
        raise FileNotFoundError("planner skill bank provenance manifest is missing")
    destination.mkdir(parents=True, exist_ok=False)
    for source_path in required:
        shutil.copy2(source_path, destination / source_path.name)
    shutil.copy2(training_manifest, destination / training_manifest.name)
    shutil.copy2(data_path, destination / "imitation-manifest.json")
    shutil.copy2(skills_path, destination / "skill-bank.json")
    shutil.copy2(skills_manifest_path, destination / "skill-bank-manifest.json")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "kind": "actor_imitation_m1_handoff",
        "base_model_id": base_model_id,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "adapter_model_sha256": _sha256(destination / required[0].name),
        "adapter_config_sha256": _sha256(destination / required[1].name),
        "imitation_training_manifest": training_manifest.name,
        "imitation_training_manifest_sha256": _sha256(
            destination / training_manifest.name
        ),
        "source_imitation_manifest": str(data_path),
        "imitation_manifest": "imitation-manifest.json",
        "imitation_manifest_sha256": _sha256(destination / "imitation-manifest.json"),
        "source_skill_bank": str(skills_path),
        "skill_bank": "skill-bank.json",
        "skill_bank_sha256": _sha256(destination / "skill-bank.json"),
        "skill_bank_manifest": "skill-bank-manifest.json",
        "skill_bank_manifest_sha256": _sha256(
            destination / "skill-bank-manifest.json"
        ),
    }
    manifest["handoff_sha256"] = _payload_sha256(manifest)
    _atomic_json(destination / "m1-handoff.json", manifest)
    _atomic_json(
        destination / "checkpoint.complete.json",
        {
            "schema_version": 1,
            "kind": "actor_imitation_m1_handoff",
            "handoff_sha256": manifest["handoff_sha256"],
        },
    )
    return manifest


def load_handoff(directory: str | Path) -> dict[str, object]:
    root = Path(directory).expanduser().resolve()
    manifest_path = root / "m1-handoff.json"
    marker = root / "checkpoint.complete.json"
    if not manifest_path.is_file() or not marker.is_file():
        raise FileNotFoundError("m1-handoff is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("handoff_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "handoff_sha256"}
    if expected != _payload_sha256(unsigned):
        raise ValueError("m1-handoff manifest checksum mismatch")
    checks = {
        "adapter_model.safetensors": "adapter_model_sha256",
        "adapter_config.json": "adapter_config_sha256",
        "imitation-training-manifest.json": (
            "imitation_training_manifest_sha256"
        ),
        "imitation-manifest.json": "imitation_manifest_sha256",
        "skill-bank.json": "skill_bank_sha256",
        "skill-bank-manifest.json": "skill_bank_manifest_sha256",
    }
    for name, field in checks.items():
        if _sha256(root / name) != manifest.get(field):
            raise ValueError(f"m1-handoff content checksum mismatch: {name}")
    if int(manifest.get("lora_rank", 0)) != 16 or int(
        manifest.get("lora_alpha", 0)
    ) != 32:
        raise ValueError("m1-handoff LoRA configuration is incompatible with M1")
    training = json.loads(
        (root / "imitation-training-manifest.json").read_text(encoding="utf-8")
    )
    if training.get("status") != "complete":
        raise ValueError("m1-handoff imitation training is incomplete")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
