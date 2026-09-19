from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from infoskill.imitation.handoff import (
    create_handoff,
    load_handoff,
    validate_handoff_for_runtime,
)
from infoskill.persistence.model_identity import get_pinned_policy_model


class M1HandoffTests(unittest.TestCase):
    def test_handoff_is_content_bound_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "r": 16,
                        "lora_alpha": 32,
                        "target_modules": [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (adapter / "imitation-training-manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "actor_imitation_lora_sft",
                        "status": "complete",
                        "train_file_sha256": "a" * 64,
                        "validation_file_sha256": "b" * 64,
                        "policy_model": _policy_identity(),
                    }
                ),
                encoding="utf-8",
            )
            data_manifest = root / "data.json"
            data_manifest.write_text(
                json.dumps(
                    {
                        "sample_count": 10,
                        "trajectory_count": 10,
                        "expected_trajectory_count": 10,
                        "source_planner_samples_sha256": "c" * 64,
                    }
                ),
                encoding="utf-8",
            )
            skills = root / "skills.json"
            skills.write_text("{}", encoding="utf-8")
            skills.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "trajectory_count": 10,
                        "source_samples_sha256": "c" * 64,
                    }
                ),
                encoding="utf-8",
            )
            destination = root / "handoff"

            first = create_handoff(
                adapter_directory=adapter,
                imitation_manifest=data_manifest,
                skill_bank=skills,
                base_model_id="qwen2.5-7b-instruct",
                output_directory=destination,
            )
            loaded = load_handoff(destination)
            self.assertEqual(first["handoff_sha256"], loaded["handoff_sha256"])
            self.assertEqual(loaded["lora_rank"], 16)
            self.assertEqual(
                loaded["base_model_sha256"],
                _policy_identity()["sha256"],
            )
            validated, resolved = validate_handoff_for_runtime(
                destination,
                policy_model_identity=_policy_identity(),
            )
            self.assertEqual(validated["handoff_sha256"], first["handoff_sha256"])
            self.assertEqual(Path(resolved), destination.resolve())
            changed_data = json.loads(data_manifest.read_text(encoding="utf-8"))
            changed_data["source_planner_samples_sha256"] = "d" * 64
            data_manifest.write_text(json.dumps(changed_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different planner"):
                create_handoff(
                    adapter_directory=adapter,
                    imitation_manifest=data_manifest,
                    skill_bank=skills,
                    base_model_id="qwen2.5-7b-instruct",
                    output_directory=root / "mixed-handoff",
                )
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                create_handoff(
                    adapter_directory=adapter,
                    imitation_manifest=data_manifest,
                    skill_bank=skills,
                    base_model_id="qwen2.5-7b-instruct",
                    output_directory=destination,
                )

    def test_handoff_rejects_unregistered_base_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "r": 16,
                        "lora_alpha": 32,
                        "target_modules": [
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            identity = _policy_identity()
            identity["sha256"] = "0" * 64
            (adapter / "imitation-training-manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "actor_imitation_lora_sft",
                        "status": "complete",
                        "train_file_sha256": "a" * 64,
                        "validation_file_sha256": "b" * 64,
                        "policy_model": identity,
                    }
                ),
                encoding="utf-8",
            )
            data = root / "data.json"
            data.write_text(
                json.dumps(
                    {
                        "trajectory_count": 1,
                        "expected_trajectory_count": 1,
                        "source_planner_samples_sha256": "c" * 64,
                    }
                ),
                encoding="utf-8",
            )
            skills = root / "skills.json"
            skills.write_text("{}", encoding="utf-8")
            skills.with_suffix(".manifest.json").write_text(
                json.dumps(
                    {
                        "trajectory_count": 1,
                        "source_samples_sha256": "c" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "content-bound"):
                create_handoff(
                    adapter_directory=adapter,
                    imitation_manifest=data,
                    skill_bank=skills,
                    base_model_id="qwen2.5-7b-instruct",
                    output_directory=root / "handoff",
                )


def _policy_identity() -> dict[str, object]:
    pinned = get_pinned_policy_model("qwen2.5-7b-instruct")
    return {
        "algorithm": "infoskill-policy-model-v2",
        "model_id": pinned.model_id,
        "revision": pinned.revision,
        "sha256": pinned.sha256,
    }


if __name__ == "__main__":
    unittest.main()
