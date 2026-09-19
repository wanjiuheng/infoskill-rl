from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from infoskill.imitation.handoff import create_handoff, load_handoff


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
                    }
                ),
                encoding="utf-8",
            )
            data_manifest = root / "data.json"
            data_manifest.write_text(json.dumps({"sample_count": 10}), encoding="utf-8")
            skills = root / "skills.json"
            skills.write_text("{}", encoding="utf-8")
            skills.with_suffix(".manifest.json").write_text(
                "{}", encoding="utf-8"
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
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                create_handoff(
                    adapter_directory=adapter,
                    imitation_manifest=data_manifest,
                    skill_bank=skills,
                    base_model_id="qwen2.5-7b-instruct",
                    output_directory=destination,
                )


if __name__ == "__main__":
    unittest.main()
