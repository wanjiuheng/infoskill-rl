from __future__ import annotations

import shutil
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from infoskill.app_config import AppConfig
from infoskill.persistence import model_identity
from infoskill.persistence.model_identity import (
    PinnedPolicyModel,
    fingerprint_policy_model,
    verify_policy_model_identity,
)
from infoskill.training import TrainingProfile, resolve_training_plan
from infoskill.training.m0 import run_m0_training


class PolicyModelIdentityTests(unittest.TestCase):
    def _scratch(self) -> Path:
        root = Path.cwd() / "test-output" / f"policy-model-{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, root, True)
        return root

    @staticmethod
    def _model(root: Path, name: str, *, weights: bytes = b"weights") -> Path:
        model = root / name
        model.mkdir()
        (model / "config.json").write_text('{"model_type":"qwen2"}\n')
        (model / "modeling_custom.py").write_text("MODEL_VERSION = 1\n")
        (model / "tokenizer_config.json").write_text('{"chat_template":"x"}\n')
        (model / "model-00001-of-00001.safetensors").write_bytes(weights)
        return model

    def test_fingerprint_is_path_independent_and_content_sensitive(self) -> None:
        root = self._scratch()
        first = self._model(root, "first")
        second = self._model(root, "second")

        first_identity = fingerprint_policy_model(first)
        second_identity = fingerprint_policy_model(second)

        self.assertEqual(first_identity.sha256, second_identity.sha256)
        self.assertEqual(first_identity.file_count, 4)
        self.assertEqual(first_identity.total_bytes, second_identity.total_bytes)

        (second / "modeling_custom.py").write_text("MODEL_VERSION = 2\n")
        self.assertNotEqual(
            first_identity.sha256,
            fingerprint_policy_model(second).sha256,
        )
        (second / "modeling_custom.py").write_text("MODEL_VERSION = 1\n")
        (second / "model-00001-of-00001.safetensors").write_bytes(b"changed")
        self.assertNotEqual(
            first_identity.sha256,
            fingerprint_policy_model(second).sha256,
        )

    def test_verification_requires_registered_id_and_matching_sha256(self) -> None:
        model = self._model(self._scratch(), "model")
        expected = fingerprint_policy_model(model).sha256
        pinned = PinnedPolicyModel(
            model_id="test-sft",
            revision="test/checkpoint-1",
            sha256=expected,
        )

        with patch.dict(
            model_identity._PINNED_POLICY_MODELS,
            {"test-sft": pinned},
        ):
            identity = verify_policy_model_identity(model, model_id="test-sft")
        self.assertEqual(identity.revision, "test/checkpoint-1")
        self.assertEqual(identity.model_id, "test-sft")

        with self.assertRaisesRegex(ValueError, "policy_model_id"):
            verify_policy_model_identity(
                model,
                model_id=None,
            )
        with self.assertRaisesRegex(ValueError, "not registered"):
            verify_policy_model_identity(
                model,
                model_id="unregistered-model",
            )
        with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
            with patch.dict(
                model_identity._PINNED_POLICY_MODELS,
                {
                    "test-sft": replace(pinned, sha256="0" * 64),
                },
            ):
                verify_policy_model_identity(model, model_id="test-sft")

    def test_model_requires_config_and_weights(self) -> None:
        model = self._scratch()
        (model / "config.json").write_text("{}\n")

        with self.assertRaisesRegex(RuntimeError, "weight files"):
            fingerprint_policy_model(model)

    def test_primary_config_pins_the_registered_base_model(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")

        self.assertEqual(config.policy_model_id, "qwen2.5-7b-instruct")
        self.assertEqual(
            config.paths.policy_model,
            "/root/autodl-tmp/wjh/models/Qwen/Qwen2.5-7B-Instruct",
        )

    def test_qwen3_config_pins_the_registered_base_model(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen3_1p7b.yaml")
        pinned = model_identity.get_pinned_policy_model(config.policy_model_id)

        self.assertEqual(config.policy_model_id, "qwen3-1.7b")
        self.assertEqual(
            config.paths.policy_model,
            "/root/autodl-tmp/wjh/models/Qwen/Qwen3-1.7B",
        )
        self.assertEqual(pinned.revision, "Qwen/Qwen3-1.7B")
        self.assertEqual(
            pinned.sha256,
            "9b0a7e2fff78e5746a5564de06f43386d277891b672be818fa45c10855716c5b",
        )

    def test_sft_comparison_config_remains_registered(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b_sft.yaml")

        self.assertEqual(
            config.policy_model_id,
            "alfworld-7b-sft-checkpoint-140",
        )

    def test_original_qwen_config_pins_registered_base_model(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b_instruct.yaml")
        pinned = model_identity.get_pinned_policy_model(config.policy_model_id)

        self.assertEqual(config.policy_model_id, "qwen2.5-7b-instruct")
        self.assertEqual(
            config.paths.policy_model,
            "/root/autodl-tmp/wjh/models/Qwen/Qwen2.5-7B-Instruct",
        )
        self.assertEqual(pinned.revision, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(
            pinned.sha256,
            "8305dee0a659a8f9e0650129eaaf584006338a42f237d071ef5cdbaed91fc14a",
        )

    def test_policy_model_id_rejects_surrounding_whitespace(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")

        with self.assertRaisesRegex(ValueError, "surrounding whitespace"):
            replace(config, policy_model_id=" qwen2.5-7b-instruct ").validate()

    def test_m0_rejects_adapter_as_shared_initialization(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")
        config = replace(
            config,
            paths=replace(config.paths, policy_adapter="existing-adapter"),
        )

        with self.assertRaisesRegex(ValueError, "policy_adapter must be null"):
            run_m0_training(
                config=config,
                plan=resolve_training_plan(TrainingProfile.SMOKE),
                num_gpus=1,
                run_name=None,
                resume=None,
            )

    def test_m0_verifies_model_before_discovering_tasks(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")
        with (
            patch(
                "infoskill.training.m0.verify_policy_model_identity",
                side_effect=RuntimeError("fingerprint mismatch"),
            ) as verify,
            patch("infoskill.training.m0.discover_tasks") as discover,
        ):
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                run_m0_training(
                    config=config,
                    plan=resolve_training_plan(TrainingProfile.SMOKE),
                    num_gpus=1,
                    run_name=None,
                    resume=None,
                )

        verify.assert_called_once_with(
            config.paths.policy_model,
            model_id="qwen2.5-7b-instruct",
        )
        discover.assert_not_called()


if __name__ == "__main__":
    unittest.main()
