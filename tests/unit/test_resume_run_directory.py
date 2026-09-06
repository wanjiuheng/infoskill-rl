from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from infoskill.training.run_directory import (
    resolve_training_run_directory,
    validate_resume_config,
)


class ResumeRunDirectoryTests(unittest.TestCase):
    def test_resume_without_name_continues_in_source_run(self) -> None:
        root = Path.cwd() / "test-output"
        checkpoint = root / "source" / "checkpoints" / "step-000001"

        run, selected, forked = resolve_training_run_directory(
            output_root=root,
            profile_name="smoke",
            run_name=None,
            resume=str(checkpoint),
        )

        self.assertEqual(run, checkpoint.parent.parent.resolve())
        self.assertEqual(selected, checkpoint.resolve())
        self.assertFalse(forked)

    def test_named_resume_forks_into_a_new_run_directory(self) -> None:
        root = Path.cwd() / "test-output"
        checkpoint = root / "source" / "checkpoints" / "step-000001"
        with patch.object(Path, "mkdir") as mkdir:
            run, selected, forked = resolve_training_run_directory(
                output_root=root,
                profile_name="smoke",
                run_name="m0-smoke-4to2",
                resume=str(checkpoint),
            )

            self.assertNotEqual(run, checkpoint.parent.parent.resolve())
            self.assertEqual(run.parent, root.resolve())
            self.assertTrue(run.name.endswith("-m0-smoke-4to2"))
            mkdir.assert_called_once_with(parents=True, exist_ok=False)
            self.assertEqual(selected, checkpoint.resolve())
            self.assertTrue(forked)

    def test_only_forked_resume_may_change_gpu_count(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        previous = {
            "schema_version": 1,
            "mode": "no_skill",
            "num_gpus": 4,
            "app_config": {"model": "same"},
            "training_plan": {"max_updates": 2},
        }
        current = {**previous, "num_gpus": 2}
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(previous)),
        ):
            source_gpus = validate_resume_config(
                checkpoint,
                current,
                allow_gpu_change=True,
            )

            self.assertEqual(source_gpus, 4)
            with self.assertRaisesRegex(RuntimeError, "new run_name"):
                validate_resume_config(
                    checkpoint,
                    current,
                    allow_gpu_change=False,
                )

    def test_fork_still_rejects_non_gpu_configuration_changes(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        previous = {
            "num_gpus": 4,
            "app_config": {"model": "source"},
        }
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(previous)),
        ):
            with self.assertRaisesRegex(RuntimeError, "differs"):
                validate_resume_config(
                    checkpoint,
                    {"num_gpus": 2, "app_config": {"model": "changed"}},
                    allow_gpu_change=True,
                )

    def test_missing_historical_environment_workers_means_serial(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        previous = {
            "num_gpus": 4,
            "runtime_options": {"persistent_rollout_session": True},
        }
        current = {
            "num_gpus": 4,
            "runtime_options": {
                "persistent_rollout_session": True,
                "environment_workers": 1,
                "environment_backend": "individual",
            },
        }
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(previous)),
        ):
            source_gpus = validate_resume_config(
                checkpoint,
                current,
                allow_gpu_change=False,
            )

        self.assertEqual(source_gpus, 4)

    def test_historical_checkpoint_cannot_silently_adopt_native_batch(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        previous = {
            "num_gpus": 4,
            "runtime_options": {"persistent_rollout_session": True},
        }
        current = {
            "num_gpus": 4,
            "runtime_options": {
                "persistent_rollout_session": True,
                "environment_workers": 1,
                "environment_backend": "native_batch",
            },
        }
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(previous)),
        ):
            with self.assertRaisesRegex(RuntimeError, "differs"):
                validate_resume_config(
                    checkpoint,
                    current,
                    allow_gpu_change=False,
                )

    def test_historical_checkpoint_preserves_empty_cache_default(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        previous = {
            "num_gpus": 4,
            "runtime_options": {
                "persistent_rollout_session": True,
                "environment_workers": 1,
                "environment_backend": "native_batch",
            },
        }
        current = {
            "num_gpus": 4,
            "runtime_options": {
                **previous["runtime_options"],
                "rollout_empty_cache_between_steps": True,
            },
        }
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(previous)),
        ):
            source_gpus = validate_resume_config(
                checkpoint,
                current,
                allow_gpu_change=False,
            )

        self.assertEqual(source_gpus, 4)


if __name__ == "__main__":
    unittest.main()
