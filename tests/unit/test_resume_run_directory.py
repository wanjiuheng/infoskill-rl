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
    def test_named_fork_may_change_monitoring_eval_batch_size(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000050"
        previous = {
            "num_gpus": 3,
            "app_config": {"eval_batch_size": 12},
            "evaluation_manifest": {
                "split": "valid_seen",
                "task_count": 140,
                "task_manifest_sha256": "same",
                "eval_batch_size": 12,
                "comparison_role": "monitoring_only",
            },
        }
        current = {
            "num_gpus": 3,
            "app_config": {"eval_batch_size": 64},
            "evaluation_manifest": {
                "split": "valid_seen",
                "task_count": 140,
                "task_manifest_sha256": "same",
                "eval_batch_size": 64,
                "comparison_role": "monitoring_only",
            },
        }
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(previous)),
        ):
            source_gpus = validate_resume_config(
                checkpoint,
                current,
                allow_gpu_change=True,
                allow_performance_candidate_change=True,
            )

        self.assertEqual(source_gpus, 3)

    def test_named_fork_may_add_cuda_graph_evaluation_execution_mode(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000050"
        previous = {
            "num_gpus": 3,
            "evaluation_manifest": {
                "split": "valid_seen",
                "task_count": 140,
                "sha256": "same",
                "eval_batch_size": 12,
                "comparison_role": "nonregistered_monitoring_curve",
            },
        }
        current = {
            "num_gpus": 3,
            "evaluation_manifest": {
                "split": "valid_seen",
                "task_count": 140,
                "sha256": "same",
                "eval_batch_size": 64,
                "comparison_role": "nonregistered_monitoring_curve",
                "execution_mode": "cuda_graph",
            },
        }
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(previous)),
        ):
            source_gpus = validate_resume_config(
                checkpoint,
                current,
                allow_gpu_change=True,
                allow_performance_candidate_change=True,
            )

        self.assertEqual(source_gpus, 3)

    def test_in_place_resume_rejects_changed_evaluation_execution_mode(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000050"
        previous = {
            "num_gpus": 3,
            "evaluation_manifest": {
                "split": "valid_seen",
            },
        }
        current = {
            "num_gpus": 3,
            "evaluation_manifest": {
                "split": "valid_seen",
                "execution_mode": "cuda_graph",
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

    def test_in_place_resume_rejects_changed_monitoring_eval_batch_size(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000050"
        previous = {
            "num_gpus": 3,
            "app_config": {"eval_batch_size": 12},
        }
        current = {
            "num_gpus": 3,
            "app_config": {"eval_batch_size": 64},
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

    def test_named_fork_may_adopt_bounded_checkpoint_retention(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000050"
        previous = {
            "num_gpus": 3,
            "runtime_options": {"persistent_rollout_session": True},
        }
        current = {
            "num_gpus": 3,
            "runtime_options": {
                "persistent_rollout_session": True,
                "checkpoint_keep_recent": 5,
                "checkpoint_keep_best_valid": True,
            },
        }
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(previous)),
        ):
            source_gpus = validate_resume_config(
                checkpoint,
                current,
                allow_gpu_change=True,
                allow_performance_candidate_change=True,
            )

        self.assertEqual(source_gpus, 3)

    def test_in_place_resume_rejects_changed_checkpoint_retention(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000050"
        previous = {
            "num_gpus": 3,
            "runtime_options": {"persistent_rollout_session": True},
        }
        current = {
            "num_gpus": 3,
            "runtime_options": {
                "persistent_rollout_session": True,
                "checkpoint_keep_recent": 5,
                "checkpoint_keep_best_valid": True,
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

    def test_fork_may_change_only_registered_performance_candidates(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000050"
        previous = {
            "num_gpus": 3,
            "runtime_options": {
                "skip_unused_old_logprob_entropy": False,
                "rollout_max_batched_tokens": 16_384,
                "hybrid_prefix_cuda_graph": False,
                "fuse_kl_ppo_forward": False,
                "policy_max_tokens_per_gpu": 12_288,
            },
        }
        current = {
            "num_gpus": 3,
            "runtime_options": {
                "skip_unused_old_logprob_entropy": True,
                "rollout_max_batched_tokens": 32_768,
                "hybrid_prefix_cuda_graph": True,
                "hybrid_prefix_cuda_graph_custom_kernels": True,
                "hybrid_prefix_cuda_graph_use_inductor": False,
                "fuse_kl_ppo_forward": True,
                "policy_max_tokens_per_gpu": 12_288,
            },
        }
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(previous)),
        ):
            source_gpus = validate_resume_config(
                checkpoint,
                current,
                allow_gpu_change=True,
                allow_performance_candidate_change=True,
            )

        self.assertEqual(source_gpus, 3)

    def test_in_place_resume_rejects_changed_cuda_graph_kernel_policy(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000050"
        previous = {
            "num_gpus": 3,
            "runtime_options": {"hybrid_prefix_cuda_graph": True},
        }
        current = {
            "num_gpus": 3,
            "runtime_options": {
                "hybrid_prefix_cuda_graph": True,
                "hybrid_prefix_cuda_graph_custom_kernels": True,
                "hybrid_prefix_cuda_graph_use_inductor": False,
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

    def test_in_place_resume_rejects_performance_candidate_change(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000050"
        previous = {
            "num_gpus": 3,
            "runtime_options": {
                "skip_unused_old_logprob_entropy": False,
                "rollout_max_batched_tokens": 16_384,
            },
        }
        current = {
            "num_gpus": 3,
            "runtime_options": {
                "skip_unused_old_logprob_entropy": True,
                "rollout_max_batched_tokens": 32_768,
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

    def test_resume_may_extend_target_when_warmup_schedule_is_unchanged(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        previous = {
            "num_gpus": 3,
            "training_plan": {
                "profile": "smoke",
                "max_updates": 1,
                "action_minibatch_size": 16,
            },
        }
        current = {
            "num_gpus": 3,
            "training_plan": {
                "profile": "smoke",
                "max_updates": 2,
                "action_minibatch_size": 16,
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

        self.assertEqual(source_gpus, 3)

    def test_resume_rejects_shorter_target(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        previous = {
            "num_gpus": 3,
            "training_plan": {"profile": "smoke", "max_updates": 2},
        }
        current = {
            "num_gpus": 3,
            "training_plan": {"profile": "smoke", "max_updates": 1},
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

    def test_resume_rejects_extension_across_warmup_boundary(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        previous = {
            "num_gpus": 3,
            "training_plan": {"profile": "smoke", "max_updates": 32},
        }
        current = {
            "num_gpus": 3,
            "training_plan": {"profile": "smoke", "max_updates": 34},
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

    def test_resume_extension_still_rejects_other_plan_changes(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        previous = {
            "num_gpus": 3,
            "training_plan": {
                "profile": "smoke",
                "max_updates": 1,
                "action_minibatch_size": 16,
            },
        }
        current = {
            "num_gpus": 3,
            "training_plan": {
                "profile": "smoke",
                "max_updates": 2,
                "action_minibatch_size": 32,
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

    def test_new_null_m1_path_does_not_break_historical_control_resume(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        previous = {
            "num_gpus": 4,
            "app_config": {"paths": {"policy_model": "/same/model"}},
        }
        current = {
            "num_gpus": 4,
            "app_config": {
                "paths": {
                    "policy_model": "/same/model",
                    "grounding_data": None,
                }
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

    def test_historical_checkpoint_cannot_silently_adopt_new_token_budget(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        previous = {
            "num_gpus": 4,
            "runtime_options": {"persistent_rollout_session": True},
        }
        current = {
            "num_gpus": 4,
            "runtime_options": {
                "persistent_rollout_session": True,
                "policy_max_tokens_per_gpu": 12_288,
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

    def test_registered_policy_model_may_move_without_changing_identity(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        previous = {
            "num_gpus": 4,
            "app_config": {
                "policy_model_id": "alfworld-7b-sft-checkpoint-140",
                "paths": {"policy_model": "/old/model"},
            },
        }
        current = {
            "num_gpus": 4,
            "app_config": {
                "policy_model_id": "alfworld-7b-sft-checkpoint-140",
                "paths": {"policy_model": "/new/model"},
            },
        }
        provenance = {
            "policy_model": {
                "algorithm": "infoskill-policy-model-v2",
                "model_id": "alfworld-7b-sft-checkpoint-140",
                "revision": "Alfworld-7B-SFT/checkpoint-140",
                "sha256": (
                    "ede304d8ae0fb27df55a9bcf22482b8a7d83626a4711f9525e0388f7b3d39d99"
                ),
            },
        }
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(
                Path,
                "read_text",
                side_effect=(json.dumps(previous), json.dumps(provenance)),
            ),
        ):
            self.assertEqual(
                validate_resume_config(
                    checkpoint,
                    current,
                    allow_gpu_change=False,
                ),
                4,
            )

    def test_registered_model_resume_rejects_changed_provenance(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        config = {
            "num_gpus": 4,
            "app_config": {
                "policy_model_id": "alfworld-7b-sft-checkpoint-140",
                "paths": {"policy_model": "/same/model"},
            },
        }
        changed = {
            "policy_model": {
                "algorithm": "infoskill-policy-model-v2",
                "model_id": "alfworld-7b-sft-checkpoint-140",
                "revision": "Alfworld-7B-SFT/checkpoint-140",
                "sha256": "0" * 64,
            },
        }
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(
                Path,
                "read_text",
                side_effect=(json.dumps(config), json.dumps(changed)),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "policy provenance differs"):
                validate_resume_config(
                    checkpoint,
                    config,
                    allow_gpu_change=False,
                )

    def test_policy_model_id_cannot_change_during_resume(self) -> None:
        checkpoint = Path.cwd() / "source" / "checkpoints" / "step-000001"
        previous = {
            "num_gpus": 4,
            "app_config": {
                "policy_model_id": "approved-a",
                "paths": {"policy_model": "/same/model"},
            },
        }
        current = {
            "num_gpus": 4,
            "app_config": {
                "policy_model_id": "approved-b",
                "paths": {"policy_model": "/same/model"},
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


if __name__ == "__main__":
    unittest.main()
