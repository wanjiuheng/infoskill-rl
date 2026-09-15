from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from infoskill.app_config import AppConfig
from infoskill.cli import _parser, main


class TrainingCliTests(unittest.TestCase):
    def test_eval_batch_diagnostic_controls_are_explicit(self) -> None:
        default = _parser().parse_args(
            [
                "eval",
                "--config",
                "configs/alfworld_qwen25_7b.yaml",
                "--mode",
                "infoskill",
            ]
        )
        diagnostic = _parser().parse_args(
            [
                "eval",
                "--config",
                "configs/alfworld_qwen25_7b.yaml",
                "--mode",
                "infoskill",
                "--diagnostic-task-manifest",
                "configs/m1_eval_batch_pressure_valid_seen.json",
                "--eval-batch-size",
                "12",
                "--cuda-memory-poll-interval-ms",
                "200",
            ]
        )

        self.assertIsNone(default.diagnostic_task_manifest)
        self.assertIsNone(default.eval_batch_size)
        self.assertEqual(default.cuda_memory_poll_interval_ms, 0)
        self.assertEqual(diagnostic.eval_batch_size, 12)
        self.assertEqual(diagnostic.cuda_memory_poll_interval_ms, 200)

    def test_infoskill_eval_grouped_conditioning_is_explicitly_opt_in(self) -> None:
        default = _parser().parse_args(
            [
                "eval",
                "--config",
                "configs/alfworld_qwen25_7b.yaml",
                "--mode",
                "infoskill",
            ]
        )
        optimized = _parser().parse_args(
            [
                "eval",
                "--config",
                "configs/alfworld_qwen25_7b.yaml",
                "--mode",
                "infoskill",
                "--grouped-infoskill-conditioning",
            ]
        )

        self.assertFalse(default.grouped_infoskill_conditioning)
        self.assertTrue(optimized.grouped_infoskill_conditioning)

    def test_infoskill_eval_cuda_graph_is_explicitly_opt_in(self) -> None:
        default = _parser().parse_args(
            [
                "eval",
                "--config",
                "configs/alfworld_qwen25_7b.yaml",
                "--mode",
                "infoskill",
            ]
        )
        optimized = _parser().parse_args(
            [
                "eval",
                "--config",
                "configs/alfworld_qwen25_7b.yaml",
                "--mode",
                "infoskill",
                "--hybrid-prefix-cuda-graph",
            ]
        )

        self.assertFalse(default.hybrid_prefix_cuda_graph)
        self.assertTrue(optimized.hybrid_prefix_cuda_graph)

    def test_m1_performance_candidates_are_explicitly_opt_in(self) -> None:
        default = _parser().parse_args(self._arguments())
        optimized = _parser().parse_args(
            self._arguments()
            + [
                "--skip-unused-old-logprob-entropy",
                "--rollout-max-batched-tokens",
                "32768",
                "--hybrid-prefix-cuda-graph",
                "--fuse-kl-ppo-forward",
            ]
        )

        self.assertFalse(default.skip_unused_old_logprob_entropy)
        self.assertEqual(default.rollout_max_batched_tokens, 16_384)
        self.assertFalse(default.hybrid_prefix_cuda_graph)
        self.assertFalse(default.fuse_kl_ppo_forward)
        self.assertTrue(optimized.skip_unused_old_logprob_entropy)
        self.assertEqual(optimized.rollout_max_batched_tokens, 32_768)
        self.assertTrue(optimized.hybrid_prefix_cuda_graph)
        self.assertTrue(optimized.fuse_kl_ppo_forward)

    def test_m1_policy_gradient_clip_candidate_is_explicitly_opt_in(self) -> None:
        default = _parser().parse_args(self._arguments())
        candidate = _parser().parse_args(
            self._arguments()
            + ["--policy-gradient-clip-mode", "separate"]
        )

        self.assertEqual(default.policy_gradient_clip_mode, "joint")
        self.assertEqual(candidate.policy_gradient_clip_mode, "separate")

    def test_training_segment_end_is_an_invocation_boundary(self) -> None:
        default = _parser().parse_args(self._arguments())
        bounded = _parser().parse_args(
            self._arguments() + ["--segment-end-update", "51"]
        )

        self.assertIsNone(default.segment_end_update)
        self.assertEqual(bounded.segment_end_update, 51)

    def test_checkpoint_best_valid_retention_is_explicitly_opt_in(self) -> None:
        default = _parser().parse_args(self._arguments())
        bounded = _parser().parse_args(
            self._arguments()
            + [
                "--checkpoint-keep-recent",
                "5",
                "--checkpoint-keep-best-valid",
            ]
        )

        self.assertEqual(default.checkpoint_keep_recent, 2)
        self.assertFalse(default.checkpoint_keep_best_valid)
        self.assertEqual(bounded.checkpoint_keep_recent, 5)
        self.assertTrue(bounded.checkpoint_keep_best_valid)

        output = io.StringIO()
        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(
                    self._arguments()
                    + [
                        "--checkpoint-keep-recent",
                        "5",
                        "--checkpoint-keep-best-valid",
                    ]
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["checkpoint_keep_recent"], 5)
        self.assertTrue(payload["checkpoint_keep_best_valid"])

        with patch("pathlib.Path.exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "checkpoint_keep_recent"):
                main(
                    self._arguments()
                    + ["--checkpoint-keep-recent", "0"]
                )

    def test_grounding_accepts_explicit_resume_and_inactivity_timeout(self) -> None:
        arguments = _parser().parse_args(
            [
                "grounding",
                "--config",
                "configs/alfworld_qwen25_7b.yaml",
                "--resume-run",
                "/runs/formal-grounding",
                "--worker-inactivity-timeout-seconds",
                "420",
            ]
        )

        self.assertEqual(arguments.resume_run, "/runs/formal-grounding")
        self.assertEqual(arguments.worker_inactivity_timeout_seconds, 420.0)

    def test_grounding_rescue_accepts_a_committed_snapshot_to_finalize(self) -> None:
        arguments = _parser().parse_args(
            [
                "grounding-timeout-rescue",
                "--config",
                "configs/alfworld_qwen25_7b.yaml",
                "--source-grounding-run",
                "/runs/formal-grounding",
                "--finalize-committed-rescue-run",
                "/runs/partial-rescue",
                "--run-name",
                "formal-grounding-rescued",
            ]
        )

        self.assertEqual(
            arguments.finalize_committed_rescue_run,
            "/runs/partial-rescue",
        )

    @staticmethod
    def _arguments() -> list[str]:
        return [
            "train",
            "--config",
            "configs/alfworld_qwen25_7b.yaml",
            "--mode",
            "no_skill",
            "--profile",
            "smoke",
            "--max-updates",
            "1",
            "--num-gpus",
            "4",
            "--dry-run",
        ]

    def test_no_skill_smoke_dry_run_resolves_without_loading_gpu_runtime(self) -> None:
        output = io.StringIO()

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(self._arguments())

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["mode"], "no_skill")
        self.assertEqual(payload["profile"], "smoke")
        self.assertEqual(payload["max_updates"], 1)
        self.assertEqual(payload["num_gpus"], 4)
        self.assertEqual(payload["trajectories_per_full_update"], 2)
        self.assertTrue(payload["persistent_rollout_session"])
        self.assertEqual(payload["environment_workers"], 1)
        self.assertEqual(payload["environment_backend"], "native_batch")
        self.assertFalse(payload["verbose_runtime_logs"])
        self.assertEqual(payload["cuda_memory_poll_interval_ms"], 0)
        self.assertEqual(payload["policy_max_tokens_per_gpu"], 12_288)
        self.assertTrue(payload["balance_policy_tokens_across_ranks"])

    def test_training_eval_batch_size_can_be_explicitly_overridden(self) -> None:
        output = io.StringIO()
        arguments = self._arguments() + ["--eval-batch-size", "12"]

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["eval_batch_size"], 12)

    def test_raw_skill_prompt_dry_run_uses_the_shared_training_interface(self) -> None:
        output = io.StringIO()
        arguments = self._arguments()
        arguments[arguments.index("no_skill")] = "raw_skill_prompt"

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["mode"], "raw_skill_prompt")
        self.assertEqual(payload["profile"], "smoke")
        self.assertEqual(payload["trajectories_per_full_update"], 2)
        self.assertEqual(payload["retrieval_mode"], "embedding")
        self.assertEqual(payload["raw_skill_prompt_format"], "full")

    def test_raw_skill_prompt_format_can_explicitly_use_compact_text(self) -> None:
        output = io.StringIO()
        arguments = self._arguments()
        arguments[arguments.index("no_skill")] = "raw_skill_prompt"
        arguments.extend(["--raw-skill-prompt-format", "compact"])

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        self.assertEqual(result, 0)
        self.assertEqual(
            json.loads(output.getvalue())["raw_skill_prompt_format"],
            "compact",
        )

    def test_raw_skill_retrieval_mode_can_be_overridden_without_editing_yaml(self) -> None:
        output = io.StringIO()
        arguments = self._arguments()
        arguments[arguments.index("no_skill")] = "raw_skill_prompt"
        arguments.extend(["--retrieval-mode", "template"])

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["retrieval_mode"], "template")

    def test_infoskill_training_dry_run_reports_complete_m1_contract(self) -> None:
        output = io.StringIO()
        arguments = self._arguments()
        arguments[arguments.index("no_skill")] = "infoskill"
        base = AppConfig.load("configs/alfworld_qwen25_7b.yaml")

        from dataclasses import replace

        configured = replace(
            base,
            paths=replace(base.paths, grounding_data="/runs/grounding"),
        )

        with (
            patch("infoskill.cli.AppConfig.load", return_value=configured),
            patch("pathlib.Path.exists", return_value=True),
            redirect_stdout(output),
        ):
            result = main(arguments)

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["mode"], "infoskill")
        self.assertEqual(payload["retrieval_mode"], "embedding")
        self.assertIsNone(payload["raw_skill_prompt_format"])
        self.assertTrue(payload["infoskill_auxiliary_enabled"])
        self.assertEqual(payload["infoskill_latent_mode"], "sample")
        self.assertEqual(payload["infoskill_soft_prefix_length"], 5)
        self.assertEqual(payload["policy_gradient_clip_mode"], "joint")

    def test_infoskill_separate_policy_gradient_clip_is_reported(self) -> None:
        output = io.StringIO()
        arguments = self._arguments()
        arguments[arguments.index("no_skill")] = "infoskill"
        arguments.extend(
            [
                "--grounding-data",
                "/runs/formal-grounding",
                "--policy-gradient-clip-mode",
                "separate",
            ]
        )

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["policy_gradient_clip_mode"], "separate")

    def test_non_infoskill_rejects_separate_policy_gradient_clip(self) -> None:
        with patch("pathlib.Path.exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "only for infoskill"):
                main(
                    self._arguments()
                    + ["--policy-gradient-clip-mode", "separate"]
                )

    def test_infoskill_training_requires_grounding_data(self) -> None:
        arguments = self._arguments()
        arguments[arguments.index("no_skill")] = "infoskill"

        with patch("pathlib.Path.exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "grounding_data"):
                main(arguments)

    def test_infoskill_grounding_data_can_be_overridden_from_cli(self) -> None:
        output = io.StringIO()
        arguments = self._arguments()
        arguments[arguments.index("no_skill")] = "infoskill"
        arguments.extend(["--grounding-data", "/runs/formal-grounding"])

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["mode"], "infoskill")
        self.assertEqual(payload["grounding_data"], "/runs/formal-grounding")

    def test_infoskill_grounding_override_reaches_training_runtime(self) -> None:
        arguments = self._arguments()
        arguments[arguments.index("no_skill")] = "infoskill"
        arguments.remove("--dry-run")
        arguments.extend(["--grounding-data", "/runs/formal-grounding"])

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "infoskill.training.m0.run_policy_training",
                return_value=0,
            ) as train,
        ):
            result = main(arguments)

        self.assertEqual(result, 0)
        self.assertEqual(
            train.call_args.kwargs["config"].paths.grounding_data,
            "/runs/formal-grounding",
        )

    def test_raw_skill_prompt_dispatches_to_shared_policy_training(self) -> None:
        arguments = self._arguments()
        arguments[arguments.index("no_skill")] = "raw_skill_prompt"
        arguments.remove("--dry-run")

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "infoskill.training.m0.run_m0_training",
                side_effect=AssertionError("legacy no-skill-only runner used"),
            ),
            patch(
                "infoskill.training.m0.run_policy_training",
                create=True,
                return_value=0,
            ) as train,
        ):
            result = main(arguments)

        self.assertEqual(result, 0)
        self.assertEqual(train.call_args.kwargs["mode"].value, "raw_skill_prompt")

    def test_persistent_rollout_session_allows_explicit_opt_out(self) -> None:
        output = io.StringIO()
        arguments = self._arguments() + ["--no-persistent-rollout-session"]

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        self.assertEqual(result, 0)
        self.assertFalse(json.loads(output.getvalue())["persistent_rollout_session"])

    def test_environment_worker_count_is_explicitly_configurable(self) -> None:
        output = io.StringIO()
        arguments = self._arguments() + [
            "--environment-backend",
            "individual",
            "--environment-workers",
            "64",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["environment_workers"], 64)

    def test_individual_environment_backend_is_available_for_rollback(self) -> None:
        output = io.StringIO()
        arguments = self._arguments() + ["--environment-backend", "individual"]

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        self.assertEqual(result, 0)
        self.assertEqual(
            json.loads(output.getvalue())["environment_backend"], "individual"
        )

    def test_native_environment_batch_rejects_thread_worker_count(self) -> None:
        arguments = self._arguments() + [
            "--environment-backend",
            "native_batch",
            "--environment-workers",
            "64",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "must remain 1"):
                main(arguments)

    def test_verbose_runtime_logs_can_be_enabled_for_debugging(self) -> None:
        output = io.StringIO()
        arguments = self._arguments() + ["--verbose-runtime-logs"]

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        self.assertEqual(result, 0)
        self.assertTrue(json.loads(output.getvalue())["verbose_runtime_logs"])

    def test_cuda_memory_polling_is_explicitly_diagnostic(self) -> None:
        output = io.StringIO()
        arguments = self._arguments() + [
            "--cuda-memory-poll-interval-ms",
            "200",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        self.assertEqual(result, 0)
        self.assertEqual(
            json.loads(output.getvalue())["cuda_memory_poll_interval_ms"], 200
        )

    def test_dynamic_token_budget_is_explicitly_configurable(self) -> None:
        output = io.StringIO()
        arguments = self._arguments() + [
            "--policy-max-tokens-per-gpu",
            "20480",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        self.assertEqual(result, 0)
        self.assertEqual(
            json.loads(output.getvalue())["policy_max_tokens_per_gpu"],
            20_480,
        )

    def test_dynamic_token_budget_must_fit_one_maximum_sequence(self) -> None:
        arguments = self._arguments() + [
            "--policy-max-tokens-per-gpu",
            "4096",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "must be at least"):
                main(arguments)

    def test_policy_rank_token_balance_allows_explicit_rollback(self) -> None:
        output = io.StringIO()
        arguments = self._arguments() + [
            "--no-balance-policy-tokens-across-ranks"
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        self.assertEqual(result, 0)
        self.assertFalse(
            json.loads(output.getvalue())["balance_policy_tokens_across_ranks"]
        )

    def test_no_skill_training_does_not_require_embedding_or_skill_files(self) -> None:
        output = io.StringIO()

        def exists(path: Path) -> bool:
            return path.as_posix() not in {
                "/models/Qwen3-Embedding-0.6B",
                "/workspace/SkillRL/memory_data/alfworld/claude_style_skills.json",
            }

        with patch("pathlib.Path.exists", autospec=True, side_effect=exists):
            with redirect_stdout(output):
                result = main(self._arguments())

        self.assertEqual(result, 0)

    def test_raw_template_training_does_not_require_embedding_model(self) -> None:
        output = io.StringIO()
        arguments = self._arguments()
        arguments[arguments.index("no_skill")] = "raw_skill_prompt"

        base = AppConfig.load("configs/alfworld_qwen25_7b.yaml")

        def load_template(path):
            from dataclasses import replace

            del path
            return replace(base, retrieval_mode="template")

        def exists(path: Path) -> bool:
            return not path.as_posix().endswith("/Qwen3-Embedding-0.6B")

        with (
            patch("infoskill.cli.AppConfig.load", side_effect=load_template),
            patch("pathlib.Path.exists", autospec=True, side_effect=exists),
        ):
            with redirect_stdout(output):
                result = main(arguments)

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["mode"], "raw_skill_prompt")

    def test_named_resume_dry_run_reports_checkpoint_fork(self) -> None:
        output = io.StringIO()
        arguments = self._arguments() + [
            "--resume",
            "/runs/source/checkpoints/step-000001",
            "--run-name",
            "m0-smoke-4to2",
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with redirect_stdout(output):
                result = main(arguments)

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(
            payload["resume"],
            "/runs/source/checkpoints/step-000001",
        )
        self.assertTrue(payload["resume_forked"])

    def test_verl_evaluation_accepts_gpu_and_portable_checkpoint_options(self) -> None:
        arguments = [
            "eval",
            "--config",
            "configs/alfworld_qwen25_7b.yaml",
            "--mode",
            "no_skill",
            "--backend",
            "verl",
            "--num-gpus",
            "4",
            "--policy-checkpoint",
            "/runs/pilot/checkpoints/step-000025",
        ]

        with patch("infoskill.cli._evaluate", return_value=0) as evaluate:
            result = main(arguments)

        parsed = evaluate.call_args.args[1]
        self.assertEqual(result, 0)
        self.assertEqual(parsed.backend, "verl")
        self.assertEqual(parsed.num_gpus, 4)
        self.assertEqual(
            parsed.policy_checkpoint,
            "/runs/pilot/checkpoints/step-000025",
        )

    def test_raw_skill_prompt_verl_evaluation_passes_the_mode_gate(self) -> None:
        arguments = [
            "eval",
            "--config",
            "configs/alfworld_qwen25_7b.yaml",
            "--mode",
            "raw_skill_prompt",
            "--backend",
            "verl",
            "--num-gpus",
            "4",
        ]

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "infoskill.integrations.alfworld.discover_tasks",
                side_effect=RuntimeError("reached task discovery"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "reached task discovery"):
                main(arguments)

    def test_infoskill_verl_evaluation_reaches_task_discovery_without_legacy_pt(self) -> None:
        arguments = [
            "eval",
            "--config",
            "configs/alfworld_qwen25_7b.yaml",
            "--mode",
            "infoskill",
            "--backend",
            "verl",
            "--num-gpus",
            "3",
        ]

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "infoskill.integrations.alfworld.discover_tasks",
                side_effect=RuntimeError("reached task discovery"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "reached task discovery"):
                main(arguments)

    def test_infoskill_transformers_evaluation_is_rejected_explicitly(self) -> None:
        arguments = [
            "eval",
            "--config",
            "configs/alfworld_qwen25_7b.yaml",
            "--mode",
            "infoskill",
            "--backend",
            "transformers",
            "--num-gpus",
            "1",
        ]

        with self.assertRaisesRegex(ValueError, "backend=verl"):
            main(arguments)

    def test_checkpoint_effect_accepts_portable_checkpoint_and_gpu_count(self) -> None:
        arguments = [
            "checkpoint-effect",
            "--config",
            "configs/alfworld_qwen25_7b.yaml",
            "--num-gpus",
            "4",
            "--policy-checkpoint",
            "/runs/pilot/checkpoints/step-000025",
            "--max-new-tokens",
            "48",
        ]

        with patch("infoskill.cli._checkpoint_effect", return_value=0) as diagnose:
            result = main(arguments)

        parsed = diagnose.call_args.args[1]
        self.assertEqual(result, 0)
        self.assertEqual(parsed.num_gpus, 4)
        self.assertEqual(parsed.max_new_tokens, 48)
        self.assertEqual(
            parsed.policy_checkpoint,
            "/runs/pilot/checkpoints/step-000025",
        )

    def test_m1_lora_reproducibility_accepts_checkpoint_and_execution_mode(self) -> None:
        arguments = [
            "m1-lora-reproducibility",
            "--config",
            "configs/alfworld_qwen25_7b.yaml",
            "--num-gpus",
            "3",
            "--policy-checkpoint",
            "/runs/formal/checkpoints/step-000205",
            "--case-count",
            "2",
            "--max-new-tokens",
            "24",
            "--no-hybrid-prefix-cuda-graph",
        ]

        with patch(
            "infoskill.cli._m1_lora_reproducibility",
            return_value=0,
            create=True,
        ) as diagnose:
            result = main(arguments)

        parsed = diagnose.call_args.args[1]
        self.assertEqual(result, 0)
        self.assertEqual(parsed.num_gpus, 3)
        self.assertEqual(parsed.case_count, 2)
        self.assertEqual(parsed.max_new_tokens, 24)
        self.assertFalse(parsed.hybrid_prefix_cuda_graph)

    def test_raw_skill_ab_dispatches_one_runtime_diagnostic(self) -> None:
        arguments = [
            "raw-skill-ab",
            "--config",
            "configs/alfworld_qwen25_7b.yaml",
            "--num-gpus",
            "4",
            "--tasks-per-type",
            "2",
        ]

        with patch("infoskill.cli._raw_skill_ab", return_value=0, create=True) as probe:
            result = main(arguments)

        self.assertEqual(result, 0)
        parsed = probe.call_args.args[1]
        self.assertEqual(parsed.num_gpus, 4)
        self.assertEqual(parsed.tasks_per_type, 2)

    def test_raw_skill_ab_accepts_the_skillrl_rl_exact_variant(self) -> None:
        arguments = [
            "raw-skill-ab",
            "--config",
            "configs/alfworld_qwen25_7b.yaml",
            "--num-gpus",
            "4",
            "--variants",
            "skillrl-rl-exact",
        ]

        with patch("infoskill.cli._raw_skill_ab", return_value=0, create=True) as probe:
            result = main(arguments)

        self.assertEqual(result, 0)
        self.assertEqual(probe.call_args.args[1].variants, ["skillrl-rl-exact"])

    def test_raw_skill_ab_accepts_the_skillrl_sft_exact_variant(self) -> None:
        arguments = [
            "raw-skill-ab",
            "--config",
            "configs/alfworld_qwen25_7b.yaml",
            "--num-gpus",
            "4",
            "--variants",
            "skillrl-sft-exact",
        ]

        with patch("infoskill.cli._raw_skill_ab", return_value=0, create=True) as probe:
            result = main(arguments)

        self.assertEqual(result, 0)
        self.assertEqual(probe.call_args.args[1].variants, ["skillrl-sft-exact"])

    def test_raw_skill_ab_accepts_the_sft_causal_variants(self) -> None:
        variants = [
            "skillrl-sft-shell-no-skills-deterministic",
            "no-skill-sampled-t0.4",
            "skillrl-sft-exact-sampled-t0.4",
        ]
        arguments = [
            "raw-skill-ab",
            "--config",
            "configs/alfworld_qwen25_7b.yaml",
            "--num-gpus",
            "4",
            "--variants",
            *variants,
        ]

        with patch("infoskill.cli._raw_skill_ab", return_value=0, create=True) as probe:
            result = main(arguments)

        self.assertEqual(result, 0)
        self.assertEqual(probe.call_args.args[1].variants, variants)

    def test_raw_skill_ab_accepts_the_unified_skill_causal_variants(self) -> None:
        variants = [
            "unified-no-skill-deterministic",
            "unified-empty-skills-deterministic",
            "unified-template-skills-deterministic",
            "unified-embedding-skills-deterministic",
        ]
        arguments = [
            "raw-skill-ab",
            "--config",
            "configs/alfworld_qwen25_7b.yaml",
            "--num-gpus",
            "4",
            "--variants",
            *variants,
        ]

        with patch("infoskill.cli._raw_skill_ab", return_value=0, create=True) as probe:
            result = main(arguments)

        self.assertEqual(result, 0)
        self.assertEqual(probe.call_args.args[1].variants, variants)


if __name__ == "__main__":
    unittest.main()
