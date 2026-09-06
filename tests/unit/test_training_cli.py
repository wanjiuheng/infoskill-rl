from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from infoskill.cli import main


class TrainingCliTests(unittest.TestCase):
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
        self.assertEqual(payload["policy_max_tokens_per_gpu"], 16_384)
        self.assertTrue(payload["balance_policy_tokens_across_ranks"])

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


if __name__ == "__main__":
    unittest.main()
