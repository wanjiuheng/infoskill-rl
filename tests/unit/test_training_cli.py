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


if __name__ == "__main__":
    unittest.main()
