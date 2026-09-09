from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from infoskill.config import SkillMode
from infoskill.learning import (
    DEFAULT_LOGPROB_ALIGNMENT_THRESHOLDS,
    LogprobAlignmentError,
)
from infoskill.training.m0 import _persist_logprob_alignment_failure


class TrainingFailureArtifactTests(unittest.TestCase):
    def test_alignment_failure_preserves_summary_thresholds_and_update(self) -> None:
        summary = {
            "token_count": 100,
            "logprob_abs_error_p99": 0.28432059,
            "max_sample_index": 7,
            "max_token_position": 31,
            "max_token_id": 151643,
        }
        error = LogprobAlignmentError(
            summary=summary,
            thresholds=DEFAULT_LOGPROB_ALIGNMENT_THRESHOLDS,
            failures=("logprob_abs_error_p99=0.31 > 0.3",),
        )
        run_directory = Path("run")

        with patch("infoskill.training.m0._write_json") as write_json:
            result = _persist_logprob_alignment_failure(
                run_directory=run_directory,
                error=error,
                mode=SkillMode.RAW_SKILL_PROMPT,
                attempted_global_update=0,
            )

        self.assertEqual(
            result,
            run_directory / "rollout-recompute-alignment-failure.json",
        )
        path, payload = write_json.call_args.args
        self.assertEqual(path, result)
        self.assertEqual(payload["mode"], "raw_skill_prompt")
        self.assertEqual(payload["attempted_global_update"], 0)
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["summary"], summary)
        self.assertEqual(payload["thresholds"]["error_p99_max"], 0.30)


if __name__ == "__main__":
    unittest.main()
