from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_infoskill_efficacy_repeats import compare_repeated_evaluations


class InfoSkillEfficacyRepeatTests(unittest.TestCase):
    def _evaluation(
        self,
        root: Path,
        name: str,
        *,
        checkpoint: str,
        macro: float,
        overall: float,
    ) -> Path:
        run = root / name
        run.mkdir()
        summary = {
            "is_complete": True,
            "evaluated": 140,
            "macro_success": macro,
            "overall_success": overall,
            "invalid_action_rate": 0.1,
            "task_manifest_sha256": "fixed-manifest",
            "per_task_type_success": {"type-a": macro},
        }
        resolved = {
            "schema_version": 1,
            "evaluation_runtime": {
                "checkpoint_step": 205,
                "policy_checkpoint": checkpoint,
                "eval_batch_size": 64,
                "hybrid_prefix_cuda_graph": True,
            },
        }
        loaded = {
            "requested": True,
            "loaded": True,
            "status": "loaded",
            "checkpoint": checkpoint,
            "checkpoint_step": 205,
            "worker_reports": [
                {"lora_state_loaded": True, "infoskill_state_loaded": True}
            ],
        }
        for filename, payload in (
            ("valid_seen_summary.json", summary),
            ("resolved_config.json", resolved),
            ("checkpoint-load.json", loaded),
        ):
            (run / filename).write_text(json.dumps(payload), encoding="utf-8")
        return run

    def test_selects_only_when_two_run_mean_clears_both_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pairs = (
                (
                    self._evaluation(
                        root, "c1", checkpoint="joint", macro=0.30, overall=0.31
                    ),
                    self._evaluation(
                        root, "s1", checkpoint="separate", macro=0.35, overall=0.34
                    ),
                ),
                (
                    self._evaluation(
                        root, "c2", checkpoint="joint", macro=0.32, overall=0.32
                    ),
                    self._evaluation(
                        root, "s2", checkpoint="separate", macro=0.35, overall=0.34
                    ),
                ),
            )

            report = compare_repeated_evaluations(pairs)

        self.assertTrue(report["controls_valid"])
        self.assertEqual(report["classification"], "candidate_selected")
        self.assertEqual(report["exit_code"], 0)
        self.assertAlmostEqual(report["aggregate"]["macro_delta"], 0.04)
        self.assertAlmostEqual(report["aggregate"]["overall_delta"], 0.025)

    def test_rejects_candidate_after_required_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pairs = (
                (
                    self._evaluation(
                        root, "c1", checkpoint="joint", macro=0.32, overall=0.34
                    ),
                    self._evaluation(
                        root, "s1", checkpoint="separate", macro=0.30, overall=0.33
                    ),
                ),
                (
                    self._evaluation(
                        root, "c2", checkpoint="joint", macro=0.31, overall=0.32
                    ),
                    self._evaluation(
                        root, "s2", checkpoint="separate", macro=0.32, overall=0.33
                    ),
                ),
            )

            report = compare_repeated_evaluations(pairs)

        self.assertTrue(report["controls_valid"])
        self.assertEqual(
            report["classification"], "candidate_rejected_after_repeat"
        )
        self.assertEqual(report["exit_code"], 4)

    def test_rejects_cross_repeat_checkpoint_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pairs = (
                (
                    self._evaluation(
                        root, "c1", checkpoint="joint", macro=0.30, overall=0.30
                    ),
                    self._evaluation(
                        root, "s1", checkpoint="separate", macro=0.34, overall=0.34
                    ),
                ),
                (
                    self._evaluation(
                        root,
                        "c2",
                        checkpoint="wrong-joint",
                        macro=0.30,
                        overall=0.30,
                    ),
                    self._evaluation(
                        root, "s2", checkpoint="separate", macro=0.34, overall=0.34
                    ),
                ),
            )

            report = compare_repeated_evaluations(pairs)

        self.assertFalse(report["controls_valid"])
        self.assertEqual(report["classification"], "invalid_controls")
        self.assertEqual(report["exit_code"], 2)


if __name__ == "__main__":
    unittest.main()
