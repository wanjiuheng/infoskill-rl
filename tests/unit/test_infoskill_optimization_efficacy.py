from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.compare_infoskill_optimization_efficacy import compare_evaluations


class InfoSkillOptimizationEfficacyTests(unittest.TestCase):
    def test_macro_is_primary_even_when_overall_is_lower(self) -> None:
        with TemporaryDirectory() as temporary:
            baseline, candidate = self._runs(Path(temporary))
            self._write_summary(baseline, macro=0.25, overall=0.30)
            self._write_summary(candidate, macro=0.26, overall=0.29)

            report = compare_evaluations(baseline, candidate)

        self.assertTrue(report["controls_valid"])
        self.assertTrue(report["passed"])
        self.assertEqual(report["ordering"], "candidate_better_on_macro")

    def test_overall_breaks_an_exact_macro_tie(self) -> None:
        with TemporaryDirectory() as temporary:
            baseline, candidate = self._runs(Path(temporary))
            self._write_summary(baseline, macro=0.25, overall=0.30)
            self._write_summary(candidate, macro=0.25, overall=0.29)

            report = compare_evaluations(baseline, candidate)

        self.assertFalse(report["passed"])
        self.assertEqual(
            report["ordering"],
            "macro_tied_candidate_worse_on_overall",
        )

    def test_protocol_mismatch_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            baseline, candidate = self._runs(Path(temporary))
            self._write_summary(baseline, macro=0.25, overall=0.30)
            self._write_summary(candidate, macro=0.30, overall=0.35)
            resolved = self._read(candidate / "resolved_config.json")
            resolved["eval_batch_size"] = 12
            self._write(candidate / "resolved_config.json", resolved)

            report = compare_evaluations(baseline, candidate)

        self.assertFalse(report["control_checks"]["same_evaluation_protocol"])
        self.assertFalse(report["passed"])

    def test_eval_batch_size_difference_can_be_explicitly_controlled(self) -> None:
        with TemporaryDirectory() as temporary:
            baseline, candidate = self._runs(Path(temporary))
            self._write_summary(baseline, macro=0.25, overall=0.30)
            self._write_summary(candidate, macro=0.26, overall=0.31)
            resolved = self._read(candidate / "resolved_config.json")
            resolved["eval_batch_size"] = 64
            resolved["app_config"] = {"eval_batch_size": 64}
            resolved["evaluation_manifest"] = {"eval_batch_size": 64}
            runtime = resolved["evaluation_runtime"]
            assert isinstance(runtime, dict)
            runtime["eval_batch_size"] = 64
            self._write(candidate / "resolved_config.json", resolved)
            baseline_resolved = self._read(baseline / "resolved_config.json")
            baseline_resolved["app_config"] = {"eval_batch_size": 8}
            baseline_resolved["evaluation_manifest"] = {"eval_batch_size": 8}
            baseline_runtime = baseline_resolved["evaluation_runtime"]
            assert isinstance(baseline_runtime, dict)
            baseline_runtime["eval_batch_size"] = 8
            self._write(baseline / "resolved_config.json", baseline_resolved)
            self._make_checkpoints_identical(baseline, candidate)

            report = compare_evaluations(
                baseline,
                candidate,
                allow_eval_batch_size_difference=True,
                require_same_checkpoint=True,
            )

        self.assertTrue(report["control_checks"]["same_evaluation_protocol"])
        self.assertTrue(report["allow_eval_batch_size_difference"])
        self.assertTrue(report["passed"])

    def test_graph_execution_fields_require_an_explicit_allowlist(self) -> None:
        with TemporaryDirectory() as temporary:
            baseline, candidate = self._runs(Path(temporary))
            self._write_summary(baseline, macro=0.25, overall=0.30)
            self._write_summary(candidate, macro=0.26, overall=0.31)
            resolved = self._read(candidate / "resolved_config.json")
            runtime = resolved["evaluation_runtime"]
            assert isinstance(runtime, dict)
            runtime["hybrid_prefix_cuda_graph"] = True
            runtime["hybrid_prefix_cuda_graph_custom_kernels"] = True
            runtime["hybrid_prefix_cuda_graph_use_inductor"] = False
            self._write(candidate / "resolved_config.json", resolved)

            report = compare_evaluations(baseline, candidate)

        self.assertFalse(report["control_checks"]["same_evaluation_protocol"])
        self.assertFalse(report["passed"])

    def test_graph_execution_fields_can_be_explicitly_controlled(self) -> None:
        with TemporaryDirectory() as temporary:
            baseline, candidate = self._runs(Path(temporary))
            self._write_summary(baseline, macro=0.25, overall=0.30)
            self._write_summary(candidate, macro=0.26, overall=0.31)
            resolved = self._read(candidate / "resolved_config.json")
            runtime = resolved["evaluation_runtime"]
            assert isinstance(runtime, dict)
            runtime["hybrid_prefix_cuda_graph"] = True
            runtime["hybrid_prefix_cuda_graph_custom_kernels"] = True
            runtime["hybrid_prefix_cuda_graph_use_inductor"] = False
            self._write(candidate / "resolved_config.json", resolved)
            self._make_checkpoints_identical(baseline, candidate)

            report = compare_evaluations(
                baseline,
                candidate,
                allowed_runtime_differences=(
                    "hybrid_prefix_cuda_graph",
                    "hybrid_prefix_cuda_graph_custom_kernels",
                    "hybrid_prefix_cuda_graph_use_inductor",
                ),
                require_same_checkpoint=True,
            )

        self.assertTrue(report["control_checks"]["same_evaluation_protocol"])
        self.assertTrue(report["control_checks"]["same_checkpoint"])
        self.assertTrue(report["passed"])

    def test_split_k_one_can_be_explicitly_controlled(self) -> None:
        with TemporaryDirectory() as temporary:
            baseline, candidate = self._runs(Path(temporary))
            self._write_summary(baseline, macro=0.25, overall=0.30)
            self._write_summary(candidate, macro=0.26, overall=0.31)
            resolved = self._read(candidate / "resolved_config.json")
            runtime = resolved["evaluation_runtime"]
            assert isinstance(runtime, dict)
            runtime["lora_shrink_split_k_one"] = True
            self._write(candidate / "resolved_config.json", resolved)
            self._make_checkpoints_identical(baseline, candidate)

            report = compare_evaluations(
                baseline,
                candidate,
                allowed_runtime_differences=("lora_shrink_split_k_one",),
                require_same_checkpoint=True,
            )

        self.assertTrue(report["control_checks"]["same_evaluation_protocol"])
        self.assertTrue(report["control_checks"]["same_checkpoint"])
        self.assertTrue(report["passed"])

    def test_unexpected_protocol_difference_still_fails_with_allowlist(self) -> None:
        with TemporaryDirectory() as temporary:
            baseline, candidate = self._runs(Path(temporary))
            self._write_summary(baseline, macro=0.25, overall=0.30)
            self._write_summary(candidate, macro=0.30, overall=0.35)
            resolved = self._read(candidate / "resolved_config.json")
            resolved["eval_batch_size"] = 12
            self._write(candidate / "resolved_config.json", resolved)

            report = compare_evaluations(
                baseline,
                candidate,
                allowed_runtime_differences=("hybrid_prefix_cuda_graph",),
            )

        self.assertFalse(report["control_checks"]["same_evaluation_protocol"])
        self.assertFalse(report["passed"])

    def test_same_checkpoint_can_be_required(self) -> None:
        with TemporaryDirectory() as temporary:
            baseline, candidate = self._runs(Path(temporary))
            self._write_summary(baseline, macro=0.25, overall=0.30)
            self._write_summary(candidate, macro=0.30, overall=0.35)

            report = compare_evaluations(
                baseline,
                candidate,
                require_same_checkpoint=True,
            )

        self.assertFalse(report["control_checks"]["same_checkpoint"])
        self.assertFalse(report["passed"])

    @classmethod
    def _runs(cls, root: Path) -> tuple[Path, Path]:
        baseline = root / "baseline"
        candidate = root / "candidate"
        for name, run in (("baseline", baseline), ("candidate", candidate)):
            run.mkdir()
            cls._write(
                run / "resolved_config.json",
                {
                    "mode": "infoskill",
                    "eval_batch_size": 8,
                    "evaluation_runtime": {
                        "backend": "verl",
                        "checkpoint_step": 60,
                        "policy_checkpoint": f"/runs/{name}/step-000060",
                        "num_gpus": 3,
                    },
                },
            )
            cls._write(
                run / "checkpoint-load.json",
                {
                    "requested": True,
                    "loaded": True,
                    "status": "loaded",
                    "checkpoint": f"/runs/{name}/step-000060",
                    "checkpoint_step": 60,
                    "worker_reports": [
                        {
                            "lora_state_loaded": True,
                            "infoskill_state_loaded": True,
                        }
                    ],
                },
            )
        return baseline, candidate

    @classmethod
    def _write_summary(
        cls,
        run: Path,
        *,
        macro: float,
        overall: float,
    ) -> None:
        cls._write(
            run / "valid_seen_summary.json",
            {
                "is_complete": True,
                "evaluated": 140,
                "task_manifest_sha256": "fixed-manifest",
                "macro_success": macro,
                "overall_success": overall,
                "invalid_action_rate": 0.1,
                "per_task_type_success": {
                    "type-a": macro,
                    "type-b": macro,
                },
            },
        )

    @classmethod
    def _make_checkpoints_identical(cls, baseline: Path, candidate: Path) -> None:
        baseline_load = cls._read(baseline / "checkpoint-load.json")
        candidate_load = cls._read(candidate / "checkpoint-load.json")
        checkpoint = baseline_load["checkpoint"]
        candidate_load["checkpoint"] = checkpoint
        cls._write(candidate / "checkpoint-load.json", candidate_load)

        candidate_resolved = cls._read(candidate / "resolved_config.json")
        runtime = candidate_resolved["evaluation_runtime"]
        assert isinstance(runtime, dict)
        runtime["policy_checkpoint"] = checkpoint
        cls._write(candidate / "resolved_config.json", candidate_resolved)

    @staticmethod
    def _write(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _read(path: Path) -> dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        return payload


if __name__ == "__main__":
    unittest.main()
