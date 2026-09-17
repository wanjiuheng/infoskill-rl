"""Paired clipping gate must detect fake activation without filesystem writes."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compare_m1_precapture_clip_ab import compare_clip_ab


class M1PrecaptureClipABTests(unittest.TestCase):
    def _compare(self, *, candidate_clip_metric: float = 1.0) -> dict:
        source = Path("source/step-000215").resolve()
        control = Path("joint").resolve()
        candidate = Path("separate").resolve()
        control_eval = Path("joint-eval").resolve()
        candidate_eval = Path("separate-eval").resolve()
        invocation = {"segment_start_update": 215, "segment_end_update": 225}

        def read_json(path: Path) -> dict:
            run = path.parent
            name = path.name
            if run in (control, candidate):
                clip = "joint" if run == control else "separate"
                if name == "resolved_config.json":
                    return {"mode": "infoskill", "runtime_options": {
                        "policy_gradient_clip_mode": clip,
                    }}
                if name == "provenance.json":
                    return {
                        "resume_source_checkpoint": str(source),
                        "resume_forked": True,
                        "invocation": invocation,
                    }
                if name == "training_summary.json":
                    return {"status": "paused", "global_update": 225, "max_updates": 445}
            if run in (control_eval, candidate_eval):
                if name == "checkpoint-load.json":
                    train = control if run == control_eval else candidate
                    return {
                        "checkpoint": str(train / "checkpoints" / "step-000225"),
                        "checkpoint_step": 225,
                    }
                if name == "provenance.json":
                    return {"evaluation_runtime": {
                        "hybrid_prefix_cuda_graph": True,
                        "lora_shrink_split_k_one": True,
                        "eval_batch_size": 64,
                        "num_gpus": 3,
                    }}
            raise AssertionError(f"unexpected JSON read: {path}")

        def train_metrics(run: Path) -> dict[int, dict]:
            metric = 0.0 if run == control else candidate_clip_metric
            return {
                step: {
                    "policy/separate_gradient_clipping": metric,
                    "policy/optimizer_step_applied": 1.0,
                    "aux/optimizer_step_applied": 1.0,
                    "policy/optimizer_skip_nonfinite": 0.0,
                    "perf/hybrid_prefix_cuda_graph": 1.0,
                    "perf/lora_shrink_split_k_one_verified": 1.0,
                    "policy/actor_clip_coefficient": 0.3,
                    "policy/projector_clip_coefficient": 0.3,
                    "policy/projector_to_actor_grad_norm_ratio": 80.0,
                    "grpo_signal/mixed_outcome_group_count": 3.0,
                    "rollout/success_rate": 0.25,
                }
                for step in range(216, 226)
            }

        workload = [
            {"task_id": f"task-{index}", "rollout_id": index}
            for index in range(64)
        ]
        joint_tasks = {
            f"task-{index}": {"won": index < 48}
            for index in range(140)
        }
        separate_tasks = {
            f"task-{index}": {"won": index < 55}
            for index in range(140)
        }
        eval_report = {
            "controls_valid": True,
            "baseline": {"success_count": 48},
            "candidate": {"success_count": 55},
            "candidate_minus_baseline": {
                "macro_success": 0.04,
                "overall_success": 0.05,
            },
        }
        with (
            patch("scripts.compare_m1_precapture_clip_ab._read_json", side_effect=read_json),
            patch("scripts.compare_m1_precapture_clip_ab._train_metrics", side_effect=train_metrics),
            patch("scripts.compare_m1_precapture_clip_ab._checkpoint_committed", return_value=True),
            patch("scripts.compare_m1_precapture_clip_ab._read_training_trace", return_value=workload),
            patch("scripts.compare_m1_precapture_clip_ab._evaluation_trace", side_effect=[joint_tasks, separate_tasks]),
            patch("scripts.compare_m1_precapture_clip_ab._exact_internal_external_trace", return_value=True),
            patch("scripts.compare_m1_precapture_clip_ab.compare_evaluations", return_value=eval_report),
        ):
            return compare_clip_ab(
                source, control, control_eval, candidate, candidate_eval
            )

    def test_valid_pair_is_only_a_preliminary_gain(self) -> None:
        report = self._compare()
        self.assertTrue(report["controls_valid"])
        self.assertEqual(
            report["classification"],
            "candidate_preliminary_gain_needs_confirmation",
        )
        self.assertEqual(report["paired_task_outcomes"]["gain_count"], 7)
        self.assertEqual(report["paired_task_outcomes"]["loss_count"], 0)
        self.assertEqual(len(report["training_workload_by_update"]), 10)

    def test_candidate_that_silently_used_joint_is_invalid(self) -> None:
        report = self._compare(candidate_clip_metric=0.0)
        self.assertFalse(report["controls_valid"])
        self.assertFalse(report["controls"]["clip_modes_executed_on_every_update"])
        self.assertEqual(report["classification"], "invalid_controls")


if __name__ == "__main__":
    unittest.main()
