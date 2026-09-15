from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import compare_infoskill_training_optimization_runs as gate
from scripts.compare_infoskill_training_optimization_runs import (
    _candidate_option_checks,
    _checkpoint_step,
    _compare_training_trace_workload,
    _compare_values,
    _without_candidate_options,
)


class InfoSkillTrainingOptimizationGateTests(unittest.TestCase):
    def test_training_workload_ignores_stochastic_trajectory_content(self) -> None:
        baseline = [
            {"task_id": "task-b", "rollout_id": 1, "steps": ["open"]},
            {"task_id": "task-a", "rollout_id": 0, "steps": ["look"]},
        ]
        candidate = [
            {"task_id": "task-a", "rollout_id": 0, "steps": ["go north"]},
            {"task_id": "task-b", "rollout_id": 1, "steps": ["take mug"]},
        ]

        report = _compare_training_trace_workload(baseline, candidate)
        missing = _compare_training_trace_workload(baseline, candidate[:1])
        duplicate = _compare_training_trace_workload(
            baseline,
            [candidate[0], candidate[0]],
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["same_trajectory_keys"])
        self.assertFalse(missing["passed"])
        self.assertFalse(duplicate["passed"])

    def test_candidate_modes_require_only_their_registered_changes(self) -> None:
        baseline = {
            "skip_unused_old_logprob_entropy": False,
            "rollout_max_batched_tokens": 16_384,
        }
        entropy_only = {
            "skip_unused_old_logprob_entropy": True,
            "rollout_max_batched_tokens": 16_384,
        }
        combined = {
            "skip_unused_old_logprob_entropy": True,
            "rollout_max_batched_tokens": 32_768,
        }
        cuda_graph = {
            "skip_unused_old_logprob_entropy": False,
            "rollout_max_batched_tokens": 16_384,
            "hybrid_prefix_cuda_graph": True,
            "fuse_kl_ppo_forward": False,
        }
        fused_kl_ppo = {
            "skip_unused_old_logprob_entropy": False,
            "rollout_max_batched_tokens": 16_384,
            "hybrid_prefix_cuda_graph": False,
            "fuse_kl_ppo_forward": True,
        }
        separate_grad_clip = {
            "skip_unused_old_logprob_entropy": False,
            "rollout_max_batched_tokens": 16_384,
            "hybrid_prefix_cuda_graph": False,
            "fuse_kl_ppo_forward": False,
            "policy_gradient_clip_mode": "separate",
        }

        self.assertTrue(
            all(
                _candidate_option_checks(
                    baseline,
                    entropy_only,
                    candidate_mode="entropy-only",
                ).values()
            )
        )
        self.assertTrue(
            all(
                _candidate_option_checks(
                    baseline,
                    separate_grad_clip,
                    candidate_mode="separate-grad-clip",
                ).values()
            )
        )
        self.assertTrue(
            all(
                _candidate_option_checks(
                    baseline,
                    cuda_graph,
                    candidate_mode="cuda-graph",
                ).values()
            )
        )
        self.assertTrue(
            all(
                _candidate_option_checks(
                    baseline,
                    fused_kl_ppo,
                    candidate_mode="fused-kl-ppo",
                ).values()
            )
        )
        self.assertFalse(
            all(
                _candidate_option_checks(
                    baseline,
                    combined,
                    candidate_mode="entropy-only",
                ).values()
            )
        )
        self.assertTrue(
            all(
                _candidate_option_checks(
                    baseline,
                    combined,
                    candidate_mode="combined",
                ).values()
            )
        )

    def test_only_registered_candidate_options_are_ignored(self) -> None:
        baseline = {
            "mode": "infoskill",
            "runtime_options": {
                "skip_unused_old_logprob_entropy": False,
                "rollout_max_batched_tokens": 16_384,
                "hybrid_prefix_cuda_graph": False,
                "fuse_kl_ppo_forward": False,
                "policy_gradient_clip_mode": "joint",
                "policy_max_tokens_per_gpu": 12_288,
            },
        }
        candidate = {
            "mode": "infoskill",
            "runtime_options": {
                "skip_unused_old_logprob_entropy": True,
                "rollout_max_batched_tokens": 32_768,
                "hybrid_prefix_cuda_graph": True,
                "fuse_kl_ppo_forward": True,
                "policy_gradient_clip_mode": "separate",
                "policy_max_tokens_per_gpu": 12_288,
            },
        }

        self.assertEqual(
            _without_candidate_options(baseline),
            _without_candidate_options(candidate),
        )
        candidate["runtime_options"]["policy_max_tokens_per_gpu"] = 16_384
        self.assertNotEqual(
            _without_candidate_options(baseline),
            _without_candidate_options(candidate),
        )

    def test_separate_clip_preserves_an_approved_cuda_graph_control(self) -> None:
        baseline = {
            "skip_unused_old_logprob_entropy": False,
            "rollout_max_batched_tokens": 16_384,
            "hybrid_prefix_cuda_graph": True,
            "fuse_kl_ppo_forward": False,
            "policy_gradient_clip_mode": "joint",
        }
        candidate = {
            **baseline,
            "policy_gradient_clip_mode": "separate",
        }

        self.assertTrue(
            all(
                _candidate_option_checks(
                    baseline,
                    candidate,
                    candidate_mode="separate-grad-clip",
                ).values()
            )
        )

    def test_checkpoint_step_is_fail_closed(self) -> None:
        self.assertEqual(_checkpoint_step("/run/checkpoints/step-000050"), 50)
        self.assertEqual(_checkpoint_step("/run/checkpoints/latest"), -1)
        self.assertEqual(_checkpoint_step(None), -1)

    def test_tensor_comparison_accepts_only_tiny_numeric_drift(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        left = {"weight": torch.tensor([1.0, 2.0])}
        close = {"weight": torch.tensor([1.0, 2.0 + 1e-7])}
        changed = {"weight": torch.tensor([1.0, 2.01])}
        close_report = _compare_values(
            left,
            close,
            path="state",
            atol=1e-7,
            rtol=1e-6,
            mismatch_limit=20,
        )
        changed_report = _compare_values(
            left,
            changed,
            path="state",
            atol=1e-7,
            rtol=1e-6,
            mismatch_limit=20,
        )

        self.assertEqual(close_report["mismatches"], [])
        self.assertNotEqual(changed_report["mismatches"], [])

    def test_one_update_gate_requires_parity_memory_and_speedup(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            baseline.mkdir()
            candidate.mkdir()
            common_resolved = {
                "mode": "infoskill",
                "num_gpus": 3,
                "runtime_options": {
                    "cuda_memory_poll_interval_ms": 1_000,
                    "policy_max_tokens_per_gpu": 12_288,
                },
            }
            self._write_json(
                baseline / "resolved_config.json",
                {
                    **common_resolved,
                    "runtime_options": {
                        **common_resolved["runtime_options"],
                        "skip_unused_old_logprob_entropy": False,
                        "rollout_max_batched_tokens": 16_384,
                    },
                },
            )
            self._write_json(
                candidate / "resolved_config.json",
                {
                    **common_resolved,
                    "runtime_options": {
                        **common_resolved["runtime_options"],
                        "skip_unused_old_logprob_entropy": True,
                        "rollout_max_batched_tokens": 32_768,
                    },
                },
            )
            provenance = {
                "resume_source_checkpoint": "/run/checkpoints/step-000050",
                "resume_forked": True,
                "invocation": {
                    "segment_start_update": 50,
                    "segment_end_update": 51,
                },
            }
            for run in (baseline, candidate):
                self._write_json(run / "provenance.json", provenance)
                self._write_json(
                    run / "training_summary.json",
                    {"status": "paused", "global_update": 51},
                )
                checkpoint = run / "checkpoints" / "step-000051"
                checkpoint.mkdir(parents=True)
                self._write_json(
                    checkpoint / "checkpoint.complete.json",
                    {
                        "global_update": 51,
                        "portable": True,
                        "emergency": False,
                        "runtime_manifest": {
                            "portable": True,
                            "infoskill_modules_included": True,
                        },
                        "files": [
                            "runtime/actor/adapter_model.safetensors",
                            "runtime/actor/lora_optimizer_full.pt",
                            "runtime/actor/lora_scheduler.pt",
                            "runtime/actor/infoskill/infoskill_modules.pt",
                            "runtime/actor/infoskill/infoskill_optimizers.pt",
                            "runtime/actor/infoskill/infoskill_schedulers.pt",
                            "runtime/actor/infoskill/infoskill_rng_state.pt",
                            "trainer_state.json",
                        ],
                    },
                )
            self._write_metric(baseline, core=100.0, entropy_skipped=0.0)
            self._write_metric(candidate, core=80.0, entropy_skipped=1.0)
            parity = {"passed": True, "semantic_exact": True}
            checkpoint = {"passed": True, "all_tensors_exact": True}
            with (
                patch.object(
                    gate,
                    "_read_training_trace",
                    return_value=[{"task_id": "task", "rollout_id": 0}],
                ),
                patch.object(gate, "compare_records", return_value=parity),
                patch.object(gate, "_compare_checkpoints", return_value=checkpoint),
            ):
                report = gate.compare_runs(baseline, candidate)

            behavior_change = {
                "passed": False,
                "semantic_exact": False,
                "semantic_mismatches": ["generation changed"],
            }
            changed_checkpoint = {"passed": False, "all_tensors_exact": False}
            with (
                patch.object(
                    gate,
                    "_read_training_trace",
                    return_value=[{"task_id": "task", "rollout_id": 0}],
                ),
                patch.object(
                    gate,
                    "compare_records",
                    return_value=behavior_change,
                ),
                patch.object(
                    gate,
                    "_compare_checkpoints",
                    return_value=changed_checkpoint,
                ),
            ):
                changed_report = gate.compare_runs(baseline, candidate)

            self._write_json(
                candidate / "resolved_config.json",
                {
                    **common_resolved,
                    "runtime_options": {
                        **common_resolved["runtime_options"],
                        "skip_unused_old_logprob_entropy": False,
                        "rollout_max_batched_tokens": 16_384,
                        "hybrid_prefix_cuda_graph": False,
                        "fuse_kl_ppo_forward": True,
                    },
                },
            )
            self._write_metric(
                candidate,
                core=70.0,
                entropy_skipped=0.0,
                fused_kl_ppo=1.0,
            )
            with (
                patch.object(
                    gate,
                    "_read_training_trace",
                    return_value=[{"task_id": "task", "rollout_id": 0}],
                ),
                patch.object(gate, "compare_records", return_value=parity),
                patch.object(
                    gate,
                    "_compare_checkpoints",
                    return_value=changed_checkpoint,
                ),
            ):
                algorithm_report = gate.compare_runs(
                    baseline,
                    candidate,
                    candidate_mode="fused-kl-ppo",
                )

            self._write_json(
                candidate / "resolved_config.json",
                {
                    **common_resolved,
                    "runtime_options": {
                        **common_resolved["runtime_options"],
                        "skip_unused_old_logprob_entropy": False,
                        "rollout_max_batched_tokens": 16_384,
                        "hybrid_prefix_cuda_graph": False,
                        "fuse_kl_ppo_forward": False,
                        "policy_gradient_clip_mode": "separate",
                    },
                },
            )
            self._write_metric(
                candidate,
                core=70.0,
                entropy_skipped=0.0,
                separate_grad_clip=1.0,
            )
            with (
                patch.object(
                    gate,
                    "_read_training_trace",
                    return_value=[{"task_id": "task", "rollout_id": 0}],
                ),
                patch.object(gate, "compare_records", return_value=parity),
                patch.object(
                    gate,
                    "_compare_checkpoints",
                    return_value=changed_checkpoint,
                ),
            ):
                separate_clip_report = gate.compare_runs(
                    baseline,
                    candidate,
                    candidate_mode="separate-grad-clip",
                )

        self.assertTrue(report["passed"])
        self.assertTrue(report["safe_to_continue_candidate"])
        self.assertEqual(report["core_speedup"], 1.25)
        self.assertEqual(
            report["training_sample_normalized_speedups"]["core"],
            1.25,
        )
        self.assertTrue(changed_report["behavior_change_detected"])
        self.assertTrue(changed_report["efficacy_gate_required"])
        self.assertFalse(changed_report["performance_comparison_valid"])
        self.assertFalse(changed_report["safe_to_continue_candidate"])
        self.assertTrue(algorithm_report["algorithm_change_requested"])
        self.assertTrue(algorithm_report["efficacy_gate_required"])
        self.assertTrue(algorithm_report["performance_comparison_valid"])
        self.assertFalse(algorithm_report["equivalent_optimization_passed"])
        self.assertFalse(algorithm_report["safe_to_continue_candidate"])
        self.assertTrue(separate_clip_report["settings_valid"])
        self.assertTrue(separate_clip_report["algorithm_change_requested"])
        self.assertTrue(separate_clip_report["efficacy_gate_required"])
        self.assertFalse(separate_clip_report["safe_to_continue_candidate"])

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        import json

        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def _write_metric(
        cls,
        run: Path,
        *,
        core: float,
        entropy_skipped: float,
        cuda_graph: float = 0.0,
        fused_kl_ppo: float = 0.0,
        separate_grad_clip: float = 0.0,
    ) -> None:
        cls._write_json(
            run / "metrics.jsonl",
            {
                "phase": "train",
                "step": 51,
                "perf/core_update_seconds": core,
                "perf/rollout_seconds": 40.0,
                "perf/policy_update_seconds": 40.0,
                "perf/old_logprob_seconds": 10.0,
                "perf/old_logprob_entropy_skipped": entropy_skipped,
                "perf/hybrid_prefix_cuda_graph": cuda_graph,
                "perf/fuse_kl_ppo_forward": fused_kl_ppo,
                "policy/separate_gradient_clipping": separate_grad_clip,
                "runtime/training_sample_count": 100.0,
                "rollout/mean_steps": 20.0,
                "perf/cuda/policy_physical_min_free_gb_min": 12.0,
                "perf/cuda/rollout_physical_min_free_gb_min": 11.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
