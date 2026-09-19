from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compare_m0_lora_lr_sweep import compare


class M0LoraLearningRateSweepTests(unittest.TestCase):
    def test_valid_three_way_pair_verifies_effective_lr_and_paired_flips(self) -> None:
        source = Path("source/checkpoints/step-000100").resolve()
        trains = [Path(f"train-{index}").resolve() for index in range(3)]
        evaluations = [Path(f"eval-{index}").resolve() for index in range(3)]
        rates = (1e-6, 3e-6, 1e-5)

        def read_json(path: Path) -> dict:
            if path == source / "checkpoint.complete.json":
                return {"global_update": 100}
            if path.parent in trains:
                index = trains.index(path.parent)
                if path.name == "resolved_config.json":
                    return {
                        "mode": "no_skill",
                        "runtime_options": {
                            "actor_learning_rate": rates[index],
                            "persistent_rollout_session": True,
                        },
                    }
                if path.name == "provenance.json":
                    return {
                        "resume_forked": True,
                        "resume_source_checkpoint": str(source),
                        "invocation": {
                            "segment_start_update": 100,
                            "segment_end_update": 105,
                            "actor_learning_rate": rates[index],
                        },
                    }
                if path.name == "training_summary.json":
                    return {"status": "paused", "global_update": 105}
            if (
                path.name == "checkpoint.complete.json"
                and path.parent.name == "step-000105"
                and path.parent.parent.parent in trains
            ):
                return {
                    "portable": True,
                    "emergency": False,
                    "global_update": 105,
                }
            if path.parent in evaluations:
                index = evaluations.index(path.parent)
                if path.name == "valid_seen_summary.json":
                    successes = (2, 3, 1)[index]
                    return {
                        "is_complete": True,
                        "evaluated": 4,
                        "task_manifest_sha256": "same",
                        "macro_success": successes / 4,
                        "overall_success": successes / 4,
                        "invalid_action_rate": 0.1,
                        "mean_steps": 20.0,
                        "per_task_type_success": {"task": successes / 4},
                    }
                if path.name == "provenance.json":
                    return {
                        "evaluation_runtime": {
                            "backend": "verl",
                            "num_gpus": 3,
                            "eval_batch_size": 64,
                            "environment_backend": "native_batch",
                            "persistent_rollout_session": True,
                            "hybrid_prefix_cuda_graph": False,
                        }
                    }
                if path.name == "checkpoint-load.json":
                    return {
                        "status": "loaded",
                        "checkpoint_step": 105,
                        "checkpoint": str(
                            trains[index] / "checkpoints" / "step-000105"
                        ),
                    }
            raise AssertionError(f"unexpected JSON read: {path}")

        def train_rows(run: Path, _start: int, _end: int) -> dict[int, dict]:
            rate = rates[trains.index(run)]
            return {
                step: {
                    "actor/lr": rate,
                    "rollout/success_rate": 0.25,
                    "rollout/mean_reward": 0.2,
                    "rollout/invalid_action_rate": 0.1,
                    "actor/ppo_kl": 0.01,
                    "actor/pg_clipfrac": 0.02,
                    "actor/grad_norm": 0.3,
                    "perf/core_update_seconds": 10.0,
                }
                for step in range(101, 106)
            }

        task_results = [
            {"a": True, "b": True, "c": False, "d": False},
            {"a": True, "b": True, "c": True, "d": False},
            {"a": True, "b": False, "c": False, "d": False},
        ]
        workload = [
            {"task_id": f"task-{index // 8}", "rollout_id": index}
            for index in range(64)
        ]
        with (
            patch(
                "scripts.compare_m0_lora_lr_sweep._read_json",
                side_effect=read_json,
            ),
            patch(
                "scripts.compare_m0_lora_lr_sweep._train_rows",
                side_effect=train_rows,
            ),
            patch(
                "scripts.compare_m0_lora_lr_sweep._evaluation_tasks",
                side_effect=task_results,
            ),
            patch(
                "scripts.compare_m0_lora_lr_sweep._read_training_trace",
                return_value=workload,
            ),
        ):
            report = compare(
                source,
                trains,
                evaluations,
                expected_task_count=4,
            )

        self.assertTrue(report["controls_valid"])
        self.assertEqual(report["classification"], "paired_lr_screen_complete")
        self.assertEqual(report["ranking_macro_then_overall"][0], "3e-06")
        self.assertEqual(report["branches"]["3e-06"]["paired_vs_1e-6"]["net_gain"], 1)
        self.assertEqual(report["branches"]["1e-05"]["paired_vs_1e-6"]["net_gain"], -1)


if __name__ == "__main__":
    unittest.main()
