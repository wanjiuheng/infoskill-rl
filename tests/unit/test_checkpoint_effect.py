from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

from infoskill.checkpoint_effect import (
    build_checkpoint_effect_probes,
    classify_checkpoint_effect,
    collect_independent_checkpoint_effect_samples,
    compare_generation_results,
    compare_named_tensors,
    compare_vllm_to_checkpoint_aggregate,
    flatten_lora_model_tensors,
    summarize_named_tensors,
)
from infoskill.rollout import GenerationResult


def _result(request_id: str, tokens: tuple[int, ...], logprobs: tuple[float, ...]):
    return GenerationResult(
        request_id=request_id,
        text="decoded " + " ".join(map(str, tokens)),
        finish_reason="stop",
        token_ids=tokens,
        token_logprobs=logprobs,
        prompt_token_count=10,
    )


class CheckpointEffectTests(unittest.TestCase):
    def test_probe_requests_are_deterministic_alfworld_messages(self) -> None:
        first = build_checkpoint_effect_probes(master_seed=7)
        second = build_checkpoint_effect_probes(master_seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(not item.parameters.do_sample for item in first))
        self.assertIn("admissible actions", first[0].user_message)

    @unittest.skipUnless(torch is not None, "requires torch")
    def test_tensor_comparison_identifies_exact_and_changed_weights(self) -> None:
        expected = {
            "layer.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "layer.lora_B.weight": torch.tensor([[0.0], [3.0]]),
        }
        exact = compare_named_tensors(expected, {key: value.clone() for key, value in expected.items()})
        changed = {key: value.clone() for key, value in expected.items()}
        changed["layer.lora_B.weight"][1, 0] = 4.0

        self.assertTrue(exact["exact"])
        mismatch = compare_named_tensors(expected, changed)
        self.assertFalse(mismatch["exact"])
        self.assertEqual(mismatch["unequal_tensor_count"], 1)
        self.assertEqual(mismatch["max_abs_error"], 1.0)

    @unittest.skipUnless(torch is not None, "requires torch")
    def test_tensor_summary_separates_lora_a_and_b(self) -> None:
        summary = summarize_named_tensors(
            {
                "x.lora_a": torch.ones(2),
                "x.lora_b": torch.tensor([0.0, 2.0]),
            }
        )

        partitions = summary["partitions"]
        self.assertEqual(partitions["lora_a"]["nonzero_count"], 2)
        self.assertEqual(partitions["lora_b"]["nonzero_count"], 1)

    def test_generation_comparison_detects_logit_effect_without_token_change(self) -> None:
        baseline = (_result("a", (1, 2), (-0.1, -0.2)),)
        checkpoint = (_result("a", (1, 2), (-0.1, -0.3)),)

        report = compare_generation_results(baseline, checkpoint)

        self.assertTrue(report["tokens_exact"])
        self.assertTrue(report["effect_visible"])
        self.assertEqual(report["changed_logprob_count"], 1)

    def test_classification_locates_actor_then_vllm_then_precision(self) -> None:
        valid_vllm = {
            "active_adapter_ids": [1],
            "adapters": {
                "1": {
                    "summary": {
                        "partitions": {"lora_b": {"nonzero_count": 1}}
                    }
                }
            }
        }
        actor_failure = classify_checkpoint_effect(
            actor_snapshots=({"exact": False},),
            vllm_snapshots=(valid_vllm,),
            vllm_checkpoint_comparison={"all_ranks_match": True},
            generation_comparison={"effect_visible": False},
        )
        vllm_failure = classify_checkpoint_effect(
            actor_snapshots=({"exact": True},),
            vllm_snapshots=({"adapters": {}},),
            vllm_checkpoint_comparison={"all_ranks_match": False},
            generation_comparison={"effect_visible": False},
        )
        below_precision = classify_checkpoint_effect(
            actor_snapshots=({"exact": True},),
            vllm_snapshots=(valid_vllm,),
            vllm_checkpoint_comparison={"all_ranks_match": True},
            generation_comparison={"effect_visible": False},
        )

        self.assertEqual(actor_failure["classification"], "checkpoint_to_fsdp_mismatch")
        self.assertEqual(vllm_failure["classification"], "fsdp_to_vllm_missing_or_zero")
        self.assertEqual(
            below_precision["classification"],
            "checkpoint_effect_below_probe_precision",
        )

    def test_vllm_aggregate_comparison_accounts_for_b_scaling(self) -> None:
        actor = ({
            "rank": 0,
            "expected_summary": {
                "partitions": {
                    "lora_a": {
                        "tensor_count": 1,
                        "element_count": 2,
                        "nonzero_count": 2,
                        "max_abs": 3.0,
                        "l2_squared": 10.0,
                    },
                    "lora_b": {
                        "tensor_count": 1,
                        "element_count": 2,
                        "nonzero_count": 2,
                        "max_abs": 4.0,
                        "l2_squared": 20.0,
                    },
                }
            },
        },)
        vllm = ({
            "rank": 0,
            "adapters": {
                "9": {
                    "summary": {
                        "partitions": {
                            "lora_a": {
                                "tensor_count": 1,
                                "element_count": 2,
                                "nonzero_count": 2,
                                "max_abs": 3.0,
                                "l2_squared": 10.0,
                            },
                            "lora_b": {
                                "tensor_count": 1,
                                "element_count": 2,
                                "nonzero_count": 2,
                                "max_abs": 8.0,
                                "l2_squared": 80.0,
                            },
                        }
                    }
                }
            },
        },)

        report = compare_vllm_to_checkpoint_aggregate(
            actor,
            vllm,
            lora_scaling=2.0,
        )

        self.assertTrue(report["all_ranks_match"])

    def test_observed_bf16_quantization_is_within_default_aggregate_gate(self) -> None:
        actor = ({
            "rank": 0,
            "expected_summary": {
                "partitions": {
                    "lora_a": {
                        "tensor_count": 196,
                        "element_count": 18120704,
                        "nonzero_count": 18120704,
                        "max_abs": 0.016795914620161057,
                        "l2_squared": 1045.403272151947,
                    },
                    "lora_b": {
                        "tensor_count": 196,
                        "element_count": 22249472,
                        "nonzero_count": 22249472,
                        "max_abs": 9.853662049863487e-05,
                        "l2_squared": 0.008097046680063613,
                    },
                }
            },
        },)
        vllm = ({
            "rank": 0,
            "adapters": {
                "1": {
                    "summary": {
                        "partitions": {
                            "lora_a": {
                                "tensor_count": 196,
                                "element_count": 18120704,
                                "nonzero_count": 18120704,
                                "max_abs": 0.016845703125,
                                "l2_squared": 1045.3974795341492,
                            },
                            "lora_b": {
                                "tensor_count": 196,
                                "element_count": 22249472,
                                "nonzero_count": 22249472,
                                "max_abs": 0.00019741058349609375,
                                "l2_squared": 0.032388119302140694,
                            },
                        }
                    }
                }
            },
        },)

        report = compare_vllm_to_checkpoint_aggregate(
            actor,
            vllm,
            lora_scaling=2.0,
        )

        self.assertTrue(report["all_ranks_match"])

    def test_checkpoint_runtime_loads_before_its_first_rollout(self) -> None:
        events: list[str] = []

        class Runtime:
            def __init__(self, label: str) -> None:
                self.label = label

            def load_portable_state(self, path: Path) -> None:
                events.append(f"{self.label}:load:{path.name}")

            def compare_portable_actor_state(self, path: Path):
                events.append(f"{self.label}:compare:{path.name}")
                return ({"rank": 0, "exact": True},)

            @contextmanager
            def rollout_session(self):
                events.append(f"{self.label}:session-enter")
                try:
                    yield
                finally:
                    events.append(f"{self.label}:session-exit")

            def reset_rollout_prefix_cache(self) -> None:
                events.append(f"{self.label}:reset")

            def vllm_lora_snapshot(self):
                events.append(f"{self.label}:snapshot")
                return ({"rank": 0},)

            def generate(self, probes):
                events.append(f"{self.label}:generate")
                return (_result("a", (1,), (-0.1,)),)

            def close(self) -> None:
                events.append(f"{self.label}:close")

        labels = iter(("checkpoint", "baseline"))
        collect_independent_checkpoint_effect_samples(
            runtime_factory=lambda: Runtime(next(labels)),
            checkpoint_runtime_directory=Path("runtime"),
            probes=build_checkpoint_effect_probes(master_seed=0),
        )

        self.assertLess(
            events.index("checkpoint:load:runtime"),
            events.index("checkpoint:session-enter"),
        )
        self.assertGreater(
            events.index("baseline:session-enter"),
            events.index("checkpoint:close"),
        )

    @unittest.skipUnless(torch is not None, "requires torch")
    def test_packed_vllm_lora_tensors_are_flattened(self) -> None:
        class Layer:
            lora_a = [torch.ones(1), None, torch.ones(2)]
            lora_b = [torch.ones(3), None, torch.ones(4)]

        class Adapter:
            loras = {"qkv": Layer()}

        flattened = flatten_lora_model_tensors(Adapter())

        self.assertEqual(
            set(flattened),
            {"qkv.lora_a.0", "qkv.lora_a.2", "qkv.lora_b.0", "qkv.lora_b.2"},
        )


if __name__ == "__main__":
    unittest.main()
