from __future__ import annotations

from dataclasses import replace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

from infoskill.m1_reproducibility import (
    RuntimeReproducibilitySample,
    build_hybrid_prefix_reproducibility_probes,
    build_m1_reproducibility_report,
)
from infoskill.rollout import GenerationResult


def _result(token: int, logprob: float = -0.1) -> GenerationResult:
    return GenerationResult(
        request_id="checkpoint-effect/pick-place",
        text=str(token),
        finish_reason="stop",
        token_ids=(token,),
        token_logprobs=(logprob,),
        prompt_token_count=10,
    )


def _rank_summary(rank: int, digest: str) -> dict[str, object]:
    return {
        "rank": rank,
        "selected_tensor_names": ["model.norm.weight"],
        "summary": {"tensor_payload_sha256": digest},
    }


def _actor(rank: int, digest: str) -> dict[str, object]:
    return {
        "rank": rank,
        "exact": True,
        "actual_summary": {"tensor_payload_sha256": digest},
    }


def _vllm_lora(
    rank: int,
    digest: str,
    adapter_id: int,
    *,
    slot_digest: str | None = None,
) -> dict[str, object]:
    return {
        "rank": rank,
        "active_gpu_slots": {str(adapter_id): 0},
        "active_gpu_slot_summary": {
            "tensor_payload_sha256": slot_digest or digest,
            "partitions": {"lora_b": {"nonzero_count": 0}},
        },
        "adapters": {
            str(adapter_id): {
                "summary": {
                    "tensor_payload_sha256": digest,
                    "partitions": {"lora_b": {"nonzero_count": 0}},
                },
            }
        },
    }


def _vllm_without_lora(rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "registered_adapter_ids": [],
        "active_adapter_ids": [],
        "active_gpu_slots": {},
        "active_gpu_slot_summary": {
            "partitions": {"lora_b": {"nonzero_count": 0}},
        },
        "adapters": {},
    }


def _sample(
    label: str,
    *,
    token: int = 1,
    lora_digest: str = "vllm",
    slot_digest: str | None = None,
    second_token: int | None = None,
) -> RuntimeReproducibilitySample:
    checkpoint = label.startswith("checkpoint")
    adapter_id = 1 if label.endswith("a") else 99
    lora = tuple(
        _vllm_lora(
            rank,
            lora_digest,
            adapter_id,
            slot_digest=slot_digest,
        )
        for rank in range(2)
    )
    base = tuple(_rank_summary(rank, "base") for rank in range(2))
    return RuntimeReproducibilitySample(
        label=label,
        checkpoint_loaded=checkpoint,
        load_reports=(
            ({
                "rank": 0,
                "lora_state_loaded": True,
                "infoskill_state_loaded": True,
            },)
            if checkpoint
            else ()
        ),
        actor_snapshots=(
            tuple(_actor(rank, "actor") for rank in range(2)) if checkpoint else ()
        ),
        infoskill_snapshots=(
            tuple(_rank_summary(rank, "m1") for rank in range(2)) if checkpoint else ()
        ),
        vllm_lora_snapshots=(lora, lora, lora),
        vllm_base_snapshots=(base, base, base),
        generation_rounds=(
            (_result(token),),
            (_result(token if second_token is None else second_token),),
        ),
    )


class M1ReproducibilityTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None, "requires torch")
    def test_hybrid_prefix_probes_are_exact_and_deterministic(self) -> None:
        first = build_hybrid_prefix_reproducibility_probes(
            hidden_size=8,
            prefix_length=5,
            master_seed=7,
            max_new_tokens=4,
            case_count=2,
        )
        second = build_hybrid_prefix_reproducibility_probes(
            hidden_size=8,
            prefix_length=5,
            master_seed=7,
            max_new_tokens=4,
            case_count=2,
        )

        self.assertEqual(len(first), 2)
        self.assertTrue(torch.equal(first[0].soft_prefix, second[0].soft_prefix))
        self.assertEqual(tuple(first[0].soft_prefix.shape), (5, 8))
        self.assertEqual(first[0].soft_prefix.dtype, torch.bfloat16)

    def test_report_accepts_exact_reproduction_despite_adapter_id_changes(self) -> None:
        report = build_m1_reproducibility_report(
            tuple(_sample(label) for label in (
                "checkpoint-a",
                "checkpoint-b",
                "base-a",
                "base-b",
            ))
        )

        self.assertTrue(report["reproducible"])
        self.assertEqual(report["classification"], "reproducible_under_probe")

    def test_report_locates_fsdp_to_vllm_lora_hash_drift(self) -> None:
        report = build_m1_reproducibility_report(
            (
                _sample("checkpoint-a", lora_digest="one"),
                _sample("checkpoint-b", lora_digest="two"),
                _sample("base-a"),
                _sample("base-b"),
            )
        )

        self.assertEqual(
            report["classification"],
            "fsdp_to_vllm_registry_sync_nondeterminism",
        )

    def test_report_locates_vllm_active_gpu_slot_drift(self) -> None:
        report = build_m1_reproducibility_report(
            (
                _sample("checkpoint-a", slot_digest="one"),
                _sample("checkpoint-b", slot_digest="two"),
                _sample("base-a"),
                _sample("base-b"),
            )
        )

        self.assertEqual(
            report["classification"],
            "vllm_registry_to_gpu_slot_sync_nondeterminism",
        )

    def test_report_locates_lora_execution_drift_after_exact_hashes(self) -> None:
        report = build_m1_reproducibility_report(
            (
                _sample("checkpoint-a", token=1),
                _sample("checkpoint-b", token=2),
                _sample("base-a", token=3),
                _sample("base-b", token=3),
            )
        )

        self.assertEqual(
            report["classification"],
            "vllm_lora_execution_nondeterminism",
        )

    def test_absent_base_lora_does_not_mask_checkpoint_execution_drift(self) -> None:
        empty = tuple(_vllm_without_lora(rank) for rank in range(2))
        base_a = replace(
            _sample("base-a", token=3),
            vllm_lora_snapshots=(empty, empty, empty),
        )
        base_b = replace(
            _sample("base-b", token=3),
            vllm_lora_snapshots=(empty, empty, empty),
        )
        report = build_m1_reproducibility_report(
            (
                _sample("checkpoint-a", token=1),
                _sample("checkpoint-b", token=2),
                base_a,
                base_b,
            )
        )

        self.assertTrue(report["checks"]["base_controls_have_zero_lora_b"])
        self.assertEqual(
            report["classification"],
            "vllm_lora_execution_nondeterminism",
        )

    def test_absent_base_adapter_with_active_gpu_slot_is_not_zero_control(self) -> None:
        empty = tuple(_vllm_without_lora(rank) for rank in range(2))
        stale = dict(_vllm_without_lora(0))
        stale["active_gpu_slots"] = {"42": 0}
        base_a = replace(
            _sample("base-a", token=3),
            vllm_lora_snapshots=((stale, empty[1]),) * 3,
        )
        base_b = replace(
            _sample("base-b", token=3),
            vllm_lora_snapshots=(empty, empty, empty),
        )
        report = build_m1_reproducibility_report(
            (
                _sample("checkpoint-a", token=1),
                _sample("checkpoint-b", token=2),
                base_a,
                base_b,
            )
        )

        self.assertFalse(report["checks"]["base_controls_have_zero_lora_b"])
        self.assertEqual(report["classification"], "base_control_not_lora_zero")


if __name__ == "__main__":
    unittest.main()
