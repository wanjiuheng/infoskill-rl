from __future__ import annotations

import unittest

from scripts.compare_m1_fresh_runtime_matrix import compare_matrix


def _report(
    *,
    graph: bool,
    split_k_one: bool,
    intervention: str,
    checkpoint_exact: bool = True,
    base_exact: bool = True,
) -> dict[str, object]:
    checks = {
        "input_fingerprints_exact": True,
        "checkpoint_load_reports_complete": True,
        "base_controls_have_zero_lora_b": True,
        "actor_matches_checkpoint_on_all_ranks": True,
        "actor_lora_exact_across_runtimes": True,
        "infoskill_modules_exact_across_runtimes": True,
        "vllm_registered_lora_exact_across_runtimes": True,
        "vllm_active_gpu_slots_exact_across_runtimes": True,
        "vllm_registered_lora_immutable_within_runtimes": True,
        "vllm_active_gpu_slots_immutable_within_runtimes": True,
        "checkpoint_vllm_base_fingerprint_exact_across_runtimes": True,
        "base_control_vllm_base_fingerprint_exact_across_runtimes": True,
        "split_k_one_verified_for_all_runtimes": split_k_one,
        "checkpoint_a_within_runtime": checkpoint_exact,
        "checkpoint_b_within_runtime": checkpoint_exact,
        "checkpoint_across_runtimes": checkpoint_exact,
        "base_a_within_runtime": base_exact,
        "base_b_within_runtime": base_exact,
        "base_across_runtimes": base_exact,
    }
    return {
        "classification": "fixture",
        "checkpoint": {
            "directory": "/checkpoint",
            "provenance_sha256": "checkpoint-sha",
        },
        "probe": {"request_ids": ["one", "two", "three"]},
        "execution": {
            "hybrid_prefix_cuda_graph": graph,
            "lora_shrink_split_k_one": split_k_one,
            "lora_kernel_intervention": intervention,
        },
        "checks": checks,
    }


class M1FreshRuntimeMatrixTests(unittest.TestCase):
    def test_matrix_accepts_exact_split_k_one_across_graph_and_eager(self) -> None:
        report = compare_matrix(
            _report(graph=True, split_k_one=True, intervention="none"),
            _report(graph=False, split_k_one=True, intervention="none"),
            _report(
                graph=False,
                split_k_one=False,
                intervention="reference_full",
            ),
        )

        self.assertTrue(report["controls_valid"])
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["classification"],
            "split_k_one_reproducible_across_fresh_runtimes",
        )

    def test_matrix_attributes_graph_independent_drift_to_native_lora_kernels(self) -> None:
        report = compare_matrix(
            _report(
                graph=True,
                split_k_one=True,
                intervention="none",
                checkpoint_exact=False,
            ),
            _report(
                graph=False,
                split_k_one=True,
                intervention="none",
                checkpoint_exact=False,
            ),
            _report(
                graph=False,
                split_k_one=False,
                intervention="reference_full",
                checkpoint_exact=True,
            ),
        )

        self.assertTrue(report["controls_valid"])
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["classification"],
            "native_lora_kernel_nondeterminism_after_split_k_one",
        )

    def test_matrix_attributes_graph_only_drift(self) -> None:
        report = compare_matrix(
            _report(
                graph=True,
                split_k_one=True,
                intervention="none",
                checkpoint_exact=False,
            ),
            _report(graph=False, split_k_one=True, intervention="none"),
            _report(
                graph=False,
                split_k_one=False,
                intervention="reference_full",
            ),
        )

        self.assertEqual(
            report["classification"],
            "cuda_graph_specific_checkpoint_nondeterminism",
        )

    def test_matrix_rejects_invalid_base_control(self) -> None:
        graph = _report(graph=True, split_k_one=True, intervention="none")
        eager = _report(graph=False, split_k_one=True, intervention="none")
        reference = _report(
            graph=False,
            split_k_one=False,
            intervention="reference_full",
        )
        reference["checks"]["base_controls_have_zero_lora_b"] = False

        report = compare_matrix(graph, eager, reference)

        self.assertFalse(report["controls_valid"])
        self.assertEqual(
            report["classification"],
            "invalid_fresh_runtime_matrix_controls",
        )


if __name__ == "__main__":
    unittest.main()
