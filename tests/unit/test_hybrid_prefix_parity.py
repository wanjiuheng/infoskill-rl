from __future__ import annotations

import unittest

from scripts.hybrid_prefix_parity import (
    build_case_specs,
    build_vllm_request_plan,
    summarize_parity,
)


def _comparison(error: float, *, token_match: bool = True) -> dict[str, object]:
    left_token = 17
    right_token = left_token if token_match else 23
    return {
        "left": {"token_id": left_token, "logprob": -1.0},
        "right": {"token_id": right_token, "logprob": -1.0 - error},
    }


class HybridPrefixParityTests(unittest.TestCase):
    def test_case_plan_is_deterministic_and_covers_multiple_prompts(self) -> None:
        first = build_case_specs(case_count=8, base_seed=20260904)
        second = build_case_specs(case_count=8, base_seed=20260904)

        self.assertEqual(first, second)
        self.assertEqual(
            [case["prefix_seed"] for case in first],
            [20260904] * 4 + [20260905] * 4,
        )
        self.assertEqual(len({case["prompt"] for case in first}), 4)
        seeds_by_prompt = {
            prompt: {
                case["prefix_seed"] for case in first if case["prompt"] == prompt
            }
            for prompt in {case["prompt"] for case in first}
        }
        self.assertTrue(
            all(seeds == {20260904, 20260905} for seeds in seeds_by_prompt.values())
        )

    def test_vllm_plan_pairs_plain_and_hybrid_transport_requests(self) -> None:
        cases = [
            {
                "case_id": "case-00",
                "text_ids": [3, 4],
                "prefix_token_ids": [10, 11],
                "token_prefix": ["embedding-10", "embedding-11"],
                "random_prefix": ["random-0", "random-1"],
            }
        ]

        plan = build_vllm_request_plan(cases, placeholder_id=0)

        self.assertEqual(
            plan["transport_plain"][0]["prompt"],
            {"prompt_token_ids": [10, 11, 3, 4]},
        )
        self.assertEqual(
            plan["transport_hybrid"][0]["prompt"],
            {
                "prompt_token_ids": [0, 0, 3, 4],
                "infoskill_prefix_embeds": ["embedding-10", "embedding-11"],
                "infoskill_prefix_mask": [True, True, False, False],
            },
        )
        self.assertEqual(len(plan["transport_plain"]), 1)
        self.assertEqual(len(plan["transport_hybrid"]), 1)
        self.assertEqual(
            plan["cross_backend"][0]["prompt"]["infoskill_prefix_embeds"],
            ["random-0", "random-1"],
        )

    def test_dual_gate_report_passes_only_when_both_gates_pass(self) -> None:
        transport = [_comparison(0.00001), _comparison(0.00002)]
        cross_backend = [
            _comparison(0.01),
            _comparison(0.02),
            _comparison(0.03),
            _comparison(0.04),
        ]

        report = summarize_parity(
            transport,
            cross_backend,
            transport_logprob_atol=0.0001,
            cross_p95_logprob_atol=0.05,
            cross_max_logprob_atol=0.10,
            required_token_match_rate=1.0,
        )

        self.assertEqual(report["schema_version"], 2)
        self.assertTrue(report["transport_gate"]["passed"])
        self.assertTrue(report["cross_backend_gate"]["passed"])
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(
            report["cross_backend_gate"]["summary"]["median_logprob_abs_error"],
            0.025,
        )

    def test_transport_failure_cannot_be_hidden_by_cross_backend_pass(self) -> None:
        report = summarize_parity(
            [_comparison(0.001)],
            [_comparison(0.01)],
            transport_logprob_atol=0.0001,
            cross_p95_logprob_atol=0.05,
            cross_max_logprob_atol=0.10,
            required_token_match_rate=1.0,
        )

        self.assertFalse(report["transport_gate"]["passed"])
        self.assertTrue(report["cross_backend_gate"]["passed"])
        self.assertFalse(report["passed"])

    def test_token_mismatch_fails_even_when_available_logprobs_are_close(self) -> None:
        report = summarize_parity(
            [_comparison(0.0)],
            [_comparison(0.001, token_match=False)],
            transport_logprob_atol=0.0001,
            cross_p95_logprob_atol=0.05,
            cross_max_logprob_atol=0.10,
            required_token_match_rate=1.0,
        )

        summary = report["cross_backend_gate"]["summary"]
        self.assertEqual(summary["token_match_rate"], 0.0)
        self.assertIsNone(summary["max_logprob_abs_error"])
        self.assertFalse(report["cross_backend_gate"]["passed"])

    def test_explicit_match_rate_override_controls_token_mismatch_policy(self) -> None:
        report = summarize_parity(
            [_comparison(0.0)],
            [_comparison(0.01), _comparison(0.0, token_match=False)],
            transport_logprob_atol=0.0001,
            cross_p95_logprob_atol=0.05,
            cross_max_logprob_atol=0.10,
            required_token_match_rate=0.5,
        )

        self.assertTrue(report["cross_backend_gate"]["passed"])

    def test_non_finite_logprob_error_fails_gate(self) -> None:
        report = summarize_parity(
            [_comparison(float("nan"))],
            [_comparison(0.01)],
            transport_logprob_atol=0.0001,
            cross_p95_logprob_atol=0.05,
            cross_max_logprob_atol=0.10,
            required_token_match_rate=1.0,
        )

        summary = report["transport_gate"]["summary"]
        self.assertEqual(summary["token_match_count"], 1)
        self.assertEqual(summary["finite_error_count"], 0)
        self.assertIsNone(summary["max_logprob_abs_error"])
        self.assertFalse(report["transport_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
