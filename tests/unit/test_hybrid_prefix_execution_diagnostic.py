from __future__ import annotations

import unittest

from scripts.hybrid_prefix_execution_diagnostic import (
    build_execution_rounds,
    compare_execution_reports,
)


def _report(mode: str, token_offset: int = 0) -> dict[str, object]:
    rounds = {}
    for label in ("prefix-a", "prefix-b", "prefix-a-repeat", "prefix-a-reverse"):
        order = ("case-01", "case-00") if label.endswith("reverse") else (
            "case-00",
            "case-01",
        )
        rounds[label] = {
            case_id: {
                "token_ids": [11 + token_offset, 12],
                "token_logprobs": [-0.1, -0.2],
            }
            for case_id in order
        }
    return {"schema_version": 1, "execution_mode": mode, "rounds": rounds}


class HybridPrefixExecutionDiagnosticTests(unittest.TestCase):
    def test_rounds_repeat_prefix_and_reverse_request_order(self) -> None:
        cases = [
            {"case_id": "case-00", "prompt_token_ids": [1], "prefix_a": "a0", "prefix_b": "b0"},
            {"case_id": "case-01", "prompt_token_ids": [2], "prefix_a": "a1", "prefix_b": "b1"},
        ]

        rounds = build_execution_rounds(cases, placeholder_id=0)

        self.assertEqual(list(rounds), [
            "prefix-a",
            "prefix-b",
            "prefix-a-repeat",
            "prefix-a-reverse",
        ])
        self.assertEqual(
            [request["case_id"] for request in rounds["prefix-a-reverse"]],
            ["case-01", "case-00"],
        )
        self.assertEqual(
            rounds["prefix-a"][0]["prompt"]["infoskill_prefix_embeds"],
            "a0",
        )
        self.assertEqual(
            rounds["prefix-b"][0]["prompt"]["infoskill_prefix_embeds"],
            "b0",
        )

    def test_comparison_identifies_cuda_graph_as_first_divergent_layer(self) -> None:
        eager = _report("eager")
        persistent = _report("persistent-eager")
        compile_only = _report("compile-only")
        cuda_graph = _report("cuda-graph", token_offset=1)

        result = compare_execution_reports(
            [eager, persistent, compile_only, cuda_graph],
            logprob_atol=1e-3,
        )

        self.assertEqual(result["classification"], "cuda_graph_replay_divergence")
        self.assertTrue(result["within_mode_checks"]["cuda-graph"]["repeat_exact"])
        self.assertFalse(result["against_eager"]["cuda-graph"]["tokens_exact"])
        self.assertFalse(result["passed"])

    def test_comparison_passes_when_every_layer_matches(self) -> None:
        reports = [
            _report("eager"),
            _report("persistent-eager"),
            _report("compile-only"),
            _report("cuda-graph"),
        ]

        result = compare_execution_reports(reports, logprob_atol=1e-3)

        self.assertEqual(result["classification"], "all_execution_modes_match")
        self.assertTrue(result["passed"])
        self.assertEqual(result["against_eager"]["cuda-graph"]["sequence_count"], 8)


if __name__ == "__main__":
    unittest.main()
