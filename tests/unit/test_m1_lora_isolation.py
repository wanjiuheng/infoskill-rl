from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path

from infoskill.m1_lora_isolation import (
    build_m1_lora_isolation_report,
    build_m1_isolation_cells,
    collect_m1_isolation_samples,
    summarize_cell_repetitions,
)
from infoskill.rollout import (
    GenerationParameters,
    GenerationRequest,
    GenerationResult,
)


def _probe(index: int) -> GenerationRequest:
    return GenerationRequest(
        request_id=f"probe-{index}",
        task_id=f"task-{index}",
        rollout_id=0,
        env_step=0,
        user_message=f"Find object {index}",
        parameters=GenerationParameters.evaluation(),
        soft_prefix=bytes((index,)),
        seed=index,
    )


def _result(request_id: str, tokens: tuple[int, ...], logprobs: tuple[float, ...]) -> GenerationResult:
    return GenerationResult(
        request_id=request_id,
        text="x",
        finish_reason="length",
        token_ids=tokens,
        token_logprobs=logprobs,
        prompt_token_count=10,
    )


class M1LoRAIsolationTests(unittest.TestCase):
    def test_matrix_changes_only_prefix_or_three_gpu_padding_geometry(self) -> None:
        cells = build_m1_isolation_cells(tuple(_probe(index) for index in range(3)))
        self.assertEqual(
            {cell.name for cell in cells},
            {"hybrid-full3", "hybrid-padded2", "token-full3", "token-padded2"},
        )
        by_name = {cell.name: cell for cell in cells}
        self.assertEqual(
            [request.request_id for request in by_name["hybrid-full3"].requests],
            ["probe-0", "probe-1", "probe-2"],
        )
        self.assertEqual(
            [request.request_id for request in by_name["hybrid-padded2"].requests],
            ["probe-0", "probe-1"],
        )
        self.assertTrue(all(request.soft_prefix is not None for request in by_name["hybrid-full3"].requests))
        self.assertTrue(all(request.soft_prefix is None for request in by_name["token-full3"].requests))
        self.assertEqual(
            [request.user_message for request in by_name["hybrid-full3"].requests],
            [request.user_message for request in by_name["token-full3"].requests],
        )

    def test_cell_summary_detects_pre_branch_logprob_drift(self) -> None:
        rows = (
            (_result("probe-0", (1, 2, 3), (-0.1, -0.2, -0.3)),),
            (_result("probe-0", (1, 2, 4), (-0.11, -0.22, -0.5)),),
            (_result("probe-0", (1, 2, 3), (-0.1, -0.2, -0.3)),),
        )
        summary = summarize_cell_repetitions(rows)

        self.assertFalse(summary["reproducible"])
        self.assertEqual(summary["first_token_divergence"], 2)
        self.assertEqual(summary["changed_logprobs_before_divergence"], 2)
        self.assertEqual(summary["pairwise"]["0-2"]["changed_logprob_count"], 0)

    def test_single_suite_runs_four_runtimes_and_all_cells_continuously(self) -> None:
        created: list[_FakeRuntime] = []

        def factory(graph_enabled: bool) -> _FakeRuntime:
            runtime = _FakeRuntime(graph_enabled)
            created.append(runtime)
            return runtime

        samples = collect_m1_isolation_samples(
            runtime_factory=factory,
            checkpoint_runtime_directory=Path("checkpoint"),
            probes=tuple(_probe(index) for index in range(3)),
        )

        self.assertEqual(
            [sample.label for sample in samples],
            ["checkpoint-eager", "base-eager", "checkpoint-graph", "base-graph"],
        )
        self.assertEqual([len(sample.cells) for sample in samples], [4] * 4)
        self.assertEqual([runtime.generation_calls for runtime in created], [12] * 4)
        self.assertTrue(all(runtime.closed for runtime in created))
        self.assertTrue(all(
            len(cell.generation_rounds) == 3
            for sample in samples
            for cell in sample.cells
        ))
        report = build_m1_lora_isolation_report(samples)
        self.assertEqual(report["classification"], "no_drift_under_fixed_probe")

    def test_report_localizes_hybrid_padding_drift_without_base_drift(self) -> None:
        samples = collect_m1_isolation_samples(
            runtime_factory=lambda graph: _FakeDriftingRuntime(graph),
            checkpoint_runtime_directory=Path("checkpoint"),
            probes=tuple(_probe(index) for index in range(3)),
        )
        report = build_m1_lora_isolation_report(samples)
        self.assertEqual(
            report["classification"],
            "hybrid_prefix_padding_sensitive_lora_drift",
        )
        self.assertEqual(report["drift_cells"]["base-eager"], [])

    def test_boundary_capture_is_scoped_and_has_three_rounds_per_cell(self) -> None:
        runtimes = []
        def factory(graph: bool):
            runtime = _FakeRuntime(graph)
            runtimes.append(runtime)
            return runtime
        samples = collect_m1_isolation_samples(
            runtime_factory=factory,
            checkpoint_runtime_directory=Path("checkpoint"),
            probes=tuple(_probe(index) for index in range(3)),
            capture_boundaries=True,
        )
        self.assertTrue(all(runtime.boundary_starts == 1 and runtime.boundary_ends == 1 for runtime in runtimes))
        self.assertTrue(all(len(cell.boundary_rounds) == 3 for sample in samples for cell in sample.cells))


class _FakeRuntime:
    def __init__(self, graph_enabled: bool) -> None:
        self.graph_enabled = graph_enabled
        self.generation_calls = 0
        self.closed = False
        self.loaded = False
        self.boundary_starts = 0
        self.boundary_ends = 0

    def load_portable_state(self, _directory: Path) -> tuple[dict[str, object], ...]:
        self.loaded = True
        return ({"lora_state_loaded": True, "infoskill_state_loaded": True},)

    def compare_portable_actor_state(self, _directory: Path) -> tuple[dict[str, object], ...]:
        return ({"rank": 0, "exact": True, "actual_summary": {"tensor_payload_sha256": "actor"}},)

    def infoskill_module_snapshot(self) -> tuple[dict[str, object], ...]:
        return ({"rank": 0, "summary": {"tensor_payload_sha256": "m1"}},)

    @contextmanager
    def rollout_session(self):
        yield

    def reset_rollout_prefix_cache(self) -> None:
        return None

    def vllm_lora_snapshot(self) -> tuple[dict[str, object], ...]:
        if self.loaded:
            item = {
                "rank": 0,
                "adapters": {"1": {"summary": {"tensor_payload_sha256": "lora"}}},
                "active_gpu_slots": {"1": 0},
                "active_gpu_slot_summary": {"tensor_payload_sha256": "slot"},
            }
            return tuple({**item, "rank": rank} for rank in range(3))
        item = {
            "rank": 0,
            "adapters": {},
            "registered_adapter_ids": [],
            "active_adapter_ids": [],
            "active_gpu_slots": {},
            "active_gpu_slot_summary": {"partitions": {"lora_b": {"nonzero_count": 0}}},
        }
        return tuple({**item, "rank": rank} for rank in range(3))

    def vllm_base_fingerprint(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {"rank": rank, "summary": {"tensor_payload_sha256": "base"}}
            for rank in range(3)
        )

    def generate(self, requests: tuple[GenerationRequest, ...]) -> tuple[GenerationResult, ...]:
        self.generation_calls += 1
        return tuple(_result(request.request_id, (1,), (-0.1,)) for request in requests)

    def vllm_last_input_fingerprints(self) -> tuple[dict[str, object], ...]:
        return ({"rank": 0, "calls": [{"digest": "stable"}]},)

    def begin_vllm_boundary_capture(self) -> None:
        self.boundary_starts += 1

    def take_vllm_boundary_rows(self) -> tuple[dict[str, object], ...]:
        return ({"rank": 0, "rows": [{
            "rank": 0,
            "history_sha256": "fixed",
            "raw": {"top_values": [1.0], "logsumexp": 1.2},
            "processed": {"top_values": [1.0], "logsumexp": 1.2},
            "sampled_token_id": 1,
            "returned_logprob": -0.1,
        }]},)

    def end_vllm_boundary_capture(self) -> None:
        self.boundary_ends += 1

    def close(self) -> None:
        self.closed = True


class _FakeDriftingRuntime(_FakeRuntime):
    def generate(self, requests: tuple[GenerationRequest, ...]) -> tuple[GenerationResult, ...]:
        self.generation_calls += 1
        drift = (
            self.loaded
            and len(requests) == 2
            and requests[0].soft_prefix is not None
            and self.generation_calls % 2 == 0
        )
        logprob = -0.2 if drift else -0.1
        return tuple(_result(request.request_id, (1,), (logprob,)) for request in requests)


if __name__ == "__main__":
    unittest.main()
