from __future__ import annotations

import unittest

from infoskill.integrations.verl.memory_metrics import (
    PhysicalMemorySampler,
    summarize_cuda_memory_snapshots,
    summarize_rank_token_load,
)


_GIB = 1024**3


def _stage(
    *,
    allocated: int,
    reserved: int,
    peak: int,
    free: int,
    physical_min_free: int | None = None,
) -> dict[str, int]:
    stage = {
        "allocated_bytes": allocated * _GIB,
        "reserved_bytes": reserved * _GIB,
        "peak_allocated_bytes": peak * _GIB,
        "peak_reserved_bytes": (peak + 1) * _GIB,
        "free_bytes": free * _GIB,
        "total_bytes": 80 * _GIB,
    }
    if physical_min_free is not None:
        stage["physical_min_free_bytes"] = physical_min_free * _GIB
        stage["physical_sample_count"] = 100
    return stage


class VerlMemoryMetricsTests(unittest.TestCase):
    def test_physical_sampler_reports_minimum_across_start_and_stop(self) -> None:
        readings = iter(((70, 80), (60, 80)))
        sampler = PhysicalMemorySampler(
            read_memory=lambda: next(readings),
            interval_ms=10_000,
        )

        sampler.start()
        result = sampler.stop()

        self.assertEqual(result["physical_min_free_bytes"], 60)
        self.assertEqual(result["physical_total_bytes"], 80)
        self.assertEqual(result["physical_sample_count"], 2)

    def test_preserves_each_rank_and_reports_conservative_extrema(self) -> None:
        metrics = summarize_cuda_memory_snapshots(
            [
                {
                    "rank": 0,
                    "rollout": _stage(allocated=20, reserved=22, peak=24, free=55),
                    "policy": _stage(
                        allocated=60,
                        reserved=64,
                        peak=68,
                        free=12,
                        physical_min_free=8,
                    ),
                },
                {
                    "rank": 1,
                    "rollout": _stage(allocated=21, reserved=23, peak=25, free=54),
                    "policy": _stage(
                        allocated=61,
                        reserved=65,
                        peak=70,
                        free=10,
                        physical_min_free=6,
                    ),
                },
            ]
        )

        self.assertEqual(metrics["perf/cuda/rank_count"], 2.0)
        self.assertEqual(
            metrics["perf/cuda/rank_0/policy_peak_allocated_gb"], 68.0
        )
        self.assertEqual(
            metrics["perf/cuda/rank_1/policy_peak_allocated_gb"], 70.0
        )
        self.assertEqual(
            metrics["perf/cuda/policy_peak_allocated_gb_max"], 70.0
        )
        self.assertEqual(metrics["perf/cuda/policy_free_gb_min"], 10.0)
        self.assertEqual(metrics["perf/cuda/policy_device_used_gb_max"], 70.0)
        self.assertEqual(metrics["perf/cuda/policy_physical_min_free_gb_min"], 6.0)

    def test_rejects_duplicate_rank_snapshots(self) -> None:
        snapshot = {
            "rank": 0,
            "rollout": None,
            "policy": _stage(allocated=1, reserved=2, peak=3, free=70),
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_cuda_memory_snapshots([snapshot, snapshot])

    def test_reports_actual_contiguous_rank_token_load(self) -> None:
        metrics = summarize_rank_token_load([10, 20, 30, 40, 50, 60, 70, 80], 4)

        self.assertEqual(metrics["perf/tokens/rows_per_rank"], 2.0)
        self.assertEqual(metrics["perf/tokens/rank_0/input_tokens"], 30.0)
        self.assertEqual(metrics["perf/tokens/rank_3/input_tokens"], 150.0)
        self.assertEqual(metrics["perf/tokens/max_to_min_ratio"], 5.0)


if __name__ == "__main__":
    unittest.main()
