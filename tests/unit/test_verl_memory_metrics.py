from __future__ import annotations

import unittest

from infoskill.integrations.verl.memory_metrics import (
    summarize_cuda_memory_snapshots,
)


_GIB = 1024**3


def _stage(*, allocated: int, reserved: int, peak: int, free: int) -> dict[str, int]:
    return {
        "allocated_bytes": allocated * _GIB,
        "reserved_bytes": reserved * _GIB,
        "peak_allocated_bytes": peak * _GIB,
        "peak_reserved_bytes": (peak + 1) * _GIB,
        "free_bytes": free * _GIB,
        "total_bytes": 80 * _GIB,
    }


class VerlMemoryMetricsTests(unittest.TestCase):
    def test_preserves_each_rank_and_reports_conservative_extrema(self) -> None:
        metrics = summarize_cuda_memory_snapshots(
            [
                {
                    "rank": 0,
                    "rollout": _stage(allocated=20, reserved=22, peak=24, free=55),
                    "policy": _stage(allocated=60, reserved=64, peak=68, free=12),
                },
                {
                    "rank": 1,
                    "rollout": _stage(allocated=21, reserved=23, peak=25, free=54),
                    "policy": _stage(allocated=61, reserved=65, peak=70, free=10),
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
        self.assertEqual(
            metrics["perf/cuda/policy_conservative_headroom_gb_min"], 9.0
        )

    def test_rejects_duplicate_rank_snapshots(self) -> None:
        snapshot = {
            "rank": 0,
            "rollout": None,
            "policy": _stage(allocated=1, reserved=2, peak=3, free=70),
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_cuda_memory_snapshots([snapshot, snapshot])


if __name__ == "__main__":
    unittest.main()
