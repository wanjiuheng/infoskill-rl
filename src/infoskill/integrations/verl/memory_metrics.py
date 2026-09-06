from __future__ import annotations

from collections.abc import Mapping, Sequence


_BYTES_PER_GIB = 1024**3


def summarize_cuda_memory_snapshots(
    snapshots: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Flatten per-rank worker snapshots without averaging ranks together."""

    if not snapshots:
        return {}
    metrics: dict[str, float] = {
        "perf/cuda/rank_count": float(len(snapshots)),
    }
    aggregate: dict[str, list[float]] = {}
    seen_ranks: set[int] = set()
    for snapshot in snapshots:
        rank = _required_non_negative_int(snapshot, "rank")
        if rank in seen_ranks:
            raise ValueError(f"duplicate CUDA memory snapshot for rank {rank}")
        seen_ranks.add(rank)
        for stage in ("rollout", "policy"):
            raw_stage = snapshot.get(stage)
            if raw_stage is None:
                continue
            if not isinstance(raw_stage, Mapping):
                raise TypeError(f"CUDA memory stage {stage!r} must be a mapping")
            values = _stage_values(raw_stage)
            for name, value in values.items():
                metrics[f"perf/cuda/rank_{rank}/{stage}_{name}_gb"] = value
                aggregate.setdefault(f"{stage}_{name}_gb", []).append(value)

    metrics.update(
        {
            f"perf/cuda/{name}_max": max(values)
            for name, values in aggregate.items()
        }
    )
    for stage in ("rollout", "policy"):
        free = aggregate.get(f"{stage}_free_gb")
        total = aggregate.get(f"{stage}_total_gb")
        headroom = aggregate.get(f"{stage}_conservative_headroom_gb")
        if free:
            metrics[f"perf/cuda/{stage}_free_gb_min"] = min(free)
        if total:
            metrics[f"perf/cuda/{stage}_total_gb_min"] = min(total)
        if headroom:
            metrics[f"perf/cuda/{stage}_conservative_headroom_gb_min"] = min(
                headroom
            )
    return metrics


def _stage_values(stage: Mapping[str, object]) -> dict[str, float]:
    required = (
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "free_bytes",
        "total_bytes",
    )
    values = {
        name.removesuffix("_bytes"): _required_non_negative_int(stage, name)
        / _BYTES_PER_GIB
        for name in required
    }
    values["device_used"] = values["total"] - values["free"]
    values["conservative_headroom"] = max(
        0.0,
        values["total"] - max(values["peak_reserved"], values["device_used"]),
    )
    return values


def _required_non_negative_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"CUDA memory field {key!r} must be a non-negative integer")
    return value
