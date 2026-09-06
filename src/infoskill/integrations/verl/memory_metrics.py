from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence


_BYTES_PER_GIB = 1024**3


class PhysicalMemorySampler:
    """Poll a physical device-memory reader on a small diagnostic thread."""

    def __init__(
        self,
        *,
        read_memory: Callable[[], tuple[int, int]],
        interval_ms: int,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError("memory polling interval must be positive")
        self._read_memory = read_memory
        self._interval_seconds = interval_ms / 1000
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._minimum_free_bytes: int | None = None
        self._total_bytes: int | None = None
        self._sample_count = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="infoskill-cuda-memory",
            daemon=True,
        )

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> dict[str, int]:
        final_error: BaseException | None = None
        try:
            self._sample()
        except BaseException as error:
            final_error = error
        finally:
            self._stop_event.set()
            self._thread.join(timeout=max(1.0, self._interval_seconds * 4))
        if self._thread.is_alive():
            raise RuntimeError("physical memory sampler did not stop")
        if self._error is not None:
            raise RuntimeError("physical memory sampler failed") from self._error
        if final_error is not None:
            raise RuntimeError("physical memory sampler failed") from final_error
        with self._lock:
            if self._minimum_free_bytes is None or self._total_bytes is None:
                raise RuntimeError("physical memory sampler collected no samples")
            return {
                "physical_min_free_bytes": self._minimum_free_bytes,
                "physical_total_bytes": self._total_bytes,
                "physical_sample_count": self._sample_count,
            }

    def _run(self) -> None:
        try:
            while not self._stop_event.wait(self._interval_seconds):
                self._sample()
        except BaseException as error:
            self._error = error
            self._stop_event.set()

    def _sample(self) -> None:
        free_bytes, total_bytes = self._read_memory()
        free = int(free_bytes)
        total = int(total_bytes)
        if free < 0 or total <= 0 or free > total:
            raise ValueError("physical memory reader returned invalid free/total bytes")
        with self._lock:
            self._minimum_free_bytes = (
                free
                if self._minimum_free_bytes is None
                else min(self._minimum_free_bytes, free)
            )
            if self._total_bytes is not None and self._total_bytes != total:
                raise RuntimeError("physical device total memory changed during sampling")
            self._total_bytes = total
            self._sample_count += 1


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
            physical_samples = raw_stage.get("physical_sample_count")
            if physical_samples is not None:
                sample_count = float(
                    _required_non_negative_int(
                        raw_stage,
                        "physical_sample_count",
                    )
                )
                metrics[
                    f"perf/cuda/rank_{rank}/{stage}_physical_sample_count"
                ] = sample_count
                aggregate.setdefault(
                    f"{stage}_physical_sample_count",
                    [],
                ).append(sample_count)

    metrics.update(
        {
            f"perf/cuda/{name}_max": max(values)
            for name, values in aggregate.items()
        }
    )
    for stage in ("rollout", "policy"):
        free = aggregate.get(f"{stage}_free_gb")
        total = aggregate.get(f"{stage}_total_gb")
        physical_free = aggregate.get(f"{stage}_physical_min_free_gb")
        if free:
            metrics[f"perf/cuda/{stage}_free_gb_min"] = min(free)
        if total:
            metrics[f"perf/cuda/{stage}_total_gb_min"] = min(total)
        if physical_free:
            metrics[f"perf/cuda/{stage}_physical_min_free_gb_min"] = min(
                physical_free
            )
    return metrics


def summarize_rank_token_load(
    token_counts: Sequence[int],
    world_size: int,
    *,
    prefix: str = "perf/tokens",
) -> dict[str, float]:
    """Report the contiguous equal-row partition used by DataProto.chunk()."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not token_counts or len(token_counts) % world_size:
        raise ValueError("token counts must be non-empty and divisible by world_size")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in token_counts
    ):
        raise ValueError("token counts must be non-negative integers")
    rows_per_rank = len(token_counts) // world_size
    totals = [
        sum(token_counts[rank * rows_per_rank : (rank + 1) * rows_per_rank])
        for rank in range(world_size)
    ]
    mean = sum(totals) / world_size
    metrics = {
        f"{prefix}/rank_count": float(world_size),
        f"{prefix}/rows_per_rank": float(rows_per_rank),
        f"{prefix}/input_tokens_min": float(min(totals)),
        f"{prefix}/input_tokens_max": float(max(totals)),
        f"{prefix}/input_tokens_mean": float(mean),
        f"{prefix}/max_to_min_ratio": (
            float(max(totals) / min(totals)) if min(totals) else float("inf")
        ),
        f"{prefix}/max_to_mean_ratio": (
            float(max(totals) / mean) if mean else float("inf")
        ),
    }
    metrics.update(
        {
            f"{prefix}/rank_{rank}/input_tokens": float(total)
            for rank, total in enumerate(totals)
        }
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
    physical_min_free = stage.get("physical_min_free_bytes")
    if physical_min_free is not None:
        values["physical_min_free"] = (
            _required_non_negative_int(stage, "physical_min_free_bytes")
            / _BYTES_PER_GIB
        )
        values["physical_peak_device_used"] = (
            values["total"] - values["physical_min_free"]
        )
    return values


def _required_non_negative_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"CUDA memory field {key!r} must be a non-negative integer")
    return value
