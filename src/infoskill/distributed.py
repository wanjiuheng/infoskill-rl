from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar


BatchT = TypeVar("BatchT")
Partitioner = Callable[[list[int], int, bool], Sequence[Sequence[int]]]


def pad_batch_to_divisor(batch: BatchT, divisor: int) -> tuple[BatchT, int]:
    """Repeat leading rows without mutating ``batch`` until its size is divisible."""
    if divisor <= 0:
        raise ValueError("batch divisor must be positive")
    batch_size = len(batch)  # type: ignore[arg-type]
    if batch_size <= 0:
        raise ValueError("cannot pad an empty batch")
    padding_count = (-batch_size) % divisor
    if padding_count == 0:
        return batch, 0

    parts = [batch]
    remaining = padding_count
    while remaining:
        take = min(remaining, batch_size)
        parts.append(batch[:take])  # type: ignore[index]
        remaining -= take
    concat = getattr(type(batch), "concat", None)
    if not callable(concat):
        raise TypeError("batch type must provide a callable concat class method")
    return concat(parts), padding_count


def policy_rank_balanced_order(
    token_counts: Sequence[int],
    *,
    world_size: int,
    global_minibatch_size: int,
    partitioner: Partitioner,
) -> tuple[int, ...]:
    """Balance rank work without changing synchronized minibatch membership."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if global_minibatch_size <= 0 or global_minibatch_size % world_size:
        raise ValueError(
            "global_minibatch_size must be positive and divisible by world_size"
        )
    counts = [int(value) for value in token_counts]
    if not counts or len(counts) % world_size:
        raise ValueError("token_counts must be non-empty and divisible by world_size")
    if any(
        value < 0 or value != original
        for value, original in zip(counts, token_counts)
    ):
        raise ValueError("token_counts must contain non-negative integers")

    rows_per_rank = len(counts) // world_size
    rows_per_rank_minibatch = global_minibatch_size // world_size
    rank_orders: list[list[int]] = [[] for _ in range(world_size)]

    for local_start in range(0, rows_per_rank, rows_per_rank_minibatch):
        local_stop = min(
            rows_per_rank,
            local_start + rows_per_rank_minibatch,
        )
        synchronized_indices = [
            rank * rows_per_rank + local_index
            for rank in range(world_size)
            for local_index in range(local_start, local_stop)
        ]
        synchronized_counts = [counts[index] for index in synchronized_indices]
        partitions = partitioner(synchronized_counts, world_size, True)
        _validate_equal_partitions(
            partitions,
            item_count=len(synchronized_indices),
            world_size=world_size,
        )
        for rank, partition in enumerate(partitions):
            rank_orders[rank].extend(
                synchronized_indices[index] for index in partition
            )

    order = tuple(index for rank_order in rank_orders for index in rank_order)
    if sorted(order) != list(range(len(counts))):
        raise RuntimeError("rank balancing did not produce a complete permutation")
    if any(len(rank_order) != rows_per_rank for rank_order in rank_orders):
        raise RuntimeError("rank balancing changed the equal row count per rank")
    return order


def _validate_equal_partitions(
    partitions: Sequence[Sequence[int]],
    *,
    item_count: int,
    world_size: int,
) -> None:
    if len(partitions) != world_size or item_count % world_size:
        raise RuntimeError("partitioner returned an invalid rank count")
    expected_size = item_count // world_size
    if any(len(partition) != expected_size for partition in partitions):
        raise RuntimeError("partitioner returned unequal rank row counts")
    flattened = [index for partition in partitions for index in partition]
    if sorted(flattened) != list(range(item_count)):
        raise RuntimeError("partitioner did not return a complete local permutation")
