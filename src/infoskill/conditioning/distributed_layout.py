from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


_Row = TypeVar("_Row")
_Result = TypeVar("_Result")


def _rank_replicated_conditioning_rows(
    rows: tuple[_Row, ...],
    world_size: int,
) -> tuple[_Row, ...]:
    """Lay out one identical logical sequence for every data-parallel rank."""

    if world_size <= 0:
        raise ValueError("conditioning world size must be positive")
    return rows * world_size


def _select_rank_zero_conditioning_rows(
    rows: tuple[_Result, ...],
    *,
    logical_size: int,
    world_size: int,
) -> tuple[_Result, ...]:
    """Select the same rank-zero results retained by legacy one-row RPCs."""

    if logical_size < 0:
        raise ValueError("conditioning logical size must be nonnegative")
    if world_size <= 0:
        raise ValueError("conditioning world size must be positive")
    expected = logical_size * world_size
    if len(rows) != expected:
        raise ValueError(
            "conditioning result layout has "
            f"{len(rows)} rows instead of {expected}"
        )
    return rows[:logical_size]


def _condition_rows_individually(
    rows: tuple[_Row, ...],
    condition_batch: Callable[[tuple[_Row, ...]], tuple[_Result, ...]],
) -> tuple[_Result, ...]:
    """Amortize the RPC while retaining legacy batch-size-one numerics."""

    results: list[_Result] = []
    for row in rows:
        conditioned = condition_batch((row,))
        if len(conditioned) != 1:
            raise RuntimeError(
                "single-row conditioning must return exactly one result"
            )
        results.append(conditioned[0])
    return tuple(results)
