from __future__ import annotations

from typing import TypeVar


BatchT = TypeVar("BatchT")


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
