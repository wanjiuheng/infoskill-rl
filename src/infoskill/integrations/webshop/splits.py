from __future__ import annotations

from enum import Enum


class WebShopSplit(str, Enum):
    """Official WebShop goal-index partitions."""

    TEST = "test"
    VALIDATION = "validation"
    TRAIN = "train"


TEST_STOP = 500
VALIDATION_STOP = 1500


def goal_indices(split: WebShopSplit | str, *, goal_count: int) -> range:
    """Return the official, non-overlapping goal indices for ``split``."""

    normalized = _normalize_split(split)
    if goal_count <= VALIDATION_STOP:
        raise ValueError(
            "WebShop goal corpus must contain more than 1500 instructions"
        )
    if normalized is WebShopSplit.TEST:
        return range(0, TEST_STOP)
    if normalized is WebShopSplit.VALIDATION:
        return range(TEST_STOP, VALIDATION_STOP)
    return range(VALIDATION_STOP, goal_count)


def split_for_goal_index(index: int, *, goal_count: int) -> WebShopSplit:
    """Classify one goal index under the official WebShop protocol."""

    if index < 0 or index >= goal_count:
        raise IndexError(f"WebShop goal index out of range: {index}")
    if index < TEST_STOP:
        return WebShopSplit.TEST
    if index < VALIDATION_STOP:
        return WebShopSplit.VALIDATION
    return WebShopSplit.TRAIN


def _normalize_split(split: WebShopSplit | str) -> WebShopSplit:
    if isinstance(split, WebShopSplit):
        return split
    value = str(split).strip().lower()
    if value == "eval":
        value = WebShopSplit.VALIDATION.value
    try:
        return WebShopSplit(value)
    except ValueError as error:
        raise ValueError("WebShop split must be train, validation/eval, or test") from error
