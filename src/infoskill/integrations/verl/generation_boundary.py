from __future__ import annotations

from collections.abc import Sequence


VLLM_PADDING_LOGPROB_SENTINEL = -1.0


def trim_vllm_padding_sentinel(
    *,
    token_ids: Sequence[int],
    token_logprobs: Sequence[float],
    pad_token_id: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Remove VERL padding positions that its response mask keeps as terminal EOS."""
    if len(token_ids) != len(token_logprobs):
        raise ValueError("token IDs and logprobs must have equal length")
    end = len(token_ids)
    while (
        end > 0
        and int(token_ids[end - 1]) == pad_token_id
        and float(token_logprobs[end - 1]) == VLLM_PADDING_LOGPROB_SENTINEL
    ):
        end -= 1
    return (
        tuple(int(value) for value in token_ids[:end]),
        tuple(float(value) for value in token_logprobs[:end]),
    )
