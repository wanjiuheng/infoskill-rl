from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any


def build_hybrid_vllm_inputs(
    *,
    raw_prompt_ids: Sequence[Sequence[int]],
    soft_prefixes: Sequence[object | None],
    placeholder_token_id: int,
) -> list[dict[str, Any]]:
    """Build full token layouts while transporting only short prefix vectors."""
    if len(raw_prompt_ids) != len(soft_prefixes):
        raise ValueError("soft-prefix batch size must match prompt batch size")
    if placeholder_token_id < 0:
        raise ValueError("placeholder_token_id must be non-negative")

    results: list[dict[str, Any]] = []
    for prompt_ids, prefix in zip(raw_prompt_ids, soft_prefixes):
        text_ids = [int(token_id) for token_id in prompt_ids]
        if prefix is None:
            results.append({"prompt_token_ids": text_ids})
            continue
        if getattr(prefix, "ndim", None) != 2:
            raise TypeError("soft prefix must be a [prefix_length, hidden_size] tensor")
        prefix_length = int(prefix.shape[0])  # type: ignore[index]
        hidden_size = int(prefix.shape[1])  # type: ignore[index]
        if prefix_length <= 0 or hidden_size <= 0:
            raise ValueError("soft prefix dimensions must be positive")
        transported = prefix.detach().to("cpu").contiguous()  # type: ignore[attr-defined]
        results.append(
            {
                "prompt_token_ids": [placeholder_token_id] * prefix_length + text_ids,
                "infoskill_prefix_embeds": transported,
                "infoskill_prefix_mask": [True] * prefix_length + [False] * len(text_ids),
            }
        )
    return results


def clone_sampling_params_with_seeds(base: object, seeds: Sequence[int]) -> list[object]:
    """Clone vLLM SamplingParams so every request owns its semantic RNG seed."""
    results = []
    for seed in seeds:
        clone = base.clone()  # type: ignore[attr-defined]
        clone.seed = int(seed)  # type: ignore[attr-defined]
        results.append(clone)
    return results


@contextmanager
def temporary_sampling_overrides(
    sampling_params: object,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    response_cap: int,
):
    if max_tokens <= 0 or max_tokens > response_cap:
        raise ValueError(
            f"requested max_tokens={max_tokens} exceeds runtime response cap={response_cap}"
        )
    old = {
        name: getattr(sampling_params, name)
        for name in ("temperature", "top_p", "max_tokens")
    }
    sampling_params.temperature = float(temperature)  # type: ignore[attr-defined]
    sampling_params.top_p = float(top_p)  # type: ignore[attr-defined]
    sampling_params.max_tokens = int(max_tokens)  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(sampling_params, name, value)
