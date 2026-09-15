from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any


def fingerprint_vllm_generation_inputs(
    prompts: Sequence[dict[str, Any]], sampling_params: Sequence[object]
) -> dict[str, object]:
    """Hash the exact bounded input submitted to the pinned vLLM engine."""
    if len(prompts) != len(sampling_params):
        raise ValueError("vLLM prompts and sampling parameters have different lengths")
    requests = []
    for prompt, params in zip(prompts, sampling_params):
        prefix = prompt.get("infoskill_prefix_embeds")
        prefix_hash = None
        if prefix is not None:
            if isinstance(prefix, (bytes, bytearray)):
                raw = bytes(prefix)
                metadata = {"dtype": "bytes", "shape": [len(raw)]}
            else:
                import torch

                if not isinstance(prefix, torch.Tensor):
                    raise TypeError("vLLM prefix fingerprint requires a tensor")
                detached = prefix.detach().to("cpu").contiguous()
                raw = detached.view(torch.uint8).numpy().tobytes()
                metadata = {
                    "dtype": str(detached.dtype),
                    "shape": list(detached.shape),
                }
            digest = hashlib.sha256()
            digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
            digest.update(raw)
            prefix_hash = digest.hexdigest()
        payload = {
            "prompt_token_ids": list(prompt["prompt_token_ids"]),
            "prefix_sha256": prefix_hash,
            "prefix_mask": prompt.get("infoskill_prefix_mask"),
            "seed": getattr(params, "seed", None),
            "temperature": getattr(params, "temperature", None),
            "top_p": getattr(params, "top_p", None),
            "max_tokens": getattr(params, "max_tokens", None),
            "sampling": {
                name: getattr(params, name, None)
                for name in (
                    "stop", "stop_token_ids", "top_k", "min_p", "n", "best_of",
                    "repetition_penalty", "presence_penalty", "frequency_penalty",
                    "ignore_eos", "logprobs", "detokenize",
                    "include_stop_str_in_output",
                )
            },
        }
        requests.append(hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest())
    return {
        "request_sha256": requests,
        "batch_sha256": hashlib.sha256(
            json.dumps(requests, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


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
