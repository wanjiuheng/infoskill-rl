from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PolicyReplayExample:
    """One environment step at the policy-recomputation boundary."""

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    advantage: float
    latent: Tensor | None = None
    rollout_prefix: Tensor | None = None


def build_policy_replay_tensors(
    examples: Sequence[PolicyReplayExample],
    *,
    pad_token_id: int,
) -> dict[str, Tensor]:
    """Build one token-only or exact-latent INFO-SKILL replay batch.

    INFO-SKILL rows contain explicit, attended placeholder positions before the
    first text token. The worker must replace those embeddings with a prefix
    recomputed from ``infoskill_replay_latents``. The rollout prefix is retained
    only for parity/audit and must never be reused as the trainable prefix.
    """

    if not examples:
        raise ValueError("policy replay requires at least one example")
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be non-negative")

    prefixed = [
        example.latent is not None or example.rollout_prefix is not None
        for example in examples
    ]
    if any(prefixed) and not all(prefixed):
        raise ValueError("policy replay cannot mix prefixed and token-only examples")
    use_prefix = all(prefixed)

    prefix_length = 0
    if use_prefix:
        geometries: set[tuple[int, int]] = set()
        latent_dims: set[int] = set()
        for example in examples:
            latent = _require_vector(example.latent, "latent")
            prefix = _require_matrix(example.rollout_prefix, "rollout_prefix")
            latent_dims.add(int(latent.shape[0]))
            geometries.add((int(prefix.shape[0]), int(prefix.shape[1])))
        if len(geometries) != 1:
            raise ValueError("INFO-SKILL prefix geometry must be identical across examples")
        if len(latent_dims) != 1:
            raise ValueError("INFO-SKILL latent width must be identical across examples")
        prefix_length = next(iter(geometries))[0]

    for example in examples:
        if not example.prompt_ids:
            raise ValueError("policy replay prompt cannot be empty")
        if len(example.response_ids) != len(example.response_logprobs):
            raise ValueError("response token and logprob lengths must match")

    prompt_width = max(prefix_length + len(example.prompt_ids) for example in examples)
    response_width = max(max(len(example.response_ids) for example in examples), 1)
    total_width = prompt_width + response_width
    rows = len(examples)

    input_ids = torch.full((rows, total_width), pad_token_id, dtype=torch.long)
    prompts = torch.full((rows, prompt_width), pad_token_id, dtype=torch.long)
    responses = torch.full((rows, response_width), pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    advantages = torch.zeros((rows, response_width), dtype=torch.float32)
    rollout_logprobs = torch.zeros((rows, response_width), dtype=torch.float32)
    prefix_mask = torch.zeros((rows, total_width), dtype=torch.bool) if use_prefix else None

    for row, example in enumerate(examples):
        occupied_prompt = prefix_length + len(example.prompt_ids)
        prompt_start = prompt_width - occupied_prompt
        text_start = prompt_start + prefix_length
        if use_prefix:
            assert prefix_mask is not None
            prefix_mask[row, prompt_start:text_start] = True
            attention[row, prompt_start:text_start] = 1

        prompt_tensor = torch.tensor(example.prompt_ids, dtype=torch.long)
        prompts[row, text_start:prompt_width] = prompt_tensor
        input_ids[row, text_start:prompt_width] = prompt_tensor
        attention[row, text_start:prompt_width] = 1
        if example.response_ids:
            response_length = len(example.response_ids)
            response_tensor = torch.tensor(example.response_ids, dtype=torch.long)
            responses[row, :response_length] = response_tensor
            input_ids[row, prompt_width : prompt_width + response_length] = response_tensor
            attention[row, prompt_width : prompt_width + response_length] = 1
            advantages[row, :response_length] = float(example.advantage)
            rollout_logprobs[row, :response_length] = torch.tensor(
                example.response_logprobs,
                dtype=torch.float32,
            )

    tensors = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention,
        "position_ids": (attention.cumsum(dim=-1) - 1).clamp_min(0),
        "advantages": advantages,
        "rollout_log_probs": rollout_logprobs,
    }
    if use_prefix:
        assert prefix_mask is not None
        tensors.update(
            {
                "infoskill_prefix_mask": prefix_mask,
                "infoskill_replay_latents": torch.stack(
                    [_require_vector(example.latent, "latent") for example in examples]
                ).detach().to(device="cpu", copy=True),
                "infoskill_rollout_prefixes": torch.stack(
                    [
                        _require_matrix(example.rollout_prefix, "rollout_prefix")
                        for example in examples
                    ]
                ).detach().to(device="cpu", copy=True),
            }
        )
    return tensors


def _require_vector(value: Tensor | None, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"INFO-SKILL {name} must be a non-empty vector")
    return value


def _require_matrix(value: Tensor | None, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.shape[0] == 0
        or value.shape[1] == 0
    ):
        raise ValueError(f"INFO-SKILL {name} must be a non-empty matrix")
    return value
