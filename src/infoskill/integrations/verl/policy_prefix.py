from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def recompute_policy_inputs_embeds(
    *,
    embedding: Callable[[Tensor], Tensor],
    projector: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    prefix_mask: Tensor,
    replay_latents: Tensor,
    detach_projector_output: bool = False,
) -> Tensor:
    """Replace explicit prefix slots with the current projector output.

    The replayed latent is an observation from the rollout policy, not a live
    compressor output. Detaching it here keeps GRPO gradients in the LoRA and
    projector policy domain defined by D003.
    """

    _validate_layout(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_mask=prefix_mask,
        replay_latents=replay_latents,
        projector=projector,
    )
    token_embeddings = embedding(input_ids)
    if token_embeddings.ndim != 3 or token_embeddings.shape[:2] != input_ids.shape:
        raise ValueError("policy embedding output must be [batch, sequence, hidden]")

    parameter = next(projector.parameters(), None)
    latent_dtype = parameter.dtype if parameter is not None else torch.float32
    prefix = projector(
        replay_latents.detach().to(
            device=input_ids.device,
            dtype=latent_dtype,
        )
    )
    if detach_projector_output:
        prefix = prefix.detach()
    projector_module = getattr(projector, "module", projector)
    expected = (
        input_ids.shape[0],
        int(getattr(projector_module, "prefix_length")),
        token_embeddings.shape[-1],
    )
    if tuple(prefix.shape) != expected:
        raise ValueError(
            f"current projector output has shape {tuple(prefix.shape)}, expected {expected}"
        )
    source = prefix.to(dtype=token_embeddings.dtype).reshape(-1)
    expanded_mask = prefix_mask.bool().unsqueeze(-1).expand_as(token_embeddings)
    return token_embeddings.masked_scatter(expanded_mask, source)


def _validate_layout(
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    prefix_mask: Tensor,
    replay_latents: Tensor,
    projector: nn.Module,
) -> None:
    if input_ids.ndim != 2:
        raise ValueError("policy input_ids must be [batch, sequence]")
    if tuple(attention_mask.shape) != tuple(input_ids.shape):
        raise ValueError("policy attention_mask must match input_ids")
    if tuple(prefix_mask.shape) != tuple(input_ids.shape):
        raise ValueError("INFO-SKILL prefix_mask must match input_ids")
    if replay_latents.ndim != 2 or replay_latents.shape[0] != input_ids.shape[0]:
        raise ValueError("INFO-SKILL replay_latents must be [batch, latent_dim]")
    projector_module = getattr(projector, "module", projector)
    prefix_length = getattr(projector_module, "prefix_length", None)
    if not isinstance(prefix_length, int) or prefix_length <= 0:
        raise ValueError("INFO-SKILL projector requires a positive prefix_length")
    if (prefix_mask.bool() & ~attention_mask.bool()).any():
        raise ValueError("every INFO-SKILL prefix slot must be attended")

    for row in prefix_mask.bool():
        positions = row.nonzero(as_tuple=False).flatten()
        if positions.numel() != prefix_length:
            raise ValueError(
                "each INFO-SKILL sample must contain exactly prefix_length slots"
            )
        expected = torch.arange(
            positions[0],
            positions[0] + prefix_length,
            device=positions.device,
        )
        if not torch.equal(positions, expected):
            raise ValueError("INFO-SKILL prefix slots must be contiguous")
