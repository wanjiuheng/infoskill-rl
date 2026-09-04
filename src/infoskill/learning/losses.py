from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class GrpoLoss:
    total: Tensor
    policy: Tensor
    reference_kl: Tensor
    entropy_bonus: Tensor
    clip_fraction: Tensor
    approximate_kl: Tensor


def clipped_grpo_loss(
    *,
    new_logprob: Tensor,
    old_logprob: Tensor,
    reference_logprob: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    entropy: Tensor | None = None,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    reference_kl_coefficient: float = 0.01,
    entropy_coefficient: float = 0.001,
) -> GrpoLoss:
    mask = response_mask.to(new_logprob.dtype)
    if new_logprob.shape != old_logprob.shape or new_logprob.shape != reference_logprob.shape:
        raise ValueError("new, old, and reference logprob tensors must have equal shapes")
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    if advantages.shape[0] != new_logprob.shape[0]:
        raise ValueError("advantages must align with response sequences")
    counts = mask.sum(dim=-1)
    if (counts <= 0).any():
        raise ValueError("every response sequence requires at least one valid token")
    log_ratio = new_logprob - old_logprob
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high) * advantages
    policy_tokens = -torch.minimum(unclipped, clipped)
    policy = _sequence_mean(policy_tokens, mask, counts)
    ref_delta = reference_logprob - new_logprob
    low_variance_kl = (torch.exp(ref_delta) - ref_delta - 1.0).clamp(-10.0, 10.0)
    reference_kl = _sequence_mean(low_variance_kl, mask, counts)
    if entropy is None:
        entropy_bonus = new_logprob.new_zeros(())
    else:
        entropy_bonus = _sequence_mean(entropy, mask, counts)
    total = policy + reference_kl_coefficient * reference_kl - entropy_coefficient * entropy_bonus
    clipped_positions = (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
    clip_fraction = (clipped_positions.to(mask.dtype) * mask).sum() / mask.sum()
    approximate_kl = _sequence_mean(-log_ratio, mask, counts)
    return GrpoLoss(total, policy, reference_kl, entropy_bonus, clip_fraction, approximate_kl)


@dataclass(frozen=True)
class AuxiliaryLoss:
    total: Tensor
    fidelity: Tensor
    rate: Tensor
    grounding: Tensor


def auxiliary_loss(
    *,
    fidelity_prediction: Tensor,
    fidelity_target: Tensor,
    online_rate_by_step: Tensor,
    trajectory_index: Tensor,
    grounding_logits: Tensor,
    grounding_targets: Tensor,
    offline_rate: Tensor,
    fidelity_weight: float = 1.0,
    rate_weight: float = 0.001,
    grounding_weight: float = 0.1,
) -> AuxiliaryLoss:
    fidelity = _trajectory_balanced_mean(
        (fidelity_prediction - fidelity_target.detach()).square(), trajectory_index
    )
    online_rate = _trajectory_balanced_mean(online_rate_by_step, trajectory_index)
    rate = 0.5 * online_rate + 0.5 * offline_rate.mean()
    grounding = F.cross_entropy(grounding_logits, grounding_targets)
    total = fidelity_weight * fidelity + rate_weight * rate + grounding_weight * grounding
    return AuxiliaryLoss(total, fidelity, rate, grounding)


def _sequence_mean(values: Tensor, mask: Tensor, counts: Tensor) -> Tensor:
    return ((values * mask).sum(dim=-1) / counts).mean()


def _trajectory_balanced_mean(values: Tensor, trajectory_index: Tensor) -> Tensor:
    unique = torch.unique(trajectory_index, sorted=True)
    if unique.numel() == 0:
        raise ValueError("auxiliary loss requires at least one trajectory")
    return torch.stack([values[trajectory_index == index].mean() for index in unique]).mean()
