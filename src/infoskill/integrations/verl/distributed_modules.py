from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True, slots=True)
class DistributedInfoSkillModules:
    """Worker-local M1 modules and the projector's policy optimizer state."""

    compressor: nn.Module
    projector: nn.Module
    projector_optimizer: torch.optim.Optimizer
    projector_scheduler: torch.optim.lr_scheduler.LRScheduler


def build_distributed_infoskill_modules(
    *,
    compressor: nn.Module,
    projector: nn.Module,
    device_index: int,
    total_policy_steps: int,
    projector_learning_rate: float = 1e-4,
    projector_weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    ddp_factory: Callable[..., nn.Module] = DistributedDataParallel,
) -> DistributedInfoSkillModules:
    """Give M1's replicated modules explicit DDP and optimizer ownership.

    The compressor is wrapped here so its later auxiliary backward uses the
    same distributed ownership as rollout conditioning. Its optimizer belongs
    to the auxiliary domain and is intentionally not constructed by this
    policy-focused seam.
    """

    if device_index < 0:
        raise ValueError("device_index must be non-negative")
    if total_policy_steps <= 0:
        raise ValueError("total_policy_steps must be positive")
    if projector_learning_rate <= 0:
        raise ValueError("projector_learning_rate must be positive")
    if projector_weight_decay < 0:
        raise ValueError("projector_weight_decay must be non-negative")
    if not 0 <= warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be in [0, 1]")

    ddp_options = {
        "device_ids": [device_index],
        "output_device": device_index,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
    }
    distributed_compressor = ddp_factory(compressor, **ddp_options)
    distributed_projector = ddp_factory(projector, **ddp_options)
    optimizer = torch.optim.AdamW(
        distributed_projector.parameters(),
        lr=projector_learning_rate,
        weight_decay=projector_weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    warmup_steps = int(total_policy_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps == 0:
            return 1.0
        return min(1.0, float(step) / float(warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return DistributedInfoSkillModules(
        compressor=distributed_compressor,
        projector=distributed_projector,
        projector_optimizer=optimizer,
        projector_scheduler=scheduler,
    )
