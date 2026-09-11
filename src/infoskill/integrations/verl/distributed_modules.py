from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True, slots=True)
class DistributedInfoSkillModules:
    """Worker-local M1 modules with disjoint policy and auxiliary ownership."""

    compressor: nn.Module
    projector: nn.Module
    prior: nn.Module
    fidelity: nn.Module
    grounding: nn.Module
    projector_optimizer: torch.optim.Optimizer
    projector_scheduler: torch.optim.lr_scheduler.LRScheduler
    auxiliary_optimizer: torch.optim.Optimizer
    auxiliary_scheduler: torch.optim.lr_scheduler.LRScheduler


def build_distributed_infoskill_modules(
    *,
    compressor: nn.Module,
    projector: nn.Module,
    prior: nn.Module,
    fidelity: nn.Module,
    grounding: nn.Module,
    device_index: int,
    total_policy_steps: int,
    projector_learning_rate: float = 1e-4,
    projector_weight_decay: float = 0.01,
    auxiliary_learning_rate: float = 1e-4,
    auxiliary_weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    auxiliary_warmup_ratio: float = 0.03,
    ddp_factory: Callable[..., nn.Module] = DistributedDataParallel,
) -> DistributedInfoSkillModules:
    """Give replicated M1 modules explicit DDP and optimizer ownership."""

    if device_index < 0:
        raise ValueError("device_index must be non-negative")
    if total_policy_steps <= 0:
        raise ValueError("total_policy_steps must be positive")
    if projector_learning_rate <= 0:
        raise ValueError("projector_learning_rate must be positive")
    if projector_weight_decay < 0:
        raise ValueError("projector_weight_decay must be non-negative")
    if auxiliary_learning_rate <= 0:
        raise ValueError("auxiliary_learning_rate must be positive")
    if auxiliary_weight_decay < 0:
        raise ValueError("auxiliary_weight_decay must be non-negative")
    if not 0 <= warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be in [0, 1]")
    if not 0 <= auxiliary_warmup_ratio <= 1:
        raise ValueError("auxiliary_warmup_ratio must be in [0, 1]")

    ddp_options = {
        "device_ids": [device_index],
        "output_device": device_index,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
    }
    distributed_compressor = ddp_factory(compressor, **ddp_options)
    distributed_projector = ddp_factory(projector, **ddp_options)
    distributed_prior = ddp_factory(prior, **ddp_options)
    distributed_fidelity = ddp_factory(fidelity, **ddp_options)
    distributed_grounding = ddp_factory(grounding, **ddp_options)
    projector_optimizer = torch.optim.AdamW(
        distributed_projector.parameters(),
        lr=projector_learning_rate,
        weight_decay=projector_weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    auxiliary_modules = (
        distributed_compressor,
        distributed_prior,
        distributed_fidelity,
        distributed_grounding,
    )
    auxiliary_optimizer = torch.optim.AdamW(
        [
            parameter
            for module in auxiliary_modules
            for parameter in module.parameters()
        ],
        lr=auxiliary_learning_rate,
        weight_decay=auxiliary_weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    projector_scheduler = _warmup_scheduler(
        projector_optimizer,
        total_steps=total_policy_steps,
        warmup_ratio=warmup_ratio,
    )
    auxiliary_scheduler = _warmup_scheduler(
        auxiliary_optimizer,
        total_steps=total_policy_steps,
        warmup_ratio=auxiliary_warmup_ratio,
    )
    return DistributedInfoSkillModules(
        compressor=distributed_compressor,
        projector=distributed_projector,
        prior=distributed_prior,
        fidelity=distributed_fidelity,
        grounding=distributed_grounding,
        projector_optimizer=projector_optimizer,
        projector_scheduler=projector_scheduler,
        auxiliary_optimizer=auxiliary_optimizer,
        auxiliary_scheduler=auxiliary_scheduler,
    )


def _warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps == 0:
            return 1.0
        return min(1.0, float(step) / float(warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
