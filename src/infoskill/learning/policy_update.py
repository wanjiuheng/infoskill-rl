from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Literal

import torch
from torch import Tensor, nn


class PolicyUpdateCoordinator:
    """Apply one finite-gated, atomically stepped LoRA/projector policy update."""

    def __init__(
        self,
        *,
        actor_parameters: Iterable[nn.Parameter],
        projector_parameters: Iterable[nn.Parameter],
        actor_optimizer: torch.optim.Optimizer,
        projector_optimizer: torch.optim.Optimizer,
        actor_scheduler: object | None = None,
        projector_scheduler: object | None = None,
        max_grad_norm: float = 1.0,
        gradient_clip_mode: Literal["joint", "separate"] = "joint",
    ) -> None:
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if gradient_clip_mode not in {"joint", "separate"}:
            raise ValueError("gradient_clip_mode must be joint or separate")
        self.actor_parameters = _unique_trainable(actor_parameters)
        self.projector_parameters = _unique_trainable(projector_parameters)
        if not self.actor_parameters or not self.projector_parameters:
            raise ValueError("both policy parameter groups must be non-empty")
        if {id(item) for item in self.actor_parameters} & {
            id(item) for item in self.projector_parameters
        }:
            raise ValueError("LoRA and projector parameters must not overlap")
        _require_exact_optimizer_parameters(actor_optimizer, self.actor_parameters, "actor")
        _require_exact_optimizer_parameters(
            projector_optimizer, self.projector_parameters, "projector"
        )
        self.actor_optimizer = actor_optimizer
        self.projector_optimizer = projector_optimizer
        self.actor_scheduler = actor_scheduler
        self.projector_scheduler = projector_scheduler
        self.max_grad_norm = float(max_grad_norm)
        self.gradient_clip_mode = gradient_clip_mode

    def zero_grad(self) -> None:
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.projector_optimizer.zero_grad(set_to_none=True)

    def step(self, *, actor_global_grad_norm: Tensor) -> Mapping[str, float]:
        """Gate and step both optimizers after the shared policy backward.

        ``actor_global_grad_norm`` must be the FSDP-global LoRA norm calculated
        without clipping. Projector gradients are already DDP-synchronized, so
        their local norm is counted once rather than once per rank.
        """

        if (
            not isinstance(actor_global_grad_norm, Tensor)
            or actor_global_grad_norm.numel() != 1
        ):
            raise TypeError("actor_global_grad_norm must be a scalar tensor")
        projector_norm = _gradient_norm(self.projector_parameters)
        actor_norm = actor_global_grad_norm.detach().float().reshape(())
        combined = torch.sqrt(actor_norm.square() + projector_norm.square())
        finite = bool(torch.isfinite(combined).item())
        if not finite:
            self.zero_grad()
            return {
                "policy/actor_grad_norm_before_clip": float(actor_norm.item()),
                "policy/projector_grad_norm_before_clip": float(projector_norm.item()),
                "policy/combined_grad_norm_before_clip": float(combined.item()),
                "policy/clip_coefficient": 0.0,
                "policy/joint_clip_coefficient": 0.0,
                "policy/actor_clip_coefficient": 0.0,
                "policy/projector_clip_coefficient": 0.0,
                "policy/actor_grad_norm_after_clip": 0.0,
                "policy/projector_grad_norm_after_clip": 0.0,
                "policy/projector_to_actor_grad_norm_ratio": float("nan"),
                "policy/separate_gradient_clipping": float(
                    self.gradient_clip_mode == "separate"
                ),
                "policy/optimizer_step_applied": 0.0,
                "policy/optimizer_skip_nonfinite": 1.0,
            }

        actor_norm_value = float(actor_norm.item())
        projector_norm_value = float(projector_norm.item())
        norm_value = float(combined.item())
        joint_coefficient = _clip_coefficient(norm_value, self.max_grad_norm)
        if self.gradient_clip_mode == "separate":
            actor_coefficient = _clip_coefficient(
                actor_norm_value, self.max_grad_norm
            )
            projector_coefficient = _clip_coefficient(
                projector_norm_value, self.max_grad_norm
            )
        else:
            actor_coefficient = joint_coefficient
            projector_coefficient = joint_coefficient
        if actor_coefficient < 1.0:
            _scale_gradients(self.actor_parameters, actor_coefficient)
        if projector_coefficient < 1.0:
            _scale_gradients(self.projector_parameters, projector_coefficient)
        self.actor_optimizer.step()
        self.projector_optimizer.step()
        _step_scheduler(self.actor_scheduler)
        _step_scheduler(self.projector_scheduler)
        self.zero_grad()
        return {
            "policy/actor_grad_norm_before_clip": float(actor_norm.item()),
            "policy/projector_grad_norm_before_clip": float(projector_norm.item()),
            "policy/combined_grad_norm_before_clip": norm_value,
            # Preserve the historical scalar as the most restrictive actual
            # coefficient while exposing the two policy domains explicitly.
            "policy/clip_coefficient": min(
                actor_coefficient, projector_coefficient
            ),
            "policy/joint_clip_coefficient": joint_coefficient,
            "policy/actor_clip_coefficient": actor_coefficient,
            "policy/projector_clip_coefficient": projector_coefficient,
            "policy/actor_grad_norm_after_clip": (
                actor_norm_value * actor_coefficient
            ),
            "policy/projector_grad_norm_after_clip": (
                projector_norm_value * projector_coefficient
            ),
            "policy/projector_to_actor_grad_norm_ratio": (
                projector_norm_value / actor_norm_value
                if actor_norm_value > 0
                else float("inf")
            ),
            "policy/separate_gradient_clipping": float(
                self.gradient_clip_mode == "separate"
            ),
            "policy/optimizer_step_applied": 1.0,
            "policy/optimizer_skip_nonfinite": 0.0,
        }


def _unique_trainable(parameters: Iterable[nn.Parameter]) -> tuple[nn.Parameter, ...]:
    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return tuple(result)


def _require_exact_optimizer_parameters(
    optimizer: torch.optim.Optimizer,
    parameters: tuple[nn.Parameter, ...],
    name: str,
) -> None:
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    }
    expected = {id(parameter) for parameter in parameters}
    if optimized != expected:
        raise ValueError(f"{name} optimizer parameters do not match its policy domain")


def _gradient_norm(parameters: tuple[nn.Parameter, ...]) -> Tensor:
    gradients = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    squared = torch.stack([gradient.square().sum() for gradient in gradients]).sum()
    return torch.sqrt(squared)


def _scale_gradients(parameters: tuple[nn.Parameter, ...], coefficient: float) -> None:
    if not math.isfinite(coefficient):
        raise ValueError("gradient scale coefficient must be finite")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(coefficient)


def _clip_coefficient(norm: float, maximum: float) -> float:
    return min(1.0, maximum / (norm + 1e-6))


def _step_scheduler(scheduler: object | None) -> None:
    if scheduler is not None:
        scheduler.step()  # type: ignore[attr-defined]
