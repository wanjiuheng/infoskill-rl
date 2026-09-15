from __future__ import annotations

from typing import Literal, Protocol, cast


class _OptionSource(Protocol):
    def get(self, key: str, default: object = None) -> object: ...


def policy_gradient_clip_mode(
    model_config: _OptionSource,
) -> Literal["joint", "separate"]:
    """Resolve the worker-owned policy clip mode from the model config seam."""
    value = str(model_config.get("infoskill_policy_gradient_clip_mode", "joint"))
    if value not in {"joint", "separate"}:
        raise ValueError("INFO-SKILL policy gradient clip mode must be joint or separate")
    return cast(Literal["joint", "separate"], value)
