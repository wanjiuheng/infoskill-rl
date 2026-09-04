"""Pinned SkillRL/VERL runtime seam."""

try:
    from .runtime import VerlRuntime, VerlRuntimeConfig
except ModuleNotFoundError as error:
    if error.name not in {"torch", "ray", "omegaconf", "verl"}:
        raise
    VerlRuntime = None  # type: ignore[assignment,misc]
    VerlRuntimeConfig = None  # type: ignore[assignment,misc]

__all__ = ["VerlRuntime", "VerlRuntimeConfig"]
