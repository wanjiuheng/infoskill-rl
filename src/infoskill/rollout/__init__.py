"""Policy generation interface shared by Transformers and VERL adapters."""

from .contracts import GenerationParameters, GenerationRequest, GenerationResult, RolloutBackend

try:
    from .transformers_backend import TransformersBackend
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    TransformersBackend = None  # type: ignore[assignment,misc]

__all__ = [
    "GenerationParameters",
    "GenerationRequest",
    "GenerationResult",
    "RolloutBackend",
    "TransformersBackend",
]
