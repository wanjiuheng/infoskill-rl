"""Runtime-independent learning rules and distributed training coordination."""

from .advantages import group_relative_advantages
from .alignment import (
    DEFAULT_LOGPROB_ALIGNMENT_THRESHOLDS,
    LogprobAlignmentThresholds,
    require_logprob_alignment,
    summarize_logprob_alignment,
)

try:
    from .losses import AuxiliaryLoss, GrpoLoss, auxiliary_loss, clipped_grpo_loss
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    AuxiliaryLoss = None  # type: ignore[assignment,misc]
    GrpoLoss = None  # type: ignore[assignment,misc]
    auxiliary_loss = None  # type: ignore[assignment]
    clipped_grpo_loss = None  # type: ignore[assignment]

__all__ = [
    "AuxiliaryLoss",
    "GrpoLoss",
    "DEFAULT_LOGPROB_ALIGNMENT_THRESHOLDS",
    "LogprobAlignmentThresholds",
    "auxiliary_loss",
    "clipped_grpo_loss",
    "group_relative_advantages",
    "require_logprob_alignment",
    "summarize_logprob_alignment",
]
