"""Runtime-independent learning rules and distributed training coordination."""

from .advantages import group_relative_advantages
from .alignment import (
    DEFAULT_LOGPROB_ALIGNMENT_THRESHOLDS,
    LogprobAlignmentError,
    LogprobAlignmentThresholds,
    alignment_passes,
    collect_logprob_alignment_offenders,
    require_logprob_alignment,
    summarize_logprob_alignment,
)
from .signals import (
    GroupAdvantageSignals,
    build_group_advantage_signals,
    summarize_grpo_signals,
)
try:
    from .auxiliary import (
        AuxiliaryTrainingBatch,
        AuxiliaryUpdateConfig,
        AuxiliaryUpdater,
        CompressionReplayBatch,
        OfflineGroundingBatch,
        OnlineAuxiliaryBatch,
    )
    from .losses import AuxiliaryLoss, GrpoLoss, auxiliary_loss, clipped_grpo_loss
    from .policy_update import PolicyUpdateCoordinator
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    AuxiliaryLoss = None  # type: ignore[assignment,misc]
    AuxiliaryTrainingBatch = None  # type: ignore[assignment,misc]
    AuxiliaryUpdateConfig = None  # type: ignore[assignment,misc]
    AuxiliaryUpdater = None  # type: ignore[assignment,misc]
    CompressionReplayBatch = None  # type: ignore[assignment,misc]
    GrpoLoss = None  # type: ignore[assignment,misc]
    OfflineGroundingBatch = None  # type: ignore[assignment,misc]
    OnlineAuxiliaryBatch = None  # type: ignore[assignment,misc]
    PolicyUpdateCoordinator = None  # type: ignore[assignment,misc]
    auxiliary_loss = None  # type: ignore[assignment]
    clipped_grpo_loss = None  # type: ignore[assignment]

__all__ = [
    "AuxiliaryLoss",
    "AuxiliaryTrainingBatch",
    "AuxiliaryUpdateConfig",
    "AuxiliaryUpdater",
    "CompressionReplayBatch",
    "GrpoLoss",
    "DEFAULT_LOGPROB_ALIGNMENT_THRESHOLDS",
    "LogprobAlignmentError",
    "LogprobAlignmentThresholds",
    "alignment_passes",
    "collect_logprob_alignment_offenders",
    "OfflineGroundingBatch",
    "OnlineAuxiliaryBatch",
    "GroupAdvantageSignals",
    "auxiliary_loss",
    "clipped_grpo_loss",
    "group_relative_advantages",
    "build_group_advantage_signals",
    "require_logprob_alignment",
    "summarize_logprob_alignment",
    "PolicyUpdateCoordinator",
    "summarize_grpo_signals",
]
