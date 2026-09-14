"""Top-level INFO-SKILL training state machine."""

from .schedule import TaskSchedule, TaskScheduleState
from .plan import TrainingPlan, TrainingProfile, resolve_training_plan
from .rollout_curve import (
    TrainingRolloutStepScore,
    load_training_rollout_step_scores,
    write_training_rollout_steps_curve,
)
from .trainer import InfoSkillTrainer, TrainingRuntime, UpdateMetrics

try:
    from .auxiliary_batches import AuxiliaryBatchBuilder
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    AuxiliaryBatchBuilder = None  # type: ignore[assignment,misc]

__all__ = [
    "InfoSkillTrainer",
    "AuxiliaryBatchBuilder",
    "TaskSchedule",
    "TaskScheduleState",
    "TrainingPlan",
    "TrainingProfile",
    "TrainingRolloutStepScore",
    "TrainingRuntime",
    "UpdateMetrics",
    "load_training_rollout_step_scores",
    "resolve_training_plan",
    "write_training_rollout_steps_curve",
]
