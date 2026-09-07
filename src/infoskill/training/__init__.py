"""Top-level INFO-SKILL training state machine."""

from .schedule import TaskSchedule, TaskScheduleState
from .plan import TrainingPlan, TrainingProfile, resolve_training_plan
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
    "TrainingRuntime",
    "UpdateMetrics",
    "resolve_training_plan",
]
