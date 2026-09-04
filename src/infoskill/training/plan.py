from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class TrainingProfile(str, Enum):
    SMOKE = "smoke"
    INTEGRATION = "integration"
    PILOT = "pilot"
    FORMAL = "formal"


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    profile: TrainingProfile
    max_updates: int
    task_groups_per_update: int
    rollouts_per_task: int
    action_minibatch_size: int
    checkpoint_every: int
    evaluation_every: int
    evaluation_kind: str
    include_monitor_tasks: bool

    @property
    def trajectories_per_full_update(self) -> int:
        return self.task_groups_per_update * self.rollouts_per_task


_REGISTERED_PLANS = {
    TrainingProfile.SMOKE: TrainingPlan(
        profile=TrainingProfile.SMOKE,
        max_updates=2,
        task_groups_per_update=1,
        rollouts_per_task=2,
        action_minibatch_size=16,
        checkpoint_every=1,
        evaluation_every=25,
        evaluation_kind="none",
        include_monitor_tasks=False,
    ),
    TrainingProfile.INTEGRATION: TrainingPlan(
        profile=TrainingProfile.INTEGRATION,
        max_updates=20,
        task_groups_per_update=4,
        rollouts_per_task=4,
        action_minibatch_size=64,
        checkpoint_every=5,
        evaluation_every=25,
        evaluation_kind="none",
        include_monitor_tasks=False,
    ),
    TrainingProfile.PILOT: TrainingPlan(
        profile=TrainingProfile.PILOT,
        max_updates=100,
        task_groups_per_update=8,
        rollouts_per_task=8,
        action_minibatch_size=256,
        checkpoint_every=5,
        evaluation_every=25,
        evaluation_kind="train_monitor",
        include_monitor_tasks=False,
    ),
    TrainingProfile.FORMAL: TrainingPlan(
        profile=TrainingProfile.FORMAL,
        max_updates=445,
        task_groups_per_update=8,
        rollouts_per_task=8,
        action_minibatch_size=256,
        checkpoint_every=5,
        evaluation_every=25,
        evaluation_kind="valid_seen",
        include_monitor_tasks=True,
    ),
}


def resolve_training_plan(
    profile: TrainingProfile | str,
    *,
    max_updates: int | None = None,
) -> TrainingPlan:
    selected = TrainingProfile(profile)
    plan = _REGISTERED_PLANS[selected]
    if max_updates is None:
        return plan
    if max_updates <= 0:
        raise ValueError("max_updates must be positive")
    if selected is TrainingProfile.FORMAL and max_updates != plan.max_updates:
        raise ValueError("formal training budget is fixed at 445 updates")
    return replace(plan, max_updates=max_updates)
