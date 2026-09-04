from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

from infoskill.episode import TrajectoryCollector, TrajectoryGroup
from infoskill.learning import group_relative_advantages

from .schedule import TaskSchedule
from .schedule import TaskScheduleState


@dataclass(frozen=True, slots=True)
class UpdateMetrics:
    global_update: int
    values: Mapping[str, float]


class TrainingRuntime(Protocol):
    def update_policy(
        self,
        groups: tuple[TrajectoryGroup, ...],
        advantages: tuple[tuple[float, ...], ...],
        *,
        global_update: int,
    ) -> Mapping[str, float]: ...

    def update_auxiliary(
        self,
        groups: tuple[TrajectoryGroup, ...],
        advantages: tuple[tuple[float, ...], ...],
        *,
        global_update: int,
    ) -> Mapping[str, float]: ...

    def synchronize_rollout_weights(self) -> None: ...


class InfoSkillTrainer:
    """Own the method loop; the distributed runtime owns only tensor operations."""

    def __init__(
        self,
        *,
        collector: TrajectoryCollector,
        runtime: TrainingRuntime,
        schedule: TaskSchedule,
        task_groups_per_update: int,
        rollouts_per_task: int,
        master_seed: int,
        auxiliary_enabled: bool,
        on_update: Callable[[UpdateMetrics, tuple[TrajectoryGroup, ...]], None] | None = None,
        on_evaluate: Callable[[int], None] | None = None,
        on_checkpoint: Callable[[int, TaskSchedule], None] | None = None,
        evaluate_every: int = 25,
        checkpoint_every: int = 5,
    ) -> None:
        self.collector = collector
        self.runtime = runtime
        self.schedule = schedule
        self.task_groups_per_update = task_groups_per_update
        self.rollouts_per_task = rollouts_per_task
        self.master_seed = master_seed
        self.auxiliary_enabled = auxiliary_enabled
        self.on_update = on_update
        self.on_evaluate = on_evaluate
        self.on_checkpoint = on_checkpoint
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every
        self.global_update = 0

    def restore(self, *, global_update: int, schedule_state: TaskScheduleState) -> None:
        if global_update < 0:
            raise ValueError("global_update must be non-negative")
        self.schedule.restore(schedule_state)
        self.global_update = global_update

    def fit(self, *, max_updates: int | None = None, evaluate_at_start: bool = True) -> None:
        initial_update = self.global_update
        last_checkpoint_update: int | None = None
        if evaluate_at_start and self.on_evaluate:
            self.on_evaluate(self.global_update)
        while not self.schedule.exhausted:
            if max_updates is not None and self.global_update >= max_updates:
                break
            tasks = self.schedule.next_batch(self.task_groups_per_update)
            if not tasks:
                break
            groups = self.collector.collect_task_groups(
                tasks,
                rollouts_per_task=self.rollouts_per_task,
                master_seed=self.master_seed,
                global_update=self.global_update,
            )
            advantages = tuple(
                group_relative_advantages([trajectory.reward for trajectory in group.trajectories])
                for group in groups
            )
            values = dict(
                self.runtime.update_policy(groups, advantages, global_update=self.global_update)
            )
            if self.auxiliary_enabled:
                values.update(
                    self.runtime.update_auxiliary(
                        groups, advantages, global_update=self.global_update
                    )
                )
            self.runtime.synchronize_rollout_weights()
            self.global_update += 1
            values.update(_rollout_metrics(groups))
            if self.on_update:
                self.on_update(UpdateMetrics(self.global_update, values), groups)
            if self.on_checkpoint and self.global_update % self.checkpoint_every == 0:
                self.on_checkpoint(self.global_update, self.schedule)
                last_checkpoint_update = self.global_update
            if self.on_evaluate and self.global_update % self.evaluate_every == 0:
                self.on_evaluate(self.global_update)

        if (
            self.on_checkpoint
            and self.global_update > initial_update
            and last_checkpoint_update != self.global_update
        ):
            self.on_checkpoint(self.global_update, self.schedule)
        if (
            self.on_evaluate
            and self.global_update > initial_update
            and self.global_update % self.evaluate_every != 0
        ):
            self.on_evaluate(self.global_update)


def _rollout_metrics(groups: Sequence[TrajectoryGroup]) -> dict[str, float]:
    trajectories = [trajectory for group in groups for trajectory in group.trajectories]
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    return {
        "rollout/tasks": float(len(groups)),
        "rollout/trajectories": float(len(trajectories)),
        "rollout/success_rate": sum(trajectory.won for trajectory in trajectories) / len(trajectories),
        "rollout/mean_reward": sum(trajectory.reward for trajectory in trajectories) / len(trajectories),
        "rollout/invalid_action_rate": (
            sum(not step.action.is_executable for step in steps) / len(steps) if steps else 0.0
        ),
        "rollout/mean_steps": sum(len(trajectory.steps) for trajectory in trajectories) / len(trajectories),
    }
