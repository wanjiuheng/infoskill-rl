from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from infoskill.config import EvaluationConfig
from infoskill.episode import TaskSpec, TrajectoryCollector, TrajectoryGroup

from .metrics import EpisodeEvaluation, EvaluationSummary, aggregate_valid_seen


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    records: tuple[EpisodeEvaluation, ...]
    summary: EvaluationSummary
    groups: tuple[TrajectoryGroup, ...] = ()


class EvaluationRunner:
    def __init__(
        self,
        *,
        collector_factory: Callable[[], TrajectoryCollector],
        config: EvaluationConfig,
        task_batch_size: int = 8,
        master_seed: int = 0,
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        self.collector_factory = collector_factory
        self.config = config
        self.task_batch_size = task_batch_size
        self.master_seed = master_seed
        self.on_progress = on_progress

    def run(self, tasks: Sequence[TaskSpec], *, checkpoint_step: int = 0) -> EvaluationRun:
        records: list[EpisodeEvaluation] = []
        all_groups: list[TrajectoryGroup] = []
        for start in range(0, len(tasks), self.task_batch_size):
            batch = tuple(tasks[start : start + self.task_batch_size])
            batch_records, groups = self._run_batch(batch, checkpoint_step=checkpoint_step)
            records.extend(batch_records)
            all_groups.extend(groups)
            if self.on_progress:
                self.on_progress(len(batch))
        frozen = tuple(records)
        return EvaluationRun(
            frozen,
            aggregate_valid_seen(frozen, config=self.config),
            tuple(all_groups),
        )

    def _run_batch(
        self, tasks: tuple[TaskSpec, ...], *, checkpoint_step: int
    ) -> tuple[list[EpisodeEvaluation], tuple[TrajectoryGroup, ...]]:
        last_error: Exception | None = None
        for _ in range(self.config.infrastructure_retries + 1):
            collector = self.collector_factory()
            try:
                groups = collector.collect_task_groups(
                    tasks,
                    rollouts_per_task=1,
                    master_seed=self.master_seed,
                    global_update=checkpoint_step,
                )
                return [
                    EpisodeEvaluation(
                        task_id=group.task.task_id,
                        task_type=group.task.task_type,
                        won=group.trajectories[0].won,
                        steps=len(group.trajectories[0].steps),
                        invalid_action_count=group.trajectories[0].invalid_action_count,
                    )
                    for group in groups
                ], groups
            except Exception as error:
                last_error = error
        assert last_error is not None
        return [
            EpisodeEvaluation(
                task_id=task.task_id,
                task_type=task.task_type,
                won=False,
                steps=0,
                invalid_action_count=0,
                infrastructure_error=f"{type(last_error).__name__}: {last_error}",
            )
            for task in tasks
        ], ()
