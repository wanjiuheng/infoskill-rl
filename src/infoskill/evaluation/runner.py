from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from infoskill.config import EvaluationConfig
from infoskill.episode import TaskSpec, TrajectoryCollector, TrajectoryGroup
from infoskill.rollout import PromptLengthError

from .metrics import EpisodeEvaluation, EvaluationSummary, aggregate_valid_seen


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    records: tuple[EpisodeEvaluation, ...]
    summary: EvaluationSummary
    groups: tuple[TrajectoryGroup, ...] = ()
    performance_metrics: Mapping[str, float] | None = None


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
        # The checkpoint step is an output label, not part of evaluation
        # randomness. Every checkpoint must see the same environment stream.
        del checkpoint_step
        records: list[EpisodeEvaluation] = []
        all_groups: list[TrajectoryGroup] = []
        performance_metrics: dict[str, float] = {}
        for start in range(0, len(tasks), self.task_batch_size):
            batch = tuple(tasks[start : start + self.task_batch_size])
            batch_records, groups, batch_metrics = self._run_batch(batch)
            records.extend(batch_records)
            all_groups.extend(groups)
            _merge_performance_metrics(performance_metrics, batch_metrics)
            if self.on_progress:
                self.on_progress(len(batch))
        frozen = tuple(records)
        return EvaluationRun(
            frozen,
            aggregate_valid_seen(frozen, config=self.config),
            tuple(all_groups),
            performance_metrics,
        )

    def _run_batch(
        self, tasks: tuple[TaskSpec, ...]
    ) -> tuple[
        list[EpisodeEvaluation],
        tuple[TrajectoryGroup, ...],
        Mapping[str, float],
    ]:
        last_error: Exception | None = None
        for _ in range(self.config.infrastructure_retries + 1):
            collector = self.collector_factory()
            try:
                groups = collector.collect_task_groups(
                    tasks,
                    rollouts_per_task=1,
                    master_seed=self.master_seed,
                    global_update=0,
                )
                records = [
                    EpisodeEvaluation(
                        task_id=group.task.task_id,
                        task_type=group.task.task_type,
                        won=group.trajectories[0].won,
                        steps=len(group.trajectories[0].steps),
                        invalid_action_count=group.trajectories[0].invalid_action_count,
                    )
                    for group in groups
                ]
                metrics_factory = getattr(collector, "performance_metrics", None)
                metrics = metrics_factory() if metrics_factory is not None else {}
                return records, groups, metrics
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
                infrastructure_detail=(
                    last_error.as_dict()
                    if isinstance(last_error, PromptLengthError)
                    else None
                ),
            )
            for task in tasks
        ], (), {}


_MAXIMUM_PERFORMANCE_METRICS = {
    "perf/environment_workers",
    "perf/native_environment_batch",
}


def _merge_performance_metrics(
    aggregate: dict[str, float],
    batch: Mapping[str, float],
) -> None:
    for key, value in batch.items():
        if key in _MAXIMUM_PERFORMANCE_METRICS:
            aggregate[key] = max(aggregate.get(key, value), value)
        else:
            aggregate[key] = aggregate.get(key, 0.0) + value
