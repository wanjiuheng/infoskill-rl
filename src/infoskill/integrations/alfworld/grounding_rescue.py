from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .expert_replay import ExpertReplayResult
from .grounding_shards import GroundingWorkItem


_TIMEOUT_REASON = "expert_wall_timeout"


@dataclass(frozen=True, slots=True)
class GroundingRescueMerge:
    results: tuple[tuple[str, ExpertReplayResult], ...]
    attempted_task_ids: tuple[str, ...]
    rescued_task_ids: tuple[str, ...]
    remaining_timeout_task_ids: tuple[str, ...]


def select_timeout_work_items(
    work_items: Sequence[GroundingWorkItem],
    source_results: Sequence[tuple[str, ExpertReplayResult]],
) -> tuple[GroundingWorkItem, ...]:
    """Select source timeout tasks while preserving the formal task order."""

    _validate_parallel_task_order(work_items, source_results)
    return tuple(
        item
        for item, (_, result) in zip(work_items, source_results, strict=True)
        if not result.succeeded and result.quarantine_reason == _TIMEOUT_REASON
    )


def merge_timeout_grounding_results(
    source_results: Sequence[tuple[str, ExpertReplayResult]],
    rescue_results: Sequence[tuple[str, ExpertReplayResult]],
) -> GroundingRescueMerge:
    """Replace only successful retries of source timeout rows."""

    source_by_id: dict[str, tuple[str, ExpertReplayResult]] = {}
    for task_type, result in source_results:
        if result.task_id in source_by_id:
            raise ValueError(f"duplicate source task: {result.task_id}")
        source_by_id[result.task_id] = (task_type, result)

    rescue_by_id: dict[str, tuple[str, ExpertReplayResult]] = {}
    for task_type, result in rescue_results:
        if result.task_id in rescue_by_id:
            raise ValueError(f"duplicate rescue task: {result.task_id}")
        source = source_by_id.get(result.task_id)
        if source is None:
            raise ValueError(f"rescue targets unknown source task: {result.task_id}")
        source_type, source_result = source
        if source_type != task_type:
            raise ValueError(f"rescue changed task type: {result.task_id}")
        if (
            source_result.succeeded
            or source_result.quarantine_reason != _TIMEOUT_REASON
        ):
            raise ValueError(
                f"rescue target is not an expert timeout: {result.task_id}"
            )
        rescue_by_id[result.task_id] = (task_type, result)

    merged: list[tuple[str, ExpertReplayResult]] = []
    rescued: list[str] = []
    remaining: list[str] = []
    for task_type, source_result in source_results:
        candidate = rescue_by_id.get(source_result.task_id)
        if candidate is not None and candidate[1].succeeded:
            merged.append(candidate)
            rescued.append(source_result.task_id)
        else:
            merged.append((task_type, source_result))
        if (
            not merged[-1][1].succeeded
            and merged[-1][1].quarantine_reason == _TIMEOUT_REASON
        ):
            remaining.append(source_result.task_id)

    return GroundingRescueMerge(
        results=tuple(merged),
        attempted_task_ids=tuple(rescue_by_id),
        rescued_task_ids=tuple(rescued),
        remaining_timeout_task_ids=tuple(remaining),
    )


def _validate_parallel_task_order(
    work_items: Sequence[GroundingWorkItem],
    source_results: Sequence[tuple[str, ExpertReplayResult]],
) -> None:
    work_ids = [item.task.task_id for item in work_items]
    result_ids = [result.task_id for _, result in source_results]
    if work_ids != result_ids:
        raise ValueError(
            "grounding work items and source results differ in task count or order"
        )
    for item, (task_type, _) in zip(work_items, source_results, strict=True):
        if item.task.task_type != task_type:
            raise ValueError(
                f"grounding source changed task type: {item.task.task_id}"
            )
