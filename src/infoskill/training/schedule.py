from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from infoskill.episode import TaskSpec


@dataclass(frozen=True, slots=True)
class TaskScheduleState:
    cursor: int
    ordered_task_ids: tuple[str, ...]


class TaskSchedule:
    """World-size-independent, reproducible full-pass task order."""

    def __init__(self, tasks: Sequence[TaskSpec], *, master_seed: int, passes: int = 1) -> None:
        if not tasks or passes <= 0:
            raise ValueError("task schedule requires tasks and positive passes")
        by_id = {task.task_id: task for task in tasks}
        if len(by_id) != len(tasks):
            raise ValueError("task schedule contains duplicate task IDs")
        ordered_ids: list[str] = []
        for pass_index in range(passes):
            ordered_ids.extend(
                task.task_id
                for task in sorted(
                    tasks,
                    key=lambda task: _order_key(master_seed, pass_index, task.task_id),
                )
            )
        self._tasks = by_id
        self._ordered_ids = tuple(ordered_ids)
        self._cursor = 0

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._ordered_ids)

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def total(self) -> int:
        return len(self._ordered_ids)

    def next_batch(self, size: int) -> tuple[TaskSpec, ...]:
        if size <= 0:
            raise ValueError("batch size must be positive")
        end = min(self._cursor + size, len(self._ordered_ids))
        batch = tuple(self._tasks[task_id] for task_id in self._ordered_ids[self._cursor : end])
        self._cursor = end
        return batch

    def state(self) -> TaskScheduleState:
        return TaskScheduleState(self._cursor, self._ordered_ids)

    def restore(self, state: TaskScheduleState) -> None:
        if state.ordered_task_ids != self._ordered_ids:
            raise ValueError("checkpoint task order differs from the resolved schedule")
        if not 0 <= state.cursor <= len(self._ordered_ids):
            raise ValueError("checkpoint task cursor is out of range")
        self._cursor = state.cursor


def _order_key(master_seed: int, pass_index: int, task_id: str) -> bytes:
    return hashlib.sha256(f"task_order|{master_seed}|{pass_index}|{task_id}".encode("utf-8")).digest()
