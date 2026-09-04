from __future__ import annotations

import unittest

from infoskill.episode import TaskSpec
from infoskill.training import TaskSchedule


class TaskScheduleTests(unittest.TestCase):
    def test_resume_preserves_order_and_last_partial_batch(self) -> None:
        tasks = [TaskSpec(str(index), "train", "kind", "goal") for index in range(5)]
        first = TaskSchedule(tasks, master_seed=0)
        first_batch = first.next_batch(3)
        state = first.state()
        second = TaskSchedule(tasks, master_seed=0)
        second.restore(state)

        last_batch = second.next_batch(3)

        self.assertEqual(len(first_batch), 3)
        self.assertEqual(len(last_batch), 2)
        self.assertTrue(second.exhausted)
        self.assertEqual(
            {task.task_id for task in first_batch + last_batch},
            {task.task_id for task in tasks},
        )


if __name__ == "__main__":
    unittest.main()
