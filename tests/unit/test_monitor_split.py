from __future__ import annotations

import unittest

from infoskill.episode import TaskSpec
from infoskill.integrations.alfworld import build_train_monitor_manifest


class TrainMonitorTests(unittest.TestCase):
    def test_hash_split_selects_each_task_type_and_is_order_independent(self) -> None:
        tasks = [
            TaskSpec(f"a-{index}", "train", "a", "goal") for index in range(20)
        ] + [TaskSpec(f"b-{index}", "train", "b", "goal") for index in range(10)]

        first = build_train_monitor_manifest(tasks, master_seed=0)
        second = build_train_monitor_manifest(list(reversed(tasks)), master_seed=0)

        self.assertEqual(first.monitor_task_ids, second.monitor_task_ids)
        self.assertEqual(first.per_task_type_counts, {"a": 2, "b": 1})


if __name__ == "__main__":
    unittest.main()
