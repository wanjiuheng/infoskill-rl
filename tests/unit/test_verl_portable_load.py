from __future__ import annotations

import unittest
from pathlib import Path

from infoskill.integrations.verl.portable_load import (
    load_portable_state_after_base_sync,
)


class _WorkerGroup:
    def __init__(self, reports):
        self.reports = reports
        self.events: list[str] = []

    def prepare_infoskill_portable_checkpoint_load(self):
        self.events.append("prepare-base")
        return self.reports

    def load_portable_checkpoint(self, path: str) -> None:
        self.events.append(f"load:{Path(path).name}")


class VerlPortableLoadTests(unittest.TestCase):
    def test_base_sync_must_complete_before_portable_actor_load(self) -> None:
        workers = _WorkerGroup(
            [
                {
                    "rank": 0,
                    "base_sync_done_before": False,
                    "base_sync_done_after": True,
                    "warmup_performed": True,
                },
                {
                    "rank": 1,
                    "base_sync_done_before": False,
                    "base_sync_done_after": True,
                    "warmup_performed": True,
                },
            ]
        )

        reports = load_portable_state_after_base_sync(
            worker_group=workers,
            actor_directory=Path("runtime/actor"),
        )

        self.assertEqual(workers.events, ["prepare-base", "load:actor"])
        self.assertEqual(len(reports), 2)

    def test_incomplete_base_sync_blocks_checkpoint_load(self) -> None:
        workers = _WorkerGroup(
            [{"rank": 0, "base_sync_done_after": False}]
        )

        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            load_portable_state_after_base_sync(
                worker_group=workers,
                actor_directory=Path("runtime/actor"),
            )

        self.assertEqual(workers.events, ["prepare-base"])


if __name__ == "__main__":
    unittest.main()
