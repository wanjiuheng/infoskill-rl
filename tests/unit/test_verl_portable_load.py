from __future__ import annotations

import unittest
from pathlib import Path

from infoskill.integrations.verl.portable_load import (
    load_actor_warmstart_after_base_sync,
    load_portable_state_after_base_sync,
)


class _WorkerGroup:
    def __init__(self, reports, load_reports=None):
        self.reports = reports
        self.load_reports = load_reports
        self.events: list[str] = []

    def prepare_infoskill_portable_checkpoint_load(self):
        self.events.append("prepare-base")
        return self.reports

    def load_portable_checkpoint(
        self,
        path: str,
        actor_learning_rate_override: float | None = None,
    ):
        self.events.append(
            f"load:{Path(path).name}:lr={actor_learning_rate_override}"
        )
        return self.load_reports

    def load_actor_warmstart_adapter(self, path: str):
        self.events.append(f"warmstart:{Path(path).name}")
        return self.load_reports


class VerlPortableLoadTests(unittest.TestCase):
    def test_warmstart_loads_only_lora_after_base_sync(self) -> None:
        workers = _WorkerGroup(
            [{"rank": 0, "base_sync_done_after": True}],
            [{
                "rank": 0,
                "lora_state_loaded": True,
                "optimizer_state_loaded": False,
                "infoskill_state_loaded": False,
            }],
        )
        reports = load_actor_warmstart_after_base_sync(
            worker_group=workers,
            adapter_directory=Path("handoffs/m1"),
        )
        self.assertEqual(workers.events, ["prepare-base", "warmstart:m1"])
        self.assertFalse(reports[0]["optimizer_state_loaded"])

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
            ],
            [
                {
                    "rank": 0,
                    "global_step": 7,
                    "infoskill_state_loaded": True,
                    "actor_learning_rate": 3e-6,
                },
                {
                    "rank": 1,
                    "global_step": 7,
                    "infoskill_state_loaded": True,
                    "actor_learning_rate": 3e-6,
                },
            ],
        )

        reports = load_portable_state_after_base_sync(
            worker_group=workers,
            actor_directory=Path("runtime/actor"),
            actor_learning_rate_override=3e-6,
        )

        self.assertEqual(
            workers.events,
            ["prepare-base", "load:actor:lr=3e-06"],
        )
        self.assertEqual(len(reports), 2)
        self.assertTrue(reports[0]["infoskill_state_loaded"])
        self.assertEqual(reports[1]["global_step"], 7)

    def test_learning_rate_override_must_apply_on_every_rank(self) -> None:
        workers = _WorkerGroup(
            [{"rank": 0, "base_sync_done_after": True}],
            [{"rank": 0, "global_step": 7, "actor_learning_rate": 1e-6}],
        )

        with self.assertRaisesRegex(RuntimeError, "did not apply"):
            load_portable_state_after_base_sync(
                worker_group=workers,
                actor_directory=Path("runtime/actor"),
                actor_learning_rate_override=3e-6,
            )

    def test_checkpoint_load_requires_one_matching_report_per_rank(self) -> None:
        workers = _WorkerGroup(
            [{"rank": 0, "base_sync_done_after": True}],
            [{"rank": 1, "global_step": 7}],
        )

        with self.assertRaisesRegex(RuntimeError, "rank set differs"):
            load_portable_state_after_base_sync(
                worker_group=workers,
                actor_directory=Path("runtime/actor"),
            )

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
