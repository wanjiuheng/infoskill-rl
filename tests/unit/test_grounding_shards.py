from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from infoskill.episode import TaskSpec
from infoskill.integrations.alfworld import (
    ExpertReplayResult,
    GroundingWorkItem,
    run_bounded_grounding,
)
from infoskill.integrations.alfworld.grounding_shards import _run_worker_subprocess


class GroundingShardTests(unittest.TestCase):
    def test_subprocess_is_terminated_when_runtime_disk_guard_fires(self) -> None:
        class SlowStdout:
            def __init__(self, process) -> None:
                self._process = process

            def __iter__(self):
                while not self._process.terminated:
                    time.sleep(0.002)
                    yield ""

        class FakeProcess:
            def __init__(self) -> None:
                self.terminated = False
                self.stdout = SlowStdout(self)

            def poll(self):
                return -15 if self.terminated else None

            def terminate(self) -> None:
                self.terminated = True

            def wait(self, timeout=None):
                del timeout
                return -15

            def kill(self) -> None:
                self.terminated = True

        process = FakeProcess()
        item = GroundingWorkItem(
            task=TaskSpec(
                task_id="task-0",
                split="train",
                task_type="pick_and_place_simple",
                goal="put an object somewhere",
            ),
            candidate_skill_ids=(),
            seed=0,
        )

        def low_disk(_count: int) -> None:
            raise RuntimeError("runtime disk fuse fired")

        with tempfile.TemporaryDirectory() as temporary, patch(
            "infoskill.integrations.alfworld.grounding_shards.subprocess.Popen",
            return_value=process,
        ), patch(
            "infoskill.integrations.alfworld.grounding_shards."
            "_DISK_MONITOR_INTERVAL_SECONDS",
            0.001,
        ):
            with self.assertRaisesRegex(RuntimeError, "runtime disk fuse fired"):
                _run_worker_subprocess(
                    (item,),
                    Path(temporary) / "config.yaml",
                    Path(temporary),
                    150,
                    30,
                    "planner",
                    "native_batch",
                    4,
                    low_disk,
                )

        self.assertTrue(process.terminated)

    def test_native_batch_backend_is_explicit_and_reports_environment_slots(self) -> None:
        work_items = tuple(
            GroundingWorkItem(
                task=TaskSpec(
                    task_id=f"task-{index}",
                    split="train",
                    task_type="pick_and_place_simple",
                    goal="put an object somewhere",
                ),
                candidate_skill_ids=(),
                seed=index,
            )
            for index in range(4)
        )

        def fake_worker(
            items,
            config_path,
            temporary,
            max_steps,
            horizon,
            expert_type,
            replay_backend,
            native_batch_size,
            callback,
        ):
            del config_path, temporary, max_steps, horizon
            self.assertEqual(expert_type, "planner")
            self.assertEqual(replay_backend, "native_batch")
            self.assertEqual(native_batch_size, 4)
            if callback is not None:
                callback(len(items))
            return [
                (
                    item.task.task_type,
                    ExpertReplayResult(item.task.task_id, True, (), 1, None),
                )
                for item in items
            ]

        with tempfile.TemporaryDirectory() as temporary:
            results, report = run_bounded_grounding(
                work_items=work_items,
                config_path=Path(temporary) / "config.yaml",
                run_directory=temporary,
                worker_batch_size=4,
                worker_processes=1,
                max_replay_steps=150,
                persist_horizon=30,
                expert_type="planner",
                replay_backend="native_batch",
                native_batch_size=4,
                worker_runner=fake_worker,
            )

        self.assertEqual(len(results), 4)
        self.assertEqual(report.replay_backend, "native_batch")
        self.assertEqual(report.native_batch_size, 4)
        self.assertEqual(report.peak_environment_slots, 4)

    def test_restarts_workers_without_changing_task_order(self) -> None:
        work_items = tuple(
            GroundingWorkItem(
                task=TaskSpec(
                    task_id=f"task-{index}",
                    split="train",
                    task_type="pick_and_place_simple",
                    goal="put an object somewhere",
                ),
                candidate_skill_ids=("general-1",),
                seed=index,
            )
            for index in range(5)
        )
        observed_chunks: list[tuple[str, ...]] = []
        observed_expert_types: list[str] = []
        temporary_paths: list[Path] = []
        progress = 0

        def fake_worker(
            items,
            config_path,
            temporary,
            max_steps,
            horizon,
            expert_type,
            replay_backend,
            native_batch_size,
            callback,
        ):
            del config_path, max_steps, horizon
            observed_chunks.append(tuple(item.task.task_id for item in items))
            observed_expert_types.append(expert_type)
            self.assertEqual(replay_backend, "individual")
            self.assertEqual(native_batch_size, 1)
            temporary_paths.append(temporary)
            self.assertTrue(temporary.is_dir())
            if callback is not None:
                callback(len(items))
            return [
                (
                    item.task.task_type,
                    ExpertReplayResult(item.task.task_id, True, (), 1, None),
                )
                for item in items
            ]

        def update(count: int) -> None:
            nonlocal progress
            progress += count

        with tempfile.TemporaryDirectory() as temporary:
            results, report = run_bounded_grounding(
                work_items=work_items,
                config_path=Path(temporary) / "config.yaml",
                run_directory=temporary,
                worker_batch_size=2,
                max_replay_steps=150,
                persist_horizon=30,
                expert_type="planner",
                on_progress=update,
                worker_runner=fake_worker,
            )

        self.assertEqual(
            observed_chunks,
            [("task-0", "task-1"), ("task-2", "task-3"), ("task-4",)],
        )
        self.assertEqual([result.task_id for _, result in results], [
            "task-0", "task-1", "task-2", "task-3", "task-4"
        ])
        self.assertEqual(progress, 5)
        self.assertEqual(report.worker_processes_started, 3)
        self.assertEqual(report.worker_concurrency, 1)
        self.assertEqual(report.peak_worker_processes, 1)
        self.assertEqual(report.processed_tasks, 5)
        self.assertEqual(report.expert_type, "planner")
        self.assertEqual(observed_expert_types, ["planner", "planner", "planner"])
        self.assertTrue(report.temporary_directories_cleaned)
        self.assertTrue(all(not path.exists() for path in temporary_paths))

    def test_parallel_workers_finish_out_of_order_but_results_stay_ordered(self) -> None:
        work_items = tuple(
            GroundingWorkItem(
                task=TaskSpec(
                    task_id=f"task-{index}",
                    split="train",
                    task_type="pick_and_place_simple",
                    goal="put an object somewhere",
                ),
                candidate_skill_ids=("general-1",),
                seed=index,
            )
            for index in range(4)
        )
        state_lock = threading.Lock()
        active = 0
        peak = 0
        temporary_paths: list[Path] = []

        def fake_worker(
            items,
            config_path,
            temporary,
            max_steps,
            horizon,
            expert_type,
            replay_backend,
            native_batch_size,
            callback,
        ):
            nonlocal active, peak
            del config_path, max_steps, horizon
            self.assertEqual(expert_type, "planner")
            self.assertEqual(replay_backend, "individual")
            self.assertEqual(native_batch_size, 1)
            with state_lock:
                active += 1
                peak = max(peak, active)
                temporary_paths.append(temporary)
            first_index = int(items[0].task.task_id.rsplit("-", 1)[1])
            time.sleep(0.05 if first_index == 0 else 0.01)
            if callback is not None:
                callback(len(items))
            with state_lock:
                active -= 1
            return [
                (
                    item.task.task_type,
                    ExpertReplayResult(item.task.task_id, True, (), 1, None),
                )
                for item in items
            ]

        with tempfile.TemporaryDirectory() as temporary:
            results, report = run_bounded_grounding(
                work_items=work_items,
                config_path=Path(temporary) / "config.yaml",
                run_directory=temporary,
                worker_batch_size=2,
                worker_processes=2,
                max_replay_steps=150,
                persist_horizon=30,
                expert_type="planner",
                worker_runner=fake_worker,
            )

        self.assertEqual(peak, 2)
        self.assertEqual(
            [result.task_id for _, result in results],
            ["task-0", "task-1", "task-2", "task-3"],
        )
        self.assertEqual(report.worker_concurrency, 2)
        self.assertEqual(report.peak_worker_processes, 2)
        self.assertTrue(report.temporary_directories_cleaned)
        self.assertTrue(all(not path.exists() for path in temporary_paths))


if __name__ == "__main__":
    unittest.main()
