from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable, Sequence

from infoskill.episode import TaskSpec

from .expert_replay import ExpertReplayResult
from .grounding_io import read_grounding_results


_PROGRESS_MARKER = "INFO_SKILL_GROUNDING_PROGRESS"
_MINIMUM_FREE_DISK_BYTES = 4 * 1024**3
_PARALLEL_WORKER_DISK_RESERVE_BYTES = 3 * 1024**3
_DISK_MONITOR_INTERVAL_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class GroundingWorkItem:
    task: TaskSpec
    candidate_skill_ids: tuple[str, ...]
    seed: int


@dataclass(frozen=True, slots=True)
class GroundingShardReport:
    schema_version: int
    expert_type: str
    worker_batch_size: int
    worker_concurrency: int
    peak_worker_processes: int
    worker_processes_started: int
    processed_tasks: int
    minimum_free_disk_bytes: int
    parallel_worker_disk_reserve_bytes: int
    temporary_directories_cleaned: bool
    shards: tuple[dict[str, object], ...]
    replay_backend: str = "individual"
    native_batch_size: int = 1
    peak_environment_slots: int = 1


WorkerRunner = Callable[
    [
        Sequence[GroundingWorkItem],
        Path,
        Path,
        int,
        int,
        str,
        str,
        int,
        Callable[[int], None] | None,
    ],
    list[tuple[str, ExpertReplayResult]],
]


def run_bounded_grounding(
    *,
    work_items: Sequence[GroundingWorkItem],
    config_path: str | Path,
    run_directory: str | Path,
    worker_batch_size: int,
    worker_processes: int = 1,
    max_replay_steps: int,
    persist_horizon: int,
    expert_type: str,
    replay_backend: str = "individual",
    native_batch_size: int = 1,
    on_progress: Callable[[int], None] | None = None,
    worker_runner: WorkerRunner | None = None,
) -> tuple[list[tuple[str, ExpertReplayResult]], GroundingShardReport]:
    """Replay work in short-lived processes so TextWorld resources stay bounded."""

    if not work_items:
        raise ValueError("grounding requires at least one work item")
    if worker_batch_size <= 0:
        raise ValueError("grounding worker batch size must be positive")
    if worker_processes <= 0:
        raise ValueError("grounding worker process count must be positive")
    if max_replay_steps <= 0 or persist_horizon <= 0:
        raise ValueError("grounding replay limits must be positive")
    if expert_type not in {"handcoded", "planner"}:
        raise ValueError("expert_type must be handcoded or planner")
    if replay_backend not in {"individual", "native_batch"}:
        raise ValueError("replay_backend must be individual or native_batch")
    if native_batch_size <= 0:
        raise ValueError("native_batch_size must be positive")
    if replay_backend == "native_batch" and expert_type != "planner":
        raise ValueError("native batch grounding currently requires planner expert")
    if replay_backend == "native_batch" and native_batch_size < 2:
        raise ValueError("native batch grounding requires native_batch_size at least 2")

    destination = Path(run_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    resolved_config = Path(config_path).expanduser().resolve()
    runner = worker_runner or _run_worker_subprocess
    chunks = tuple(
        (
            shard_index,
            start,
            work_items[start : start + worker_batch_size],
        )
        for shard_index, start in enumerate(
            range(0, len(work_items), worker_batch_size),
            start=1,
        )
    )
    effective_concurrency = min(worker_processes, len(chunks))
    outcomes: dict[
        int,
        tuple[list[tuple[str, ExpertReplayResult]], dict[str, object]],
    ] = {}
    temporary_paths: list[Path] = []
    minimum_free = shutil.disk_usage(destination).free
    state_lock = Lock()
    progress_lock = Lock()
    active_processes = 0
    peak_processes = 0
    environment_slots_per_worker = (
        native_batch_size if replay_backend == "native_batch" else 1
    )
    required_free_before_parallel_launch = (
        _MINIMUM_FREE_DISK_BYTES
        + effective_concurrency
        * environment_slots_per_worker
        * _PARALLEL_WORKER_DISK_RESERVE_BYTES
        if effective_concurrency * environment_slots_per_worker > 1
        else _MINIMUM_FREE_DISK_BYTES
    )

    def run_shard(
        shard_index: int,
        start: int,
        chunk: Sequence[GroundingWorkItem],
    ) -> tuple[list[tuple[str, ExpertReplayResult]], dict[str, object]]:
        nonlocal active_processes, minimum_free, peak_processes
        free_before = shutil.disk_usage(destination).free
        with state_lock:
            minimum_free = min(minimum_free, free_before)
        if free_before < required_free_before_parallel_launch:
            raise RuntimeError(
                "grounding stopped before launching worker: free disk space "
                f"{free_before} is below the concurrency-aware requirement "
                f"{required_free_before_parallel_launch} bytes"
            )
        with state_lock:
            active_processes += 1
            peak_processes = max(peak_processes, active_processes)
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"grounding-shard-{shard_index:04d}-",
                dir=destination,
            ) as temporary:
                temporary_path = Path(temporary)
                with state_lock:
                    temporary_paths.append(temporary_path)
                shard_minimum_free = free_before

                def shard_progress(count: int) -> None:
                    nonlocal minimum_free, shard_minimum_free
                    current_free = shutil.disk_usage(destination).free
                    shard_minimum_free = min(shard_minimum_free, current_free)
                    with state_lock:
                        minimum_free = min(minimum_free, current_free)
                    if current_free < _MINIMUM_FREE_DISK_BYTES:
                        raise RuntimeError(
                            "grounding worker stopped because free disk space "
                            f"{current_free} fell below "
                            f"{_MINIMUM_FREE_DISK_BYTES} bytes"
                        )
                    if on_progress is not None:
                        with progress_lock:
                            on_progress(count)

                shard_results = runner(
                    chunk,
                    resolved_config,
                    temporary_path,
                    max_replay_steps,
                    persist_horizon,
                    expert_type,
                    replay_backend,
                    native_batch_size,
                    shard_progress,
                )
                expected_ids = [item.task.task_id for item in chunk]
                actual_ids = [result.task_id for _, result in shard_results]
                if actual_ids != expected_ids:
                    raise RuntimeError(
                        "grounding worker changed task count or order: "
                        f"expected {expected_ids!r}, got {actual_ids!r}"
                    )
                free_during = min(
                    shard_minimum_free,
                    shutil.disk_usage(destination).free,
                )
                with state_lock:
                    minimum_free = min(minimum_free, free_during)
        except Exception as error:
            _write_failure(
                destination
                / f"grounding-worker-failure-shard-{shard_index:04d}.json",
                shard_index=shard_index,
                start_index=start,
                task_ids=[item.task.task_id for item in chunk],
                error=error,
            )
            raise
        finally:
            with state_lock:
                active_processes -= 1

        free_after_cleanup = shutil.disk_usage(destination).free
        with state_lock:
            minimum_free = min(minimum_free, free_after_cleanup)
        return shard_results, {
            "shard_index": shard_index,
            "start_index": start,
            "task_count": len(chunk),
            "free_disk_bytes_before": free_before,
            "free_disk_bytes_during": free_during,
            "free_disk_bytes_after_cleanup": free_after_cleanup,
            "temporary_directory_cleaned": not temporary_path.exists(),
        }

    if effective_concurrency == 1:
        for shard_index, start, chunk in chunks:
            outcomes[shard_index] = run_shard(shard_index, start, chunk)
    else:
        with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
            futures = {
                executor.submit(run_shard, shard_index, start, chunk): shard_index
                for shard_index, start, chunk in chunks
            }
            try:
                for future in as_completed(futures):
                    outcomes[futures[future]] = future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    results: list[tuple[str, ExpertReplayResult]] = []
    shard_reports: list[dict[str, object]] = []
    for shard_index, _, _ in chunks:
        shard_results, shard_report = outcomes[shard_index]
        results.extend(shard_results)
        shard_reports.append(shard_report)

    report = GroundingShardReport(
        schema_version=2,
        expert_type=expert_type,
        worker_batch_size=worker_batch_size,
        worker_concurrency=effective_concurrency,
        peak_worker_processes=peak_processes,
        worker_processes_started=len(shard_reports),
        processed_tasks=len(results),
        minimum_free_disk_bytes=minimum_free,
        parallel_worker_disk_reserve_bytes=(
            _PARALLEL_WORKER_DISK_RESERVE_BYTES
            if effective_concurrency > 1
            else 0
        ),
        temporary_directories_cleaned=all(not path.exists() for path in temporary_paths),
        shards=tuple(shard_reports),
        replay_backend=replay_backend,
        native_batch_size=(
            native_batch_size if replay_backend == "native_batch" else 1
        ),
        peak_environment_slots=(
            peak_processes * environment_slots_per_worker
        ),
    )
    return results, report


def _run_worker_subprocess(
    work_items: Sequence[GroundingWorkItem],
    config_path: Path,
    temporary_directory: Path,
    max_replay_steps: int,
    persist_horizon: int,
    expert_type: str,
    replay_backend: str,
    native_batch_size: int,
    on_progress: Callable[[int], None] | None,
) -> list[tuple[str, ExpertReplayResult]]:
    input_path = temporary_directory / "work-items.jsonl"
    output_path = temporary_directory / "results.jsonl"
    input_path.write_text(
        "".join(
            json.dumps(
                {
                    "task": asdict(item.task),
                    "candidate_skill_ids": list(item.candidate_skill_ids),
                    "seed": item.seed,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for item in work_items
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": str(temporary_directory),
            "TMP": str(temporary_directory),
            "TEMP": str(temporary_directory),
        }
    )
    command = [
        sys.executable,
        "-m",
        "infoskill.integrations.alfworld.grounding_worker",
        "--config",
        str(config_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--max-replay-steps",
        str(max_replay_steps),
        "--persist-horizon",
        str(persist_horizon),
        "--expert-type",
        expert_type,
        "--replay-backend",
        replay_backend,
        "--native-batch-size",
        str(native_batch_size),
    ]
    tail: deque[str] = deque(maxlen=50)
    progress_count = 0
    monitor_stop = Event()
    monitor_errors: list[Exception] = []
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    def monitor_disk() -> None:
        while not monitor_stop.wait(_DISK_MONITOR_INTERVAL_SECONDS):
            if process.poll() is not None:
                return
            try:
                if on_progress is not None:
                    on_progress(0)
            except Exception as error:
                monitor_errors.append(error)
                process.terminate()
                return

    monitor = Thread(
        target=monitor_disk,
        name="grounding-worker-disk-monitor",
        daemon=True,
    )
    monitor.start()
    try:
        assert process.stdout is not None
        for line in process.stdout:
            normalized = line.rstrip()
            if normalized == _PROGRESS_MARKER:
                progress_count += 1
                if on_progress is not None:
                    on_progress(1)
            elif normalized:
                tail.append(normalized)
        return_code = process.wait()
    except BaseException:
        process.terminate()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        monitor_stop.set()
        monitor.join(timeout=2)
    if monitor_errors:
        raise monitor_errors[0]
    if return_code != 0:
        raise RuntimeError(
            "grounding worker process failed with exit code "
            f"{return_code}: {' | '.join(tail)}"
        )
    if progress_count != len(work_items):
        raise RuntimeError(
            "grounding worker progress count mismatch: "
            f"expected {len(work_items)}, got {progress_count}"
        )
    return read_grounding_results(output_path)


def _write_failure(
    path: Path,
    *,
    shard_index: int,
    start_index: int,
    task_ids: Sequence[str],
    error: Exception,
) -> None:
    payload = {
        "schema_version": 1,
        "shard_index": shard_index,
        "start_index": start_index,
        "task_ids": list(task_ids),
        "error_type": type(error).__name__,
        "error_message": " ".join(str(error).split())[:4000],
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
