from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable, Sequence

from infoskill.episode import TaskSpec

from .expert_replay import ExpertReplayResult
from .grounding_io import grounding_result_payload, read_grounding_results


_PROGRESS_MARKER = "INFO_SKILL_GROUNDING_PROGRESS"
_MINIMUM_FREE_DISK_BYTES = 4 * 1024**3
_PARALLEL_WORKER_DISK_RESERVE_BYTES = 3 * 1024**3
_DISK_MONITOR_INTERVAL_SECONDS = 0.5
_DEFAULT_WORKER_INACTIVITY_TIMEOUT_SECONDS = 300.0
_PROCESS_TERMINATION_GRACE_SECONDS = 2.0
_RESUME_SCHEMA_VERSION = 1


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
    completed_shards_reused: int = 0
    timeout_fallback_shards: int = 0
    timed_out_tasks: tuple[str, ...] = ()
    worker_inactivity_timeout_seconds: float = (
        _DEFAULT_WORKER_INACTIVITY_TIMEOUT_SECONDS
    )


class GroundingWorkerInactivityTimeout(RuntimeError):
    """A bounded worker made no task-level progress before its wall timeout."""

    def __init__(
        self,
        message: str,
        *,
        partial_results: Sequence[tuple[str, ExpertReplayResult]] = (),
    ) -> None:
        super().__init__(message)
        self.partial_results = tuple(partial_results)


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
    worker_inactivity_timeout_seconds: float = (
        _DEFAULT_WORKER_INACTIVITY_TIMEOUT_SECONDS
    ),
    source_checksum: str | None = None,
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
    if worker_inactivity_timeout_seconds <= 0:
        raise ValueError("grounding worker inactivity timeout must be positive")
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
    resume_plan = _ensure_resume_plan(
        destination=destination,
        work_items=work_items,
        config_path=resolved_config,
        worker_batch_size=worker_batch_size,
        worker_processes=worker_processes,
        max_replay_steps=max_replay_steps,
        persist_horizon=persist_horizon,
        expert_type=expert_type,
        replay_backend=replay_backend,
        native_batch_size=native_batch_size,
        worker_inactivity_timeout_seconds=worker_inactivity_timeout_seconds,
        source_checksum=source_checksum,
    )
    pending_chunks: list[
        tuple[int, int, Sequence[GroundingWorkItem]]
    ] = []
    outcomes: dict[
        int,
        tuple[list[tuple[str, ExpertReplayResult]], dict[str, object]],
    ] = {}
    completed_shards_reused = 0
    for shard_index, start, chunk in chunks:
        cached = _load_completed_shard(
            destination=destination,
            shard_index=shard_index,
            start_index=start,
            chunk=chunk,
            resume_plan_sha256=resume_plan["plan_sha256"],
        )
        if cached is None:
            pending_chunks.append((shard_index, start, chunk))
            continue
        outcomes[shard_index] = cached
        completed_shards_reused += 1
        if on_progress is not None:
            on_progress(len(chunk))

    effective_concurrency = min(worker_processes, len(pending_chunks))
    temporary_paths: list[Path] = []
    minimum_free = shutil.disk_usage(destination).free
    state_lock = Lock()
    progress_lock = Lock()
    active_processes = 0
    peak_processes = 0
    worker_processes_started = 0
    timeout_fallback_shards = 0
    timed_out_tasks: list[str] = []
    environment_slots_per_worker = (
        native_batch_size if replay_backend == "native_batch" else 1
    )
    required_free_before_parallel_launch = (
        _MINIMUM_FREE_DISK_BYTES
        + max(1, effective_concurrency)
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
        nonlocal worker_processes_started, timeout_fallback_shards
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

                try:
                    with state_lock:
                        worker_processes_started += 1
                    shard_results = _invoke_runner(
                        runner=runner,
                        is_default_runner=worker_runner is None,
                        work_items=chunk,
                        config_path=resolved_config,
                        temporary_directory=temporary_path,
                        max_replay_steps=max_replay_steps,
                        persist_horizon=persist_horizon,
                        expert_type=expert_type,
                        replay_backend=replay_backend,
                        native_batch_size=native_batch_size,
                        on_progress=shard_progress,
                        inactivity_timeout_seconds=(
                            worker_inactivity_timeout_seconds
                        ),
                    )
                except GroundingWorkerInactivityTimeout as error:
                    with state_lock:
                        timeout_fallback_shards += 1
                    shard_results = list(error.partial_results)
                    completed_ids = [
                        result.task_id for _, result in shard_results
                    ]
                    expected_prefix = [
                        item.task.task_id for item in chunk[: len(shard_results)]
                    ]
                    if completed_ids != expected_prefix:
                        raise RuntimeError(
                            "timed-out grounding worker returned a non-prefix "
                            "partial result set"
                        ) from error
                    remaining = chunk[len(shard_results) :]

                    def isolate_remaining_item(
                        indexed_item: tuple[int, GroundingWorkItem],
                    ) -> tuple[int, tuple[str, ExpertReplayResult]]:
                        nonlocal worker_processes_started
                        offset, item = indexed_item
                        try:
                            with state_lock:
                                worker_processes_started += 1
                            with tempfile.TemporaryDirectory(
                                prefix=f"isolated-{offset:04d}-",
                                dir=temporary_path,
                            ) as isolated_temporary:
                                isolated = _invoke_runner(
                                    runner=runner,
                                    is_default_runner=worker_runner is None,
                                    work_items=(item,),
                                    config_path=resolved_config,
                                    temporary_directory=Path(isolated_temporary),
                                    max_replay_steps=max_replay_steps,
                                    persist_horizon=persist_horizon,
                                    expert_type=expert_type,
                                    replay_backend="individual",
                                    native_batch_size=1,
                                    on_progress=shard_progress,
                                    inactivity_timeout_seconds=(
                                        worker_inactivity_timeout_seconds
                                    ),
                                )
                            if len(isolated) != 1:
                                raise RuntimeError(
                                    "isolated grounding retry must return exactly "
                                    "one task"
                                )
                            return offset, isolated[0]
                        except GroundingWorkerInactivityTimeout as isolated_error:
                            partial = tuple(isolated_error.partial_results)
                            if (
                                len(partial) == 1
                                and partial[0][1].task_id == item.task.task_id
                            ):
                                return offset, partial[0]
                            with state_lock:
                                timed_out_tasks.append(item.task.task_id)
                            shard_progress(1)
                            return (
                                offset,
                                (
                                    item.task.task_type,
                                    _timeout_quarantine(
                                        item.task,
                                        isolated_error,
                                    ),
                                ),
                            )

                    indexed_remaining = tuple(enumerate(remaining))
                    fallback_concurrency = min(
                        environment_slots_per_worker,
                        len(indexed_remaining),
                    )
                    isolated_results: dict[
                        int, tuple[str, ExpertReplayResult]
                    ] = {}
                    if fallback_concurrency <= 1:
                        for indexed_item in indexed_remaining:
                            offset, isolated = isolate_remaining_item(indexed_item)
                            isolated_results[offset] = isolated
                    else:
                        with ThreadPoolExecutor(
                            max_workers=fallback_concurrency
                        ) as fallback_executor:
                            fallback_futures = {
                                fallback_executor.submit(
                                    isolate_remaining_item,
                                    indexed_item,
                                ): indexed_item[0]
                                for indexed_item in indexed_remaining
                            }
                            for future in as_completed(fallback_futures):
                                offset, isolated = future.result()
                                isolated_results[offset] = isolated
                    shard_results.extend(
                        isolated_results[offset]
                        for offset in range(len(indexed_remaining))
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
        shard_report = {
            "shard_index": shard_index,
            "start_index": start,
            "task_count": len(chunk),
            "free_disk_bytes_before": free_before,
            "free_disk_bytes_during": free_during,
            "free_disk_bytes_after_cleanup": free_after_cleanup,
            "temporary_directory_cleaned": not temporary_path.exists(),
            "reused": False,
        }
        _commit_completed_shard(
            destination=destination,
            shard_index=shard_index,
            start_index=start,
            chunk=chunk,
            results=shard_results,
            shard_report=shard_report,
            resume_plan_sha256=resume_plan["plan_sha256"],
        )
        return shard_results, shard_report

    if effective_concurrency == 1:
        for shard_index, start, chunk in pending_chunks:
            outcomes[shard_index] = run_shard(shard_index, start, chunk)
    elif effective_concurrency > 1:
        with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
            futures = {
                executor.submit(run_shard, shard_index, start, chunk): shard_index
                for shard_index, start, chunk in pending_chunks
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
        worker_processes_started=worker_processes_started,
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
        completed_shards_reused=completed_shards_reused,
        timeout_fallback_shards=timeout_fallback_shards,
        timed_out_tasks=tuple(sorted(timed_out_tasks)),
        worker_inactivity_timeout_seconds=(
            worker_inactivity_timeout_seconds
        ),
    )
    return results, report


def _invoke_runner(
    *,
    runner: WorkerRunner,
    is_default_runner: bool,
    work_items: Sequence[GroundingWorkItem],
    config_path: Path,
    temporary_directory: Path,
    max_replay_steps: int,
    persist_horizon: int,
    expert_type: str,
    replay_backend: str,
    native_batch_size: int,
    on_progress: Callable[[int], None] | None,
    inactivity_timeout_seconds: float,
) -> list[tuple[str, ExpertReplayResult]]:
    arguments = (
        work_items,
        config_path,
        temporary_directory,
        max_replay_steps,
        persist_horizon,
        expert_type,
        replay_backend,
        native_batch_size,
        on_progress,
    )
    if is_default_runner:
        return _run_worker_subprocess(
            *arguments,
            inactivity_timeout_seconds=inactivity_timeout_seconds,
        )
    return runner(*arguments)


def _timeout_quarantine(
    task: TaskSpec,
    error: GroundingWorkerInactivityTimeout,
) -> ExpertReplayResult:
    return ExpertReplayResult(
        task_id=task.task_id,
        succeeded=False,
        samples=(),
        total_steps=0,
        quarantine_reason="expert_wall_timeout",
        exception_stage="worker_inactivity_timeout",
        exception_type=type(error).__name__,
        exception_message=" ".join(str(error).split())[:1000],
    )


def _work_item_payload(item: GroundingWorkItem) -> dict[str, object]:
    return {
        "task": asdict(item.task),
        "candidate_skill_ids": list(item.candidate_skill_ids),
        "seed": item.seed,
    }


def _canonical_json_lines(payloads: Sequence[dict[str, object]]) -> str:
    return "".join(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for payload in payloads
    )


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _work_items_sha256(work_items: Sequence[GroundingWorkItem]) -> str:
    return _sha256_text(
        _canonical_json_lines([_work_item_payload(item) for item in work_items])
    )


def _ensure_resume_plan(
    *,
    destination: Path,
    work_items: Sequence[GroundingWorkItem],
    config_path: Path,
    worker_batch_size: int,
    worker_processes: int,
    max_replay_steps: int,
    persist_horizon: int,
    expert_type: str,
    replay_backend: str,
    native_batch_size: int,
    worker_inactivity_timeout_seconds: float,
    source_checksum: str | None,
) -> dict[str, object]:
    config_sha256 = None
    if config_path.is_file():
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    payload: dict[str, object] = {
        "schema_version": _RESUME_SCHEMA_VERSION,
        "task_count": len(work_items),
        "work_items_sha256": _work_items_sha256(work_items),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "worker_batch_size": worker_batch_size,
        "worker_processes": worker_processes,
        "max_replay_steps": max_replay_steps,
        "persist_horizon": persist_horizon,
        "expert_type": expert_type,
        "replay_backend": replay_backend,
        "native_batch_size": (
            native_batch_size if replay_backend == "native_batch" else 1
        ),
        "worker_inactivity_timeout_seconds": (
            worker_inactivity_timeout_seconds
        ),
        "source_checksum": source_checksum,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    payload["plan_sha256"] = _sha256_text(canonical)
    path = destination / "grounding-resume.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                "grounding resume plan does not match the existing run; "
                "refusing to mix replay configurations"
            )
        return payload
    _atomic_write_json(path, payload)
    return payload


def _completed_shard_directory(destination: Path, shard_index: int) -> Path:
    return destination / "grounding-shards" / f"shard-{shard_index:04d}"


def _load_completed_shard(
    *,
    destination: Path,
    shard_index: int,
    start_index: int,
    chunk: Sequence[GroundingWorkItem],
    resume_plan_sha256: object,
) -> tuple[list[tuple[str, ExpertReplayResult]], dict[str, object]] | None:
    shard_directory = _completed_shard_directory(destination, shard_index)
    marker_path = shard_directory / "complete.json"
    results_path = shard_directory / "results.jsonl"
    if not marker_path.exists():
        return None
    if not results_path.is_file():
        raise RuntimeError(
            f"completed grounding shard {shard_index} is missing results.jsonl"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_ids = [item.task.task_id for item in chunk]
    checks = {
        "schema_version": marker.get("schema_version") == _RESUME_SCHEMA_VERSION,
        "resume_plan_sha256": (
            marker.get("resume_plan_sha256") == resume_plan_sha256
        ),
        "shard_index": marker.get("shard_index") == shard_index,
        "start_index": marker.get("start_index") == start_index,
        "task_ids": marker.get("task_ids") == expected_ids,
        "work_items_sha256": (
            marker.get("work_items_sha256") == _work_items_sha256(chunk)
        ),
        "results_sha256": (
            marker.get("results_sha256")
            == hashlib.sha256(results_path.read_bytes()).hexdigest()
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"completed grounding shard {shard_index} failed resume "
            f"validation: {failed}"
        )
    results = read_grounding_results(results_path)
    actual_ids = [result.task_id for _, result in results]
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"completed grounding shard {shard_index} changed task order"
        )
    report = marker.get("shard_report")
    if not isinstance(report, dict):
        raise RuntimeError(
            f"completed grounding shard {shard_index} lacks a shard report"
        )
    return results, {**report, "reused": True}


def _commit_completed_shard(
    *,
    destination: Path,
    shard_index: int,
    start_index: int,
    chunk: Sequence[GroundingWorkItem],
    results: Sequence[tuple[str, ExpertReplayResult]],
    shard_report: dict[str, object],
    resume_plan_sha256: object,
) -> None:
    shard_directory = _completed_shard_directory(destination, shard_index)
    shard_directory.mkdir(parents=True, exist_ok=True)
    results_path = shard_directory / "results.jsonl"
    content = _canonical_json_lines(
        [grounding_result_payload(task_type, result) for task_type, result in results]
    )
    _atomic_write_text(results_path, content)
    marker = {
        "schema_version": _RESUME_SCHEMA_VERSION,
        "resume_plan_sha256": resume_plan_sha256,
        "shard_index": shard_index,
        "start_index": start_index,
        "task_ids": [item.task.task_id for item in chunk],
        "work_items_sha256": _work_items_sha256(chunk),
        "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "shard_report": shard_report,
    }
    _atomic_write_json(shard_directory / "complete.json", marker)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


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
    *,
    inactivity_timeout_seconds: float = (
        _DEFAULT_WORKER_INACTIVITY_TIMEOUT_SECONDS
    ),
) -> list[tuple[str, ExpertReplayResult]]:
    if inactivity_timeout_seconds <= 0:
        raise ValueError("grounding worker inactivity timeout must be positive")
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
    progress_lock = Lock()
    last_progress_at = time.monotonic()
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=(os.name == "posix"),
    )

    def monitor_disk() -> None:
        while not monitor_stop.wait(_DISK_MONITOR_INTERVAL_SECONDS):
            try:
                if on_progress is not None:
                    on_progress(0)
            except Exception as error:
                monitor_errors.append(error)
                _terminate_worker_process_tree(process)
                return
            with progress_lock:
                inactive_for = time.monotonic() - last_progress_at
            if inactive_for >= inactivity_timeout_seconds:
                monitor_errors.append(
                    GroundingWorkerInactivityTimeout(
                        "grounding worker made no progress for "
                        f"{inactive_for:.1f} seconds"
                    )
                )
                _terminate_worker_process_tree(process)
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
                with progress_lock:
                    last_progress_at = time.monotonic()
                if on_progress is not None:
                    on_progress(1)
            elif normalized:
                tail.append(normalized)
        return_code = process.wait()
    except BaseException:
        _terminate_worker_process_tree(process)
        raise
    finally:
        monitor_stop.set()
        monitor.join(timeout=2)
    if monitor_errors:
        error = monitor_errors[0]
        if isinstance(error, GroundingWorkerInactivityTimeout):
            partial_results = (
                read_grounding_results(output_path)
                if output_path.is_file()
                else []
            )
            raise GroundingWorkerInactivityTimeout(
                str(error),
                partial_results=partial_results,
            ) from error
        raise error
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


def _terminate_worker_process_tree(process: subprocess.Popen[str]) -> None:
    pid = getattr(process, "pid", None)
    if os.name == "posix" and isinstance(pid, int):
        try:
            # start_new_session=True makes the worker PID its process-group ID.
            # This still reaches descendants if the direct worker has exited
            # while they keep its stdout pipe open.
            os.killpg(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            if process.poll() is None:
                process.terminate()
    else:
        if process.poll() is not None:
            return
        process.terminate()
    if process.poll() is not None:
        return
    try:
        process.wait(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        if os.name == "posix" and isinstance(pid, int):
            with suppress(OSError, ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


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
