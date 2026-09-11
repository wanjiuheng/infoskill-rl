#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from infoskill.app_config import AppConfig
from infoskill.cli import _stable_seed
from infoskill.integrations.alfworld import (
    AlfworldEnvironmentFactory,
    StrictExpertReplay,
    discover_tasks,
    load_handcoded_expert,
)

try:
    import resource
except ImportError:  # pragma: no cover - Linux server diagnostic dependency.
    resource = None  # type: ignore[assignment]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a bounded train prefix and stop at the first grounding "
            "OSError while recording process-lifecycle diagnostics."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--output", required=True)
    return parser


def _registry_size() -> int | None:
    try:
        import gym

        registry = gym.envs.registry
        if hasattr(registry, "env_specs"):
            registry = registry.env_specs
        return len(registry)
    except (AttributeError, ImportError, TypeError):
        return None


def _resource_snapshot(*, processed: int) -> dict[str, int | float | None]:
    try:
        fd_count = len(os.listdir("/proc/self/fd"))
    except OSError:
        fd_count = None
    try:
        children = Path(
            f"/proc/{os.getpid()}/task/{os.getpid()}/children"
        ).read_text(encoding="utf-8")
        child_process_count = len(children.split())
    except OSError:
        child_process_count = None
    maximum_rss_mib = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if resource is not None
        else None
    )
    return {
        "processed": processed,
        "fd_count": fd_count,
        "child_process_count": child_process_count,
        "gym_registry_size": _registry_size(),
        "maximum_rss_mib": maximum_rss_mib,
    }


def _exception_payload(error: Exception, *, stage: str) -> dict[str, object]:
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": " ".join(str(error).split())[:1000],
    }


def main() -> int:
    args = _parser().parse_args()
    if args.limit <= 0 or args.snapshot_interval <= 0:
        raise ValueError("limit and snapshot interval must be positive")

    config = AppConfig.load(args.config)
    tasks = discover_tasks(config.paths.alfworld_data, split="train")
    selected = tasks[: min(args.limit, len(tasks))]
    factory = AlfworldEnvironmentFactory.from_paths(
        alfworld_source=config.paths.alfworld_source,
        config_path=config.paths.alfworld_config,
        data_root=config.paths.alfworld_data,
        max_steps=150,
    )
    expert = load_handcoded_expert(
        alfworld_source=config.paths.alfworld_source,
        max_steps=200,
    )
    replay = StrictExpertReplay(
        max_replay_steps=150,
        persist_horizon=config.max_steps,
    )

    reason_counts: Counter[str] = Counter()
    snapshots = [_resource_snapshot(processed=0)]
    first_oserror: dict[str, Any] | None = None
    processed = 0
    for task in tqdm(
        selected,
        desc="grounding/lifecycle-diagnostic",
        unit="task",
        dynamic_ncols=True,
    ):
        try:
            environment = factory.create(
                task,
                rollout_id=0,
                seed=_stable_seed(config.master_seed, task.task_id),
            )
        except Exception as error:
            reason = f"factory_exception:{type(error).__name__}"
            reason_counts[reason] += 1
            processed += 1
            if isinstance(error, OSError):
                first_oserror = {
                    "task_index": processed - 1,
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "total_steps": 0,
                    "exception": _exception_payload(error, stage="factory_create"),
                }
        else:
            try:
                result = replay.run(
                    task=task,
                    environment=environment,
                    expert=expert,
                )
            except Exception as error:
                reason = f"replay_outer_exception:{type(error).__name__}"
                reason_counts[reason] += 1
                processed += 1
                if isinstance(error, OSError):
                    first_oserror = {
                        "task_index": processed - 1,
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "total_steps": 0,
                        "exception": _exception_payload(
                            error,
                            stage="replay_or_environment_close",
                        ),
                    }
            else:
                reason = result.quarantine_reason or "success"
                reason_counts[reason] += 1
                processed += 1
                if result.exception_type == "OSError":
                    first_oserror = {
                        "task_index": processed - 1,
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "total_steps": result.total_steps,
                        "exception": {
                            "stage": result.exception_stage,
                            "type": result.exception_type,
                            "message": result.exception_message,
                        },
                    }

        if processed % args.snapshot_interval == 0 or first_oserror is not None:
            snapshots.append(_resource_snapshot(processed=processed))
        if first_oserror is not None:
            break

    if snapshots[-1]["processed"] != processed:
        snapshots.append(_resource_snapshot(processed=processed))
    report = {
        "schema_version": 1,
        "requested_limit": args.limit,
        "available_tasks": len(tasks),
        "processed_tasks": processed,
        "oserror_reproduced": first_oserror is not None,
        "first_oserror": first_oserror,
        "reason_counts": dict(sorted(reason_counts.items())),
        "resource_snapshots": snapshots,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
