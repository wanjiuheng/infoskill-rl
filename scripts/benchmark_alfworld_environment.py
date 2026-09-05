#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from infoskill.app_config import AppConfig
from infoskill.domain.state import CanonicalAgentState
from infoskill.episode import EnvironmentTransition, TaskSpec
from infoskill.integrations.alfworld import AlfworldEnvironmentFactory, discover_tasks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare serial batch_size=1 ALFWorld environments with TextWorld's native "
            "multiprocessing batch without loading a policy model."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--rollouts-per-task", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--minimum-speedup",
        type=float,
        help="defaults to no speed gate for smoke and 1.25 for full",
    )
    parser.add_argument("--output")
    return parser


def _profile_values(args: argparse.Namespace) -> tuple[int, int, int, float]:
    defaults = {
        "smoke": (2, 2, 3),
        "full": (8, 8, 30),
    }
    task_count, rollouts_per_task, steps = defaults[args.profile]
    task_count = args.task_count or task_count
    rollouts_per_task = args.rollouts_per_task or rollouts_per_task
    steps = args.steps or steps
    if min(task_count, rollouts_per_task, steps) <= 0:
        raise ValueError("task-count, rollouts-per-task and steps must be positive")
    minimum_speedup = args.minimum_speedup
    if minimum_speedup is None:
        minimum_speedup = 0.0 if args.profile == "smoke" else 1.25
    if minimum_speedup < 0:
        raise ValueError("minimum-speedup cannot be negative")
    return task_count, rollouts_per_task, steps, minimum_speedup


def _seed(master_seed: int, task_id: str, rollout_id: int) -> int:
    material = f"environment|{master_seed}|0|{task_id}|{rollout_id}|0".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)


def _action(state: CanonicalAgentState, *, rollout_id: int, step: int) -> str | None:
    if state.done:
        return None
    commands = state.admissible_commands
    if not commands:
        raise RuntimeError(f"no admissible command for active task {state.task_id}")
    material = f"{state.task_id}|{rollout_id}|{step}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(commands)
    return commands[index]


def _state_payload(state: CanonicalAgentState) -> dict[str, object]:
    return asdict(state)


def _transition_payload(
    transition: EnvironmentTransition | None,
) -> dict[str, object] | None:
    if transition is None:
        return None
    return {
        "next_state": _state_payload(transition.next_state),
        "raw_observation": transition.raw_observation,
        "raw_reward": transition.raw_reward,
        "raw_done": transition.raw_done,
        "raw_won": transition.raw_won,
        "pre_world_state_checksum": transition.pre_world_state_checksum,
        "post_world_state_checksum": transition.post_world_state_checksum,
        "gamefile": transition.info.get("extra.gamefile"),
    }


def _first_mismatches(
    expected: Iterable[object], actual: Iterable[object], *, label: str, limit: int = 20
) -> list[str]:
    mismatches = []
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            mismatches.append(f"{label}[{index}] differs")
            if len(mismatches) >= limit:
                break
    return mismatches


def _serial_run(
    factory: AlfworldEnvironmentFactory,
    slots: tuple[tuple[TaskSpec, int, int], ...],
    *,
    steps: int,
    progress: object,
) -> tuple[dict[str, float], tuple[object, ...], tuple[tuple[str | None, ...], ...]]:
    started = time.perf_counter()
    environments = tuple(
        factory.create(task, rollout_id=rollout_id, seed=seed)
        for task, rollout_id, seed in slots
    )
    create_seconds = time.perf_counter() - started
    try:
        started = time.perf_counter()
        states = list(environment.reset() for environment in environments)
        reset_seconds = time.perf_counter() - started
        snapshots: list[object] = [tuple(_state_payload(state) for state in states)]
        actions_by_step = []
        step_seconds = 0.0
        for step in range(steps):
            actions = tuple(
                _action(state, rollout_id=slots[index][1], step=step)
                for index, state in enumerate(states)
            )
            actions_by_step.append(actions)
            if all(action is None for action in actions):
                actions_by_step.pop()
                break
            started = time.perf_counter()
            transitions = tuple(
                None if action is None else environment.step(action)
                for environment, action in zip(environments, actions)
            )
            step_seconds += time.perf_counter() - started
            for index, transition in enumerate(transitions):
                if transition is not None:
                    states[index] = transition.next_state
            snapshots.append(tuple(_transition_payload(item) for item in transitions))
            progress.update(1)  # type: ignore[attr-defined]
    finally:
        started = time.perf_counter()
        for environment in environments:
            environment.close()
        close_seconds = time.perf_counter() - started
    return (
        {
            "create_seconds": create_seconds,
            "reset_seconds": reset_seconds,
            "step_seconds": step_seconds,
            "close_seconds": close_seconds,
            "environment_work_seconds": reset_seconds + step_seconds,
        },
        tuple(snapshots),
        tuple(actions_by_step),
    )


def _native_batch_run(
    factory: AlfworldEnvironmentFactory,
    slots: tuple[tuple[TaskSpec, int, int], ...],
    *,
    actions_by_step: tuple[tuple[str | None, ...], ...],
    progress: object,
) -> tuple[dict[str, float], tuple[object, ...]]:
    tasks = tuple(task for task, _, _ in slots)
    seeds = tuple(seed for _, _, seed in slots)
    started = time.perf_counter()
    batch = factory.create_batch(tasks, seeds=seeds)
    create_seconds = time.perf_counter() - started
    try:
        started = time.perf_counter()
        states = batch.reset()
        reset_seconds = time.perf_counter() - started
        snapshots: list[object] = [tuple(_state_payload(state) for state in states)]
        step_seconds = 0.0
        for actions in actions_by_step:
            started = time.perf_counter()
            transitions = batch.step(actions)
            step_seconds += time.perf_counter() - started
            snapshots.append(tuple(_transition_payload(item) for item in transitions))
            progress.update(1)  # type: ignore[attr-defined]
    finally:
        started = time.perf_counter()
        batch.close()
        close_seconds = time.perf_counter() - started
    return (
        {
            "create_seconds": create_seconds,
            "reset_seconds": reset_seconds,
            "step_seconds": step_seconds,
            "close_seconds": close_seconds,
            "environment_work_seconds": reset_seconds + step_seconds,
        },
        tuple(snapshots),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    task_count, rollouts_per_task, steps, minimum_speedup = _profile_values(args)
    config = AppConfig.load(args.config)
    tasks = discover_tasks(config.paths.alfworld_data, split="train")
    if len(tasks) < task_count:
        raise RuntimeError(f"only {len(tasks)} train tasks available; need {task_count}")
    selected = tasks[:task_count]
    slots = tuple(
        (task, rollout_id, _seed(config.master_seed, task.task_id, rollout_id))
        for task in selected
        for rollout_id in range(rollouts_per_task)
    )
    factory = AlfworldEnvironmentFactory.from_paths(
        alfworld_source=config.paths.alfworld_source,
        config_path=config.paths.alfworld_config,
        data_root=config.paths.alfworld_data,
        max_steps=max(config.max_steps, steps),
    )

    from tqdm.auto import tqdm

    progress = tqdm(total=steps * 2, desc=f"environment/{args.profile}", unit="round", dynamic_ncols=True)
    try:
        serial_metrics, serial_snapshots, actions = _serial_run(
            factory, slots, steps=steps, progress=progress
        )
        progress.total = len(actions) * 2
        progress.refresh()
        batch_metrics, batch_snapshots = _native_batch_run(
            factory, slots, actions_by_step=actions, progress=progress
        )
    finally:
        progress.close()

    mismatches = _first_mismatches(
        serial_snapshots, batch_snapshots, label="round"
    )
    serial_work = serial_metrics["environment_work_seconds"]
    batch_work = batch_metrics["environment_work_seconds"]
    speedup = serial_work / batch_work if batch_work else float("inf")
    semantic_exact = not mismatches and len(serial_snapshots) == len(batch_snapshots)
    performance_passed = speedup >= minimum_speedup
    result = {
        "schema_version": 1,
        "profile": args.profile,
        "task_count": task_count,
        "rollouts_per_task": rollouts_per_task,
        "environment_slots": len(slots),
        "steps": steps,
        "master_seed": config.master_seed,
        "serial": serial_metrics,
        "native_multiprocessing_batch": batch_metrics,
        "environment_work_speedup": speedup,
        "minimum_speedup": minimum_speedup,
        "semantic_exact": semantic_exact,
        "semantic_mismatches": mismatches,
        "performance_passed": performance_passed,
        "passed": semantic_exact and performance_passed,
    }
    if args.output:
        output = Path(args.output).expanduser()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path(config.paths.output_root) / f"{stamp}-environment-{args.profile}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["output"] = str(output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not semantic_exact:
        return 3
    return 0 if performance_passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
