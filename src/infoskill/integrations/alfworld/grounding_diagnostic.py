from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Callable, Mapping, Sequence

from infoskill.app_config import AppConfig
from infoskill.episode import TaskSpec

from .expert_replay import ExpertReplayResult, StrictExpertReplay
from .factory import AlfworldEnvironmentFactory
from .handcoded_expert import load_handcoded_expert
from .planner_expert import PlannerPayloadExpert
from .tasks import ALFWORLD_TASK_TYPES


def select_quarantined_tasks(
    *,
    tasks: Sequence[TaskSpec],
    quarantine_rows: Sequence[Mapping[str, object]],
    tasks_per_type: int,
    selection_seed: int,
) -> tuple[TaskSpec, ...]:
    if tasks_per_type <= 0:
        raise ValueError("tasks_per_type must be positive")
    task_by_id = {task.task_id: task for task in tasks}
    ids_by_type: dict[str, set[str]] = defaultdict(set)
    for row in quarantine_rows:
        if row.get("reason") != "expert_action_not_admissible":
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in task_by_id:
            continue
        task = task_by_id[task_id]
        ids_by_type[task.task_type].add(task_id)
    selected: list[TaskSpec] = []
    for task_type in ALFWORLD_TASK_TYPES:
        ordered_ids = sorted(
            ids_by_type[task_type],
            key=lambda task_id: (
                hashlib.sha256(
                    f"grounding-diagnostic|{selection_seed}|{task_id}".encode()
                ).digest(),
                task_id,
            ),
        )
        for task_id in ordered_ids[:tasks_per_type]:
            selected.append(task_by_id[task_id])
    return tuple(selected)


def run_expert_diagnostic(
    *,
    config: AppConfig,
    tasks: Sequence[TaskSpec],
    quarantine_rows: Sequence[Mapping[str, object]],
    tasks_per_type: int,
    max_replay_steps: int,
    persist_horizon: int,
    seed_for_task: Callable[[str], int],
    selection_seed: int,
    on_progress: Callable[[int], None] | None = None,
) -> dict[str, object]:
    paths = config.paths
    selected = select_quarantined_tasks(
        tasks=tasks,
        quarantine_rows=quarantine_rows,
        tasks_per_type=tasks_per_type,
        selection_seed=selection_seed,
    )
    if not selected:
        raise ValueError("source run contains no matching expert action quarantines")
    handcoded_factory = AlfworldEnvironmentFactory.from_paths(
        alfworld_source=paths.alfworld_source,
        config_path=paths.alfworld_config,
        data_root=paths.alfworld_data,
        max_steps=max_replay_steps,
        expert_type="handcoded",
    )
    planner_factory = AlfworldEnvironmentFactory.from_paths(
        alfworld_source=paths.alfworld_source,
        config_path=paths.alfworld_config,
        data_root=paths.alfworld_data,
        max_steps=max_replay_steps,
        expert_type="planner",
    )
    handcoded = load_handcoded_expert(
        alfworld_source=paths.alfworld_source,
        max_steps=max(200, max_replay_steps),
    )
    planner = PlannerPayloadExpert()
    replay = StrictExpertReplay(
        max_replay_steps=max_replay_steps,
        persist_horizon=persist_horizon,
    )
    source_quarantine_by_task = {
        str(row["task_id"]): dict(row)
        for row in quarantine_rows
        if row.get("reason") == "expert_action_not_admissible"
        and isinstance(row.get("task_id"), str)
    }
    rows: list[dict[str, object]] = []
    for task in selected:
        seed = seed_for_task(task.task_id)
        handcoded_result = _run_one(
            task=task,
            seed=seed,
            factory=handcoded_factory,
            expert=handcoded,
            replay=replay,
        )
        handcoded_policy = _handcoded_policy_snapshot(handcoded)
        if on_progress is not None:
            on_progress(1)
        planner_result = _run_one(
            task=task,
            seed=seed,
            factory=planner_factory,
            expert=planner,
            replay=replay,
        )
        if on_progress is not None:
            on_progress(1)
        rows.append(
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "goal": task.goal,
                "seed": seed,
                "source_quarantine": source_quarantine_by_task[task.task_id],
                "handcoded": _result_payload(handcoded_result),
                "handcoded_policy": handcoded_policy,
                "planner": _result_payload(planner_result),
            }
        )
    return {
        "schema_version": 1,
        "tasks_per_type": tasks_per_type,
        "selection_seed": selection_seed,
        "selected_task_count": len(selected),
        "selected_task_type_counts": dict(
            sorted(Counter(task.task_type for task in selected).items())
        ),
        "summary": _summary(rows),
        "tasks": rows,
    }


def _run_one(
    *,
    task: TaskSpec,
    seed: int,
    factory: AlfworldEnvironmentFactory,
    expert: object,
    replay: StrictExpertReplay,
) -> ExpertReplayResult:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover - server runtime always includes NumPy.
        pass
    try:
        environment = factory.create(task, rollout_id=0, seed=seed)
    except Exception as error:
        return ExpertReplayResult(
            task_id=task.task_id,
            succeeded=False,
            samples=(),
            total_steps=0,
            quarantine_reason=f"factory_exception:{type(error).__name__}",
            exception_stage="factory_create",
            exception_type=type(error).__name__,
            exception_message=" ".join(str(error).split())[:1000],
        )
    return replay.run(
        task=task,
        environment=environment,
        expert=expert,  # type: ignore[arg-type]
    )


def _result_payload(result: ExpertReplayResult) -> dict[str, object]:
    mismatch = result.action_mismatch
    payload: dict[str, object] = {
        "succeeded": result.succeeded,
        "total_steps": result.total_steps,
        "reason": result.quarantine_reason,
        "exception_stage": result.exception_stage,
        "exception_type": result.exception_type,
        "exception_message": result.exception_message,
        "action_mismatch": asdict(mismatch) if mismatch is not None else None,
    }
    if mismatch is not None:
        proposed_without_ids = _without_numeric_ids(mismatch.proposed_action)
        payload["mismatch_analysis"] = {
            "repeated_last_action": (
                mismatch.proposed_action.strip().lower()
                == mismatch.last_action.strip().lower()
            ),
            "same_verb_commands": [
                command
                for command in mismatch.admissible_commands
                if _verb(command) == _verb(mismatch.proposed_action)
            ],
            "numeric_id_only_matches": [
                command
                for command in mismatch.admissible_commands
                if _without_numeric_ids(command) == proposed_without_ids
            ],
            "wrapper_fell_back_to_look": (
                mismatch.environment_expert_plan == ("look",)
                and mismatch.proposed_action.strip().lower() != "look"
            ),
        }
    return payload


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for expert_name in ("handcoded", "planner"):
        results = [_mapping(row[expert_name], field=expert_name) for row in rows]
        reasons = Counter(
            "success" if result["succeeded"] else str(result["reason"])
            for result in results
        )
        summary[expert_name] = {
            "success_count": sum(bool(result["succeeded"]) for result in results),
            "reason_counts": dict(sorted(reasons.items())),
        }
    summary["planner_rescue_count"] = sum(
        not bool(_mapping(row["handcoded"], field="handcoded")["succeeded"])
        and bool(_mapping(row["planner"], field="planner")["succeeded"])
        for row in rows
    )
    summary["historical_failure_reproduced_count"] = sum(
        _same_failure(
            source=_mapping(row["source_quarantine"], field="source_quarantine"),
            fresh=_mapping(row["handcoded"], field="handcoded"),
        )
        for row in rows
    )
    return summary


def _same_failure(
    *,
    source: Mapping[str, object],
    fresh: Mapping[str, object],
) -> bool:
    return (
        not bool(fresh["succeeded"])
        and source.get("reason") == fresh.get("reason")
        and source.get("total_steps") == fresh.get("total_steps")
    )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _handcoded_policy_snapshot(expert: object) -> dict[str, object] | None:
    policy = getattr(expert, "policy", None)
    if policy is None:
        return None
    subgoals = getattr(policy, "subgoals", ())
    subgoal_index = int(getattr(policy, "subgoal_idx", -1))
    current_subgoal = (
        subgoals[subgoal_index]
        if isinstance(subgoals, list) and 0 <= subgoal_index < len(subgoals)
        else None
    )
    return {
        "policy_class": type(policy).__name__,
        "steps": int(getattr(policy, "steps", 0)),
        "subgoal_index": subgoal_index,
        "current_subgoal": current_subgoal,
        "current_receptacle": str(getattr(policy, "curr_recep", "")),
        "inventory": list(getattr(policy, "inventory", ())),
        "visible_objects": dict(getattr(policy, "visible_objects", {})),
        "receptacles_to_check": list(
            getattr(policy, "receptacles_to_check", ())
        ),
        "action_backlog": list(getattr(policy, "action_backlog", ())),
        "object_blacklist": list(getattr(policy, "object_blacklist", ())),
    }


def _verb(action: str) -> str:
    words = action.strip().lower().split()
    return words[0] if words else ""


def _without_numeric_ids(action: str) -> str:
    return " ".join("".join(char for char in action.lower() if not char.isdigit()).split())
