from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

from infoskill.app_config import AppConfig
from infoskill.episode import TaskSpec

from .expert_replay import ExpertReplayResult, StrictExpertReplay
from .factory import AlfworldEnvironmentFactory
from .planner_expert import PlannerPayloadExpert


def select_loop_diagnostic_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    successful_two_object_controls: int,
    persist_horizon: int,
) -> tuple[dict[str, object], ...]:
    """Keep every pilot failure and the longest successful two-object controls."""

    if successful_two_object_controls <= 0 or persist_horizon <= 0:
        raise ValueError("diagnostic control count and persist horizon must be positive")
    failures = [
        dict(row, diagnostic_role="source_failure")
        for row in rows
        if not row.get("succeeded")
    ]
    controls = [
        dict(row, diagnostic_role="long_success_control")
        for row in rows
        if row.get("succeeded")
        and row.get("task_type") == "pick_two_obj_and_place"
        and int(row.get("total_steps", 0)) > persist_horizon
    ]
    controls.sort(
        key=lambda row: (-int(row["total_steps"]), str(row["task_id"]))
    )
    if len(controls) < successful_two_object_controls:
        raise ValueError(
            "source pilot does not contain enough long successful two-object controls"
        )
    if not failures:
        raise ValueError("source pilot contains no failures to diagnose")
    return tuple(failures + controls[:successful_two_object_controls])


class TracingPlannerExpert(PlannerPayloadExpert):
    """Planner adapter that records pre-action evidence without changing actions."""

    def __init__(self) -> None:
        self.trace: list[dict[str, object]] = []

    def reset(self, gamefile: str) -> None:
        super().reset(gamefile)
        self.trace.clear()

    def act(
        self,
        game_state: Mapping[str, object],
        reward: float,
        done: bool,
        last_action: str,
    ) -> str:
        commands = _string_sequence(game_state.get("admissible_commands"))
        plan = _string_sequence(game_state.get("extra.expert_plan"))
        feedback = str(game_state.get("feedback", ""))
        state_fingerprint, fingerprint_source, fact_count = _state_fingerprint(
            game_state=game_state,
            observation=feedback,
            admissible_commands=commands,
        )
        evidence: dict[str, object] = {
            # StrictExpertReplay executes an initial `look`; the first planner
            # decision therefore observes canonical environment step 1.
            "step_index": len(self.trace) + 1,
            "observation": feedback,
            "admissible_commands": list(commands),
            "admissible_command_count": len(commands),
            "planner_plan_length": len(plan),
            "planner_plan_head": list(plan[:5]),
            "last_action": last_action,
            "state_fingerprint": state_fingerprint,
            "state_fingerprint_source": fingerprint_source,
            "world_fact_count": fact_count,
        }
        try:
            action = super().act(game_state, reward, done, last_action)
        except Exception as error:
            evidence["action"] = None
            evidence["expert_error"] = {
                "type": type(error).__name__,
                "message": " ".join(str(error).split())[:1000],
            }
            self.trace.append(evidence)
            raise
        evidence["action"] = action
        evidence["expert_error"] = None
        self.trace.append(evidence)
        return action


def summarize_planner_trace(
    trace: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    pairs = [
        (str(step.get("state_fingerprint", "")), str(step.get("action") or ""))
        for step in trace
    ]
    pair_counts = Counter(pairs)
    state_counts = Counter(pair[0] for pair in pairs)
    maximum = max(pair_counts.values(), default=0)
    seen: Counter[tuple[str, str]] = Counter()
    first_cycle_step: int | None = None
    for step, pair in zip(trace, pairs):
        seen[pair] += 1
        if seen[pair] == 3:
            first_cycle_step = int(step.get("step_index", 0))
            break
    ranked = sorted(
        pair_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )
    return {
        "recorded_planner_steps": len(trace),
        "trace_starts_after_forced_look": True,
        "unique_state_fingerprints": len(state_counts),
        "unique_actions": len({pair[1] for pair in pairs}),
        "repeated_state_action_occurrences": sum(
            count - 1 for count in pair_counts.values() if count > 1
        ),
        "max_state_action_occurrences": maximum,
        "max_consecutive_same_action": _max_consecutive_actions(pairs),
        "cycle_detected": maximum >= 3,
        "first_cycle_step": first_cycle_step,
        "most_repeated_state_actions": [
            {
                "state_fingerprint": state,
                "action": action,
                "occurrences": count,
            }
            for (state, action), count in ranked[:5]
            if count > 1
        ],
    }


def run_planner_loop_diagnostic(
    *,
    config: AppConfig,
    tasks: Mapping[str, TaskSpec],
    selected_rows: Sequence[Mapping[str, object]],
    max_replay_steps: int,
    persist_horizon: int,
    on_progress: Callable[[int], None] | None = None,
) -> tuple[dict[str, object], ...]:
    factory = AlfworldEnvironmentFactory.from_paths(
        alfworld_source=config.paths.alfworld_source,
        config_path=config.paths.alfworld_config,
        data_root=config.paths.alfworld_data,
        max_steps=max_replay_steps,
        expert_type="planner",
    )
    replay = StrictExpertReplay(
        max_replay_steps=max_replay_steps,
        persist_horizon=persist_horizon,
    )
    output: list[dict[str, object]] = []
    for source in selected_rows:
        task_id = str(source["task_id"])
        try:
            task = tasks[task_id]
        except KeyError as error:
            raise ValueError(f"source pilot task is absent from train data: {task_id}") from error
        if task.task_type != source.get("task_type"):
            raise ValueError(f"source pilot task type changed: {task_id}")
        seed = int(source["seed"])
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed % (2**32))
        except ImportError:  # pragma: no cover - server runtime includes NumPy.
            pass
        expert = TracingPlannerExpert()
        try:
            environment = factory.create(task, rollout_id=0, seed=seed)
        except Exception as error:
            result = ExpertReplayResult(
                task_id=task_id,
                succeeded=False,
                samples=(),
                total_steps=0,
                quarantine_reason=f"factory_exception:{type(error).__name__}",
                exception_stage="factory_create",
                exception_type=type(error).__name__,
                exception_message=" ".join(str(error).split())[:1000],
            )
        else:
            result = replay.run(
                task=task,
                environment=environment,
                expert=expert,
            )
        trace = tuple(expert.trace)
        output.append(
            {
                "task_id": task_id,
                "task_type": task.task_type,
                "goal": task.goal,
                "seed": seed,
                "diagnostic_role": source["diagnostic_role"],
                "source_result": {
                    "succeeded": bool(source["succeeded"]),
                    "total_steps": int(source["total_steps"]),
                    "reason": source.get("reason"),
                },
                "diagnostic_result": {
                    "succeeded": result.succeeded,
                    "total_steps": result.total_steps,
                    "reason": result.quarantine_reason,
                    "exception_stage": result.exception_stage,
                    "exception_type": result.exception_type,
                    "exception_message": result.exception_message,
                },
                "trace_summary": summarize_planner_trace(trace),
                "trace": list(trace),
            }
        )
        if on_progress is not None:
            on_progress(1)
    return tuple(output)


def build_planner_loop_report(
    *,
    rows: Sequence[Mapping[str, object]],
    source_pilot_run: str,
    source_pilot_sha256: str,
    source_results_sha256: str,
    train_task_manifest_sha256: str,
    code_revision: str,
    source_max_replay_steps: int,
    diagnostic_max_replay_steps: int,
    successful_two_object_controls: int,
    temporary_directory_cleaned: bool,
    minimum_free_disk_bytes: int,
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot build an empty planner loop diagnostic")
    source_failures = [row for row in rows if row["diagnostic_role"] == "source_failure"]
    controls = [row for row in rows if row["diagnostic_role"] == "long_success_control"]
    if len(controls) != successful_two_object_controls:
        raise ValueError("diagnostic control count does not match selected rows")
    rescued = [row for row in source_failures if _result(row)["succeeded"]]
    still_failed = [row for row in source_failures if not _result(row)["succeeded"]]
    control_regressions = [row for row in controls if not _result(row)["succeeded"]]
    cycles = [row for row in rows if bool(_trace_summary(row)["cycle_detected"])]
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts = by_type[str(row["task_type"])]
        counts["total"] += 1
        counts["succeeded_at_diagnostic_horizon"] += int(
            bool(_result(row)["succeeded"])
        )
        counts["cycles_detected"] += int(
            bool(_trace_summary(row)["cycle_detected"])
        )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "formal_run_required": True,
        "source_pilot_run": source_pilot_run,
        "source_checksums": {
            "planner_pilot": source_pilot_sha256,
            "planner_pilot_results": source_results_sha256,
            "train_task_manifest": train_task_manifest_sha256,
            "infoskill_source": code_revision,
        },
        "code_revision": code_revision[:16],
        "expert_name": "ALFWorld planner (direct, strict admissibility, no fallback)",
        "source_max_replay_steps": source_max_replay_steps,
        "diagnostic_max_replay_steps": diagnostic_max_replay_steps,
        "selected_tasks": len(rows),
        "source_failure_tasks": len(source_failures),
        "successful_two_object_controls": successful_two_object_controls,
        "rescued_after_source_horizon": len(rescued),
        "rescued_task_ids": [str(row["task_id"]) for row in rescued],
        "still_failed_at_diagnostic_horizon": len(still_failed),
        "still_failed_task_ids": [str(row["task_id"]) for row in still_failed],
        "control_regressions": len(control_regressions),
        "control_regression_task_ids": [
            str(row["task_id"]) for row in control_regressions
        ],
        "cycles_detected": len(cycles),
        "cycle_task_ids": [str(row["task_id"]) for row in cycles],
        "task_type_counts": {
            task_type: dict(counts)
            for task_type, counts in sorted(by_type.items())
        },
        "resource_lifecycle": {
            "temporary_directory_cleaned": temporary_directory_cleaned,
            "minimum_free_disk_bytes": minimum_free_disk_bytes,
        },
    }


def write_planner_loop_rows(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _state_fingerprint(
    *,
    game_state: Mapping[str, object],
    observation: str,
    admissible_commands: Sequence[str],
) -> tuple[str, str, int | None]:
    raw_facts = game_state.get("facts")
    if raw_facts is not None:
        facts = sorted(_string_sequence(raw_facts))
        payload: object = facts
        source = "facts"
        fact_count: int | None = len(facts)
    else:
        payload = {
            "observation": observation,
            "admissible_commands": sorted(admissible_commands),
            "won": bool(game_state.get("won", False)),
        }
        source = "observable_fallback"
        fact_count = None
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest, source, fact_count


def _string_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)  # type: ignore[union-attr]
    except TypeError:
        return (str(value),)


def _max_consecutive_actions(pairs: Sequence[tuple[str, str]]) -> int:
    maximum = 0
    current = 0
    previous: str | None = None
    for _, action in pairs:
        current = current + 1 if action == previous else 1
        maximum = max(maximum, current)
        previous = action
    return maximum


def _result(row: Mapping[str, object]) -> Mapping[str, object]:
    result = row.get("diagnostic_result")
    if not isinstance(result, Mapping):
        raise TypeError("diagnostic result must be a mapping")
    return result


def _trace_summary(row: Mapping[str, object]) -> Mapping[str, object]:
    summary = row.get("trace_summary")
    if not isinstance(summary, Mapping):
        raise TypeError("trace summary must be a mapping")
    return summary
