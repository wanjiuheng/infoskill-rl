"""Fail-closed replay parity gate for WebShop imitation and online prompts.

The offline demonstration's task text is reused in the online reconstruction so
that WebShop's randomly generated price limit does not masquerade as a prompt
format error. The goal, observation, and admissible actions must still agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

from infoskill.imitation.audit import registered_webshop_manifest_failures

from .demonstrations import (
    REGISTERED_HUMAN_DEMONSTRATIONS_SHA256,
    REGISTERED_HUMAN_GOALS_SHA256,
    normalize_available_actions,
    normalize_goal,
    observation_from_state,
    resolve_demonstration_actions,
    task_description_from_state,
)
from .online_smoke import (
    _open_external_text_env,
    _page_goal_text,
    _required_assets,
    map_runtime_goals,
)
from .policy import render_webshop_policy_message
from .splits import WebShopSplit, split_for_goal_index


_TASK_ID = re.compile(r"^webshop-human-goal-(\d{5})-line-(\d{5})$")
_RESPONSE_ACTION = re.compile(r"<action>([^\n]+)</action>\s*$")
_REQUIRED_REPLAY_PHASES = ("search", "results", "product", "options", "purchase")


def compare_replayed_step(
    *,
    offline_prompt: str,
    offline_goal: str,
    offline_observation: str,
    offline_actions: Sequence[str],
    offline_history: Sequence[tuple[str, str]],
    online_goal: str,
    online_observation: str,
    online_available_actions: object,
    online_history: Sequence[tuple[str, str]],
    demonstrated_action: str,
) -> dict[str, object]:
    """Compare one demonstrated decision with a real replayed environment state."""

    offline_actions = tuple(offline_actions)
    online_actions = _normalized_online_actions(online_available_actions)
    online_observation = observation_from_state(
        online_observation, normalize_goal(online_goal)
    )
    offline_rebuilt = render_webshop_policy_message(
        task_description=offline_goal,
        current_observation=offline_observation,
        available_actions=offline_actions,
        history=offline_history,
    )
    conditional_online_prompt = render_webshop_policy_message(
        task_description=offline_goal,
        current_observation=online_observation,
        available_actions=online_actions,
        history=online_history,
    )
    actual_online_prompt = render_webshop_policy_message(
        task_description=online_goal,
        current_observation=online_observation,
        available_actions=online_actions,
        history=online_history,
    )
    checks = {
        "goal_match": normalize_goal(offline_goal) == normalize_goal(online_goal),
        "offline_prompt_reconstructs": offline_prompt == offline_rebuilt,
        "observation_match": offline_observation == online_observation,
        "actions_match": offline_actions == online_actions,
        "history_match": tuple(offline_history) == tuple(online_history),
        "conditional_prompt_match": offline_prompt == conditional_online_prompt,
    }
    executable = _resolve_online_action(demonstrated_action, online_actions) is not None
    return {
        "structural_passed": all(checks.values()),
        **checks,
        "exact_online_prompt_match": offline_prompt == actual_online_prompt,
        "full_task_text_match": offline_goal == online_goal,
        "demonstrated_action_executable": executable,
        "offline_observation_sha256": _digest(offline_observation),
        "online_observation_sha256": _digest(online_observation),
        "offline_prompt_sha256": _digest(offline_prompt),
        "conditional_online_prompt_sha256": _digest(conditional_online_prompt),
        "actual_online_prompt_sha256": _digest(actual_online_prompt),
        "offline_action_count": len(offline_actions),
        "online_action_count": len(online_actions),
    }


def replay_demonstration(
    *,
    environment: object,
    demonstration: dict[str, object],
    prepared_rows: Sequence[dict[str, object]],
    runtime_position: int,
    max_steps: int,
) -> dict[str, object]:
    """Replay one registered demonstration until completion or first bad action."""

    states = _required_sequence(demonstration, "states")
    candidates = _required_sequence(demonstration, "available_actions")
    action_indices = _required_sequence(demonstration, "action_idxs")
    actions = resolve_demonstration_actions(
        demonstration, candidates, action_indices, 0
    )
    if not (len(states) == len(candidates) == len(actions) == len(prepared_rows)):
        raise RuntimeError("WebShop replay arrays differ in length")
    if not states or max_steps < 1:
        raise RuntimeError("WebShop replay requires at least one step")
    offline_goal = task_description_from_state(states[0])
    normalized_goal = normalize_goal(states[0])
    online_observation, _ = environment.reset(session=runtime_position)
    online_goal = _page_goal_text(environment.instruction_text)
    offline_history: list[tuple[str, str]] = []
    online_history: list[tuple[str, str]] = []
    step_reports: list[dict[str, object]] = []
    stopped_reason: str | None = None
    final_done = False
    final_reward = 0.0
    for step_index, (state, available, action, row) in enumerate(
        zip(states, candidates, actions, prepared_rows, strict=True)
    ):
        if step_index >= max_steps:
            stopped_reason = "max_steps_reached"
            break
        action_text = _required_text(action, "demonstrated action")
        response_action = _response_action(row)
        if response_action != action_text:
            raise RuntimeError("prepared response differs from source demonstration")
        offline_observation = observation_from_state(state, normalized_goal)
        offline_actions = normalize_available_actions(
            available, demonstrated_action=action_text
        )
        online_available = environment.get_available_actions()
        normalized_online_actions = _normalized_online_actions(online_available)
        resolved_online_action = _resolve_online_action(
            action_text, normalized_online_actions
        )
        comparison = compare_replayed_step(
            offline_prompt=_required_text(row.get("prompt"), "prepared prompt"),
            offline_goal=offline_goal,
            offline_observation=offline_observation,
            offline_actions=offline_actions,
            offline_history=offline_history,
            online_goal=online_goal,
            online_observation=online_observation,
            online_available_actions=online_available,
            online_history=online_history,
            demonstrated_action=action_text,
        )
        comparison["step_index"] = step_index
        step_reports.append(comparison)
        if not comparison["demonstrated_action_executable"]:
            stopped_reason = "demonstrated_action_not_executable"
            break
        if resolved_online_action is None:
            stopped_reason = "demonstrated_action_not_executable"
            break
        next_observation, reward, done, _ = environment.step(resolved_online_action)
        final_done = bool(done)
        final_reward = float(reward)
        if step_index + 1 == len(states):
            if not done:
                stopped_reason = "demonstration_ended_before_environment"
            break
        offline_history.append((offline_observation, action_text))
        online_history.append(
            (
                observation_from_state(
                    online_observation, normalize_goal(online_goal)
                ),
                resolved_online_action,
            )
        )
        if done:
            stopped_reason = "environment_terminated_before_demonstration"
            break
        online_observation = next_observation
    structural_passed = (
        len(step_reports) == len(states)
        and stopped_reason is None
        and all(bool(item["structural_passed"]) for item in step_reports)
    )
    return {
        "status": "passed" if structural_passed else "failed",
        "source_steps": len(states),
        "checked_steps": len(step_reports),
        "structural_passed": structural_passed,
        "exact_online_prompt_match": bool(step_reports) and all(
            bool(item["exact_online_prompt_match"]) for item in step_reports
        ),
        "stopped_reason": stopped_reason,
        "final_done": final_done,
        "final_reward": final_reward,
        "phase_coverage": sorted(_trajectory_phases(actions)),
        "steps": step_reports,
    }


def compare_initial_step(
    *,
    offline_prompt: str,
    offline_goal: str,
    offline_observation: str,
    offline_actions: Sequence[str],
    online_goal: str,
    online_observation: str,
    online_available_actions: object,
) -> dict[str, object]:
    """Compare a demonstrated first step with the same live goal's reset state."""

    offline_actions = tuple(offline_actions)
    online_actions = _normalized_online_actions(online_available_actions)
    online_observation = observation_from_state(online_observation, normalize_goal(online_goal))
    offline_rebuilt = render_webshop_policy_message(
        task_description=offline_goal,
        current_observation=offline_observation,
        available_actions=offline_actions,
    )
    online_rebuilt = render_webshop_policy_message(
        task_description=offline_goal,
        current_observation=online_observation,
        available_actions=online_actions,
    )
    checks = {
        "goal_match": normalize_goal(offline_goal) == normalize_goal(online_goal),
        "offline_prompt_reconstructs": offline_prompt == offline_rebuilt,
        "observation_match": offline_observation == online_observation,
        "actions_match": offline_actions == online_actions,
        "prompt_match": offline_prompt == online_rebuilt,
    }
    return {
        "passed": all(checks.values()),
        **checks,
        "online_observation": online_observation,
        "online_actions": list(online_actions),
        "offline_observation_sha256": _digest(offline_observation),
        "online_observation_sha256": _digest(online_observation),
        "offline_prompt_sha256": _digest(offline_prompt),
        "online_prompt_sha256": _digest(online_rebuilt),
        "offline_action_count": len(offline_actions),
        "online_action_count": len(online_actions),
        "full_task_text_match": offline_goal == online_goal,
    }


def run_prompt_parity(
    *,
    source_root: Path,
    data_root: Path,
    prepared_root: Path,
    sample_count: int = 3,
    max_steps: int = 100,
) -> dict[str, object]:
    """Replay selected train demonstrations in one real WebShop runtime."""

    if not 1 <= sample_count <= 12:
        raise ValueError("sample_count must be between 1 and 12")
    if not 1 <= max_steps <= 128:
        raise ValueError("max_steps must be between 1 and 128")
    paths = _required_assets(source_root, data_root)
    manifest = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
    failures = registered_webshop_manifest_failures(manifest)
    if failures:
        raise RuntimeError(f"prepared WebShop manifest is not registered: {failures}")
    demonstrations_path = (
        data_root / "baseline_models/data/il_trajs_finalized_images.jsonl"
    )
    human_goals_path = paths["human_goals"]
    for path, expected in (
        (demonstrations_path, REGISTERED_HUMAN_DEMONSTRATIONS_SHA256),
        (human_goals_path, REGISTERED_HUMAN_GOALS_SHA256),
    ):
        if _sha256_file(path) != expected:
            raise RuntimeError(f"WebShop registered source checksum differs: {path.name}")
    official_goals = json.loads(human_goals_path.read_text(encoding="utf-8"))
    selected = _select_replay_trajectories(
        prepared_root / "train.jsonl", sample_count
    )
    demonstrations = _selected_demonstrations(
        demonstrations_path, {line_number for _, line_number, _ in selected}
    )
    prepared_rows = _prepared_rows_for_selected(
        prepared_root / "train.jsonl",
        {str(row["task_id"]) for _, _, row in selected},
    )
    environment = _open_external_text_env(source_root, paths)
    try:
        environment.observation_mode = "text_rich"
        positions, unknown = map_runtime_goals(official_goals, environment.server.goals)
        if unknown:
            raise RuntimeError("runtime WebShop goals differ from registered official goals")
        examples: list[dict[str, object]] = []
        replays: list[dict[str, object]] = []
        for goal_index, line_number, row in selected:
            if (
                split_for_goal_index(goal_index, goal_count=len(official_goals))
                is not WebShopSplit.TRAIN
            ):
                raise RuntimeError("prepared WebShop train row belongs to a held-out goal")
            runtime_position = positions.get(goal_index)
            if runtime_position is None:
                raise RuntimeError(f"WebShop train goal {goal_index} is missing at runtime")
            demonstration = demonstrations[line_number]
            state = demonstration["states"][0]
            offline_goal = task_description_from_state(state)
            action_match = _RESPONSE_ACTION.search(str(row["response"]))
            if action_match is None:
                raise RuntimeError("prepared WebShop first-step response has no action")
            offline_actions = normalize_available_actions(
                demonstration["available_actions"][0],
                demonstrated_action=action_match.group(1),
            )
            online_observation, _ = environment.reset(session=runtime_position)
            comparison = compare_initial_step(
                offline_prompt=str(row["prompt"]),
                offline_goal=offline_goal,
                offline_observation=observation_from_state(
                    state, normalize_goal(state)
                ),
                offline_actions=offline_actions,
                online_goal=_page_goal_text(environment.instruction_text),
                online_observation=online_observation,
                online_available_actions=environment.get_available_actions(),
            )
            # No raw instruction, product, or demonstration text in the report.
            comparison.pop("online_observation")
            comparison.pop("online_actions")
            comparison["goal_index"] = goal_index
            comparison["demonstration_line"] = line_number
            comparison["runtime_position"] = runtime_position
            examples.append(comparison)
            task_id = str(row["task_id"])
            replay = replay_demonstration(
                environment=environment,
                demonstration=demonstration,
                prepared_rows=prepared_rows[task_id],
                runtime_position=runtime_position,
                max_steps=max_steps,
            )
            replay["goal_index"] = goal_index
            replay["demonstration_line"] = line_number
            replay["runtime_position"] = runtime_position
            replays.append(replay)
        covered_phases = {
            phase for replay in replays for phase in replay["phase_coverage"]
        }
        phase_coverage_complete = set(_REQUIRED_REPLAY_PHASES) <= covered_phases
        structural_passed = (
            phase_coverage_complete
            and all(item["passed"] for item in examples)
            and all(
            item["structural_passed"] for item in replays
            )
        )
        return {
            "schema_version": 2,
            "status": "passed" if structural_passed else "failed",
            "gate_basis": "structural_parity_with_registered_offline_task_text",
            "scope": "sampled train demonstrations replayed in real WebShop",
            "not_proven": [
                "all 1010 train demonstrations",
                "policy rollout",
                "500-task evaluation",
            ],
            "sample_count": len(examples),
            "max_steps": max_steps,
            "structural_passed": structural_passed,
            "required_phase_coverage": list(_REQUIRED_REPLAY_PHASES),
            "observed_phase_coverage": sorted(covered_phases),
            "phase_coverage_complete": phase_coverage_complete,
            "all_exact_online_prompts_match": all(
                item["exact_online_prompt_match"] for item in replays
            ),
            "examples": examples,
            "replays": replays,
        }
    finally:
        environment.close()


def _select_replay_trajectories(
    path: Path, limit: int
) -> list[tuple[int, int, dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            task_id = str(row.get("task_id", ""))
            grouped.setdefault(task_id, []).append(row)
    candidates: list[tuple[int, int, dict[str, object], set[str]]] = []
    for task_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row.get("step_index", -1)))
        if not rows or rows[0].get("step_index") != 0:
            raise RuntimeError("prepared WebShop trajectory has no first step")
        row = rows[0]
        match = _TASK_ID.fullmatch(task_id)
        if match is None:
            raise RuntimeError(
                "prepared WebShop trajectory ID does not match the official source"
            )
        goal_index, line_number = map(int, match.groups())
        phases = _trajectory_phases([_response_action(item) for item in rows])
        candidates.append((goal_index, line_number, row, phases))
    selected: list[tuple[int, int, dict[str, object]]] = []
    covered: set[str] = set()
    remaining = list(candidates)
    while remaining and len(selected) < limit:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                len(remaining[index][3] - covered),
                len(remaining[index][3]),
                -index,
            ),
        )
        goal_index, line_number, row, phases = remaining.pop(best_index)
        selected.append((goal_index, line_number, row))
        covered.update(phases)
        remaining = [item for item in remaining if item[0] != goal_index]
    if len(selected) != limit:
        raise RuntimeError("not enough distinct train goals for WebShop prompt parity")
    return selected


def _selected_demonstrations(
    path: Path, line_numbers: set[int]
) -> dict[int, dict[str, object]]:
    selected: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line_number in line_numbers:
                selected[line_number] = json.loads(line)
            if len(selected) == len(line_numbers):
                break
    if set(selected) != line_numbers:
        raise RuntimeError("prepared WebShop demonstration line is missing from source")
    return selected


def _prepared_rows_for_selected(
    path: Path, task_ids: set[str]
) -> dict[str, list[dict[str, object]]]:
    selected: dict[str, list[dict[str, object]]] = {
        task_id: [] for task_id in task_ids
    }
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            task_id = str(row.get("task_id", ""))
            if task_id in selected:
                selected[task_id].append(row)
    for task_id, rows in selected.items():
        rows.sort(key=lambda row: int(row.get("step_index", -1)))
        if [row.get("step_index") for row in rows] != list(range(len(rows))):
            raise RuntimeError(
                f"prepared WebShop steps are incomplete for {task_id}"
            )
    if any(not rows for rows in selected.values()):
        raise RuntimeError("prepared WebShop trajectory rows are missing")
    return selected


def _required_sequence(payload: dict[str, object], field: str) -> list[object]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise RuntimeError(f"WebShop demonstration requires {field}")
    return value


def _normalized_online_actions(value: object) -> tuple[str, ...]:
    actions = normalize_available_actions(value)
    if isinstance(value, dict) and value.get("has_search_bar"):
        # The HTML search submit button is not a separate policy action.
        actions = tuple(action for action in actions if action != "click[search]")
    return actions


def _resolve_online_action(
    demonstrated_action: str, online_actions: Sequence[str]
) -> str | None:
    if demonstrated_action.startswith("search["):
        return (
            demonstrated_action
            if "search[<your query>]" in online_actions
            else None
        )
    expected = demonstrated_action.casefold()
    return next(
        (action for action in online_actions if action.casefold() == expected), None
    )


def _trajectory_phases(actions: Sequence[object]) -> set[str]:
    normalized = [
        _required_text(action, "demonstrated action").casefold()
        for action in actions
    ]
    phases: set[str] = set()
    search_index = next(
        (index for index, action in enumerate(normalized) if action.startswith("search[")),
        None,
    )
    if search_index is not None:
        phases.add("search")
    result_index = (
        search_index + 1
        if search_index is not None
        and search_index + 1 < len(normalized)
        and normalized[search_index + 1].startswith("click[")
        else None
    )
    if result_index is not None:
        phases.add("results")
    purchase_index = next(
        (index for index, action in enumerate(normalized) if action == "click[buy now]"),
        None,
    )
    if purchase_index is not None:
        phases.update(("product", "purchase"))
    ignored_product_clicks = {
        "click[description]",
        "click[features]",
        "click[reviews]",
        "click[back to search]",
        "click[next >]",
        "click[< prev]",
    }
    if result_index is not None and purchase_index is not None:
        if any(
            action.startswith("click[") and action not in ignored_product_clicks
            for action in normalized[result_index + 1 : purchase_index]
        ):
            phases.add("options")
    return phases


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"WebShop {field} must be non-empty text")
    return value.strip()


def _response_action(row: dict[str, object]) -> str:
    match = _RESPONSE_ACTION.search(str(row.get("response", "")))
    if match is None:
        raise RuntimeError("prepared WebShop response has no action")
    return match.group(1).strip()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop-root", required=True)
    parser.add_argument("--webshop-data-root", required=True)
    parser.add_argument("--prepared-data", required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = run_prompt_parity(
        source_root=Path(args.webshop_root).expanduser().resolve(strict=True),
        data_root=Path(args.webshop_data_root).expanduser().resolve(strict=True),
        prepared_root=Path(args.prepared_data).expanduser().resolve(strict=True),
        sample_count=args.sample_count,
        max_steps=args.max_steps,
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"PROMPT_PARITY_REPORT={output}")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
