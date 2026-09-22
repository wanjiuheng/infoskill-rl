"""Fail-closed first-step parity gate for WebShop imitation and online prompts.

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
    online_actions = normalize_available_actions(online_available_actions)
    if isinstance(online_available_actions, dict) and online_available_actions.get(
        "has_search_bar"
    ):
        # The HTML search submit button is not a separate policy action.
        online_actions = tuple(
            action for action in online_actions if action != "click[search]"
        )
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
) -> dict[str, object]:
    """Audit selected train demonstrations against one real WebShop runtime."""

    if not 1 <= sample_count <= 12:
        raise ValueError("sample_count must be between 1 and 12")
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
    selected = _select_first_steps(prepared_root / "train.jsonl", sample_count)
    demonstrations = _selected_demonstrations(
        demonstrations_path, {line_number for _, line_number, _ in selected}
    )
    environment = _open_external_text_env(source_root, paths)
    try:
        environment.observation_mode = "text_rich"
        positions, unknown = map_runtime_goals(official_goals, environment.server.goals)
        if unknown:
            raise RuntimeError("runtime WebShop goals differ from registered official goals")
        examples: list[dict[str, object]] = []
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
        return {
            "schema_version": 1,
            "status": (
                "passed" if all(item["passed"] for item in examples) else "failed"
            ),
            "scope": "train demonstration first-step versus real WebShop reset",
            "not_proven": ["later-step parity", "policy rollout", "500-task evaluation"],
            "sample_count": len(examples),
            "examples": examples,
        }
    finally:
        environment.close()


def _select_first_steps(path: Path, limit: int) -> list[tuple[int, int, dict[str, object]]]:
    selected: list[tuple[int, int, dict[str, object]]] = []
    seen_goals: set[int] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("step_index") != 0:
                continue
            match = _TASK_ID.fullmatch(str(row.get("task_id", "")))
            if match is None:
                raise RuntimeError(
                    "prepared WebShop trajectory ID does not match the official source"
                )
            goal_index, line_number = map(int, match.groups())
            if goal_index in seen_goals:
                continue
            selected.append((goal_index, line_number, row))
            seen_goals.add(goal_index)
            if len(selected) == limit:
                break
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = run_prompt_parity(
        source_root=Path(args.webshop_root).expanduser().resolve(strict=True),
        data_root=Path(args.webshop_data_root).expanduser().resolve(strict=True),
        prepared_root=Path(args.prepared_data).expanduser().resolve(strict=True),
        sample_count=args.sample_count,
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
