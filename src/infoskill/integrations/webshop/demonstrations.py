from __future__ import annotations

import json
from pathlib import Path

from infoskill.imitation.providers import DemonstrationStep, DemonstrationTrajectory

from .policy import render_webshop_policy_message
from .splits import WebShopSplit, split_for_goal_index


class OfficialWebShopHumanDemonstrationProvider:
    """Adapt official WebShop human IL trajectories to InfoSkill SFT rows."""

    environment = "webshop"
    source_split = WebShopSplit.TRAIN.value

    def __init__(
        self,
        demonstrations: str | Path,
        human_goals: str | Path,
        *,
        split: WebShopSplit | str = WebShopSplit.TRAIN,
        history_limit: int = 2,
    ) -> None:
        self.demonstrations_path = Path(demonstrations)
        self.human_goals_path = Path(human_goals)
        split_value = split.value if isinstance(split, WebShopSplit) else str(split)
        split_value = split_value.strip().lower()
        self.split = (
            WebShopSplit.VALIDATION
            if split_value == "eval"
            else WebShopSplit(split_value)
        )
        self.source_split = self.split.value
        if history_limit < 0:
            raise ValueError("history_limit must be non-negative")
        self.history_limit = history_limit

    def trajectories(self) -> tuple[DemonstrationTrajectory, ...]:
        goals = _load_goals(self.human_goals_path)
        goal_lookup: dict[str, int] = {}
        for index, goal in enumerate(goals):
            normalized = normalize_goal(goal)
            # Match the official baseline's ``human_goals.index`` behavior.
            # A duplicate normalized goal belongs to its first registered index.
            goal_lookup.setdefault(normalized, index)
        result: list[DemonstrationTrajectory] = []
        with self.demonstrations_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                trajectory = self._trajectory(
                    payload,
                    line_number=line_number,
                    goals=goals,
                    goal_lookup=goal_lookup,
                )
                if trajectory is not None:
                    result.append(trajectory)
        if not result:
            raise ValueError(
                f"official WebShop provider returned no {self.split.value} trajectories"
            )
        ids = [item.trajectory_id for item in result]
        if len(ids) != len(set(ids)):
            raise ValueError("official WebShop trajectory IDs must be unique")
        return tuple(result)

    def source_files(self) -> dict[str, Path]:
        return {
            "human_demonstrations": self.demonstrations_path,
            "human_goals": self.human_goals_path,
        }

    def _trajectory(
        self,
        payload: object,
        *,
        line_number: int,
        goals: tuple[str, ...],
        goal_lookup: dict[str, int],
    ) -> DemonstrationTrajectory | None:
        if not isinstance(payload, dict):
            raise ValueError(f"WebShop demonstration line {line_number} must be an object")
        states = _required_list(payload, "states", line_number)
        available = _required_list(payload, "available_actions", line_number)
        action_indices = _required_list(payload, "action_idxs", line_number)
        actions = _actions(payload, available, action_indices, line_number)
        lengths = {len(states), len(available), len(action_indices), len(actions)}
        if len(lengths) != 1 or not states:
            raise ValueError(
                f"WebShop demonstration arrays differ in length at line {line_number}"
            )
        goal_key = normalize_goal(states[0])
        if goal_key not in goal_lookup:
            raise ValueError(
                f"WebShop demonstration goal is absent from human_goals at line {line_number}"
            )
        goal_index = goal_lookup[goal_key]
        if split_for_goal_index(goal_index, goal_count=len(goals)) is not self.split:
            return None
        history: list[tuple[str, str]] = []
        steps: list[DemonstrationStep] = []
        task_description = task_description_from_state(states[0])
        for step_index, (state, candidates, action) in enumerate(
            zip(states, available, actions, strict=True)
        ):
            observation = observation_from_state(state, goal_key)
            action_text = _required_text(action, "action", line_number)
            admissible = normalize_available_actions(
                candidates,
                demonstrated_action=action_text,
            )
            if not _action_is_permitted(action_text, admissible):
                raise ValueError(
                    "WebShop demonstrated action is not compatible with available "
                    f"actions at line {line_number}, step {step_index}"
                )
            prompt = render_webshop_policy_message(
                task_description=task_description,
                current_observation=observation,
                available_actions=admissible,
                history=history,
                history_limit=self.history_limit,
            )
            steps.append(DemonstrationStep(prompt=prompt, action=action_text))
            history.append((observation, action_text))
        return DemonstrationTrajectory(
            trajectory_id=f"webshop-human-goal-{goal_index:05d}-line-{line_number:05d}",
            environment=self.environment,
            steps=tuple(steps),
        )


def normalize_goal(value: object) -> str:
    text = _required_text(value, "goal", 0).lower().replace('"', "").replace("'", "")
    text = text.replace("amazon shopping game\ninstruction:", "")
    text = text.replace("webshop\ninstruction:", "")
    text = text.replace("amazon shopping game [sep] instruction: [sep] ", "")
    text = text.replace("webshop [sep] instruction: [sep] ", "")
    text = text.replace("\n[button] search [button_]", "").strip()
    if ", and price lower than" in text:
        text = text.split(", and price lower than", 1)[0]
    return " ".join(text.split())


def task_description_from_state(value: object) -> str:
    """Recover the full instruction, retaining constraints such as price."""

    text = _required_text(value, "state", 0)
    lowered = text.lower()
    for marker in (
        "amazon shopping game\ninstruction:",
        "webshop\ninstruction:",
    ):
        position = lowered.find(marker)
        if position >= 0:
            instruction = text[position + len(marker) :]
            search_marker = instruction.lower().find("\n[button] search [button_]")
            if search_marker >= 0:
                instruction = instruction[:search_marker]
            return _required_text(instruction, "task description", 0)
    parts = text.split(" [SEP] ")
    for index, part in enumerate(parts[:-1]):
        if part.strip().lower() == "instruction:":
            return _required_text(parts[index + 1], "task description", 0)
    raise ValueError("WebShop demonstration state does not contain an instruction")


def observation_from_state(value: object, normalized_goal: str) -> str:
    text = _required_text(value, "state", 0)
    lowered = text.lower()
    search_marker = lowered.find("\n[button] search [button_]")
    if search_marker >= 0 and normalize_goal(text) == normalized_goal:
        return text[search_marker + 1 :].strip()
    parts = text.split(" [SEP] ")
    normalized_parts = [normalize_goal(part) for part in parts]
    try:
        goal_position = normalized_parts.index(normalized_goal)
    except ValueError:
        return text
    remainder = parts[goal_position + 1 :]
    return " [SEP] ".join(f"'{part}'" for part in remainder).strip() or text


def normalize_available_actions(
    value: object,
    *,
    demonstrated_action: str | None = None,
) -> tuple[str, ...]:
    if isinstance(value, dict):
        unknown = set(value) - {"has_search_bar", "clickables"}
        if unknown:
            raise ValueError(f"unknown WebShop available-action keys: {sorted(unknown)}")
        actions: list[str] = []
        if value.get("has_search_bar"):
            actions.append("search[<your query>]")
        clickables = value.get("clickables", [])
        if not isinstance(clickables, list):
            raise ValueError("WebShop clickables must be a list")
        actions.extend(f"click[{_required_text(item, 'clickable', 0)}]" for item in clickables)
    elif isinstance(value, list):
        actions = []
        for item in value:
            candidate = _required_text(item, "available action", 0)
            if candidate.startswith("search["):
                candidate = "search[<your query>]"
            elif not candidate.startswith("click["):
                candidate = f"click[{candidate}]"
            actions.append(candidate)
    else:
        raise ValueError("WebShop available actions must be a list or object")
    if demonstrated_action and demonstrated_action.startswith("search["):
        actions = [item for item in actions if item != "search[<your query>]"]
        actions.insert(0, "search[<your query>]")
    if not actions:
        raise ValueError("WebShop step has no available actions")
    return tuple(dict.fromkeys(actions))


def _actions(
    payload: dict[str, object],
    available: list[object],
    action_indices: list[object],
    line_number: int,
) -> list[object]:
    for key in ("actions_translate", "actions"):
        values = payload.get(key)
        if isinstance(values, list) and len(values) == len(available):
            return values
    resolved: list[object] = []
    for step_index, (candidates, raw_index) in enumerate(
        zip(available, action_indices, strict=True)
    ):
        if not isinstance(raw_index, int):
            raise ValueError(
                f"WebShop action index must be an integer at line {line_number}"
            )
        if raw_index < 0:
            raise ValueError(
                "WebShop search demonstrations require actions or actions_translate "
                f"at line {line_number}, step {step_index}"
            )
        if not isinstance(candidates, list) or raw_index >= len(candidates):
            raise ValueError(
                f"WebShop action index is out of range at line {line_number}, step {step_index}"
            )
        resolved.append(candidates[raw_index])
    return resolved


def _action_is_permitted(action: str, available: tuple[str, ...]) -> bool:
    if action.startswith("search[") and "search[<your query>]" in available:
        return True
    return action in available


def _load_goals(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("human_goals.json must contain a non-empty list")
    return tuple(_required_text(value, "human goal", 0) for value in payload)


def _required_list(
    payload: dict[str, object], field: str, line_number: int
) -> list[object]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise ValueError(f"WebShop demonstration requires {field} at line {line_number}")
    return value


def _required_text(value: object, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        location = f" at line {line_number}" if line_number else ""
        raise ValueError(f"WebShop {field} must be non-empty text{location}")
    return value.strip()
