from __future__ import annotations

import json
from pathlib import Path

from infoskill.episode import TaskSpec


ALFWORLD_TASK_TYPES = (
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read ALFWorld JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"ALFWorld JSON root must be an object: {path}")
    return payload


def _human_goal(trajectory: dict[str, object], path: Path) -> str:
    try:
        annotations = trajectory["turk_annotations"]
        assert isinstance(annotations, dict)
        entries = annotations["anns"]
        assert isinstance(entries, list) and entries
        first = entries[0]
        assert isinstance(first, dict)
        goal = first["task_desc"]
        assert isinstance(goal, str) and goal.strip()
    except (KeyError, AssertionError, TypeError) as error:
        raise ValueError(f"missing ALFWorld human task description: {path}") from error
    return goal.strip()


def discover_tasks(data_root: str | Path, *, split: str) -> tuple[TaskSpec, ...]:
    """Discover supported, solvable ALFWorld games in a deterministic order."""

    root = Path(data_root).expanduser().resolve()
    split_root = root / "json_2.1.1" / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"ALFWorld split does not exist: {split_root}")

    tasks: list[TaskSpec] = []
    for trajectory_path in sorted(split_root.rglob("traj_data.json")):
        normalized_path = trajectory_path.as_posix().lower()
        if "movable" in normalized_path or "sliced" in normalized_path:
            continue
        trajectory = _read_json(trajectory_path)
        task_type = trajectory.get("task_type")
        if task_type not in ALFWORLD_TASK_TYPES:
            continue
        game_path = trajectory_path.with_name("game.tw-pddl")
        if not game_path.is_file():
            continue
        game = _read_json(game_path)
        if game.get("solvable") is not True:
            continue
        tasks.append(
            TaskSpec(
                task_id=game_path.relative_to(root).as_posix(),
                split=split,
                task_type=str(task_type),
                goal=_human_goal(trajectory, trajectory_path),
                environment_path=str(game_path.resolve()),
                trajectory_path=str(trajectory_path.resolve()),
            )
        )
    return tuple(tasks)
