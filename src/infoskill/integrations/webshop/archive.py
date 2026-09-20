from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile


_FIXED_GOAL_INDEX = re.compile(r"_fixed_(?:100|200)_(\d+)\.jsonl$")


def audit_official_human_archive(archive: str | Path) -> dict[str, object]:
    """Audit the official raw WebShop sessions without extracting user content."""

    path = Path(archive).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"WebShop human archive is missing: {path}")
    page_counts: Counter[str] = Counter()
    terminal_page_counts: Counter[str] = Counter()
    page_sequences: Counter[str] = Counter()
    row_key_sets: Counter[str] = Counter()
    goal_key_sets: Counter[str] = Counter()
    content_key_sets: Counter[str] = Counter()
    trajectory_lengths: list[int] = []
    fixed_goal_indices: list[int] = []
    reward_values: list[float] = []
    invalid_rows: list[dict[str, object]] = []
    ignored_entries = 0
    empty_trajectories = 0
    files_with_done = 0
    try:
        with ZipFile(path) as bundle:
            members = bundle.infolist()
            trajectory_members = []
            for member in members:
                if (
                    member.is_dir()
                    or member.filename.startswith("__MACOSX/")
                    or not member.filename.endswith(".jsonl")
                ):
                    ignored_entries += 1
                    continue
                trajectory_members.append(member)
            for member in sorted(trajectory_members, key=lambda item: item.filename):
                match = _FIXED_GOAL_INDEX.search(member.filename)
                if match:
                    fixed_goal_indices.append(int(match.group(1)))
                rows: list[dict[str, object]] = []
                for line_number, raw_line in enumerate(
                    bundle.read(member).decode("utf-8").splitlines(),
                    start=1,
                ):
                    if not raw_line.strip():
                        continue
                    try:
                        payload = json.loads(raw_line)
                    except json.JSONDecodeError as error:
                        invalid_rows.append(
                            _invalid_row(member.filename, line_number, str(error))
                        )
                        continue
                    if not isinstance(payload, dict):
                        invalid_rows.append(
                            _invalid_row(
                                member.filename,
                                line_number,
                                "JSON value is not an object",
                            )
                        )
                        continue
                    rows.append(payload)
                    row_key_sets[_key_set(payload)] += 1
                    page = payload.get("page")
                    page_counts[str(page) if page is not None else "<missing>"] += 1
                    goal = payload.get("goal")
                    if isinstance(goal, dict):
                        goal_key_sets[_key_set(goal)] += 1
                    else:
                        goal_key_sets["<missing-or-non-object>"] += 1
                    content = payload.get("content")
                    if isinstance(content, dict):
                        content_key_sets[_key_set(content)] += 1
                    else:
                        content_key_sets["<missing-or-non-object>"] += 1
                if not rows:
                    empty_trajectories += 1
                    continue
                trajectory_lengths.append(len(rows))
                sequence = " -> ".join(str(row.get("page", "<missing>")) for row in rows)
                page_sequences[sequence] += 1
                terminal = rows[-1]
                terminal_page = str(terminal.get("page", "<missing>"))
                terminal_page_counts[terminal_page] += 1
                if terminal_page == "done":
                    files_with_done += 1
                reward = terminal.get("reward")
                if isinstance(reward, (int, float)) and not isinstance(reward, bool):
                    reward_values.append(float(reward))
    except (BadZipFile, UnicodeDecodeError) as error:
        raise ValueError(f"invalid WebShop human archive: {error}") from error

    report: dict[str, object] = {
        "schema_version": 1,
        "archive": str(path),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": _sha256(path),
        "zip_entry_count": len(members),
        "ignored_entry_count": ignored_entries,
        "trajectory_file_count": len(trajectory_members),
        "empty_trajectory_count": empty_trajectories,
        "invalid_row_count": len(invalid_rows),
        "invalid_rows": invalid_rows[:20],
        "row_count": sum(trajectory_lengths),
        "trajectory_lengths": _numeric_summary(trajectory_lengths),
        "page_counts": _sorted_counts(page_counts),
        "terminal_page_counts": _sorted_counts(terminal_page_counts),
        "common_page_sequences": _top_counts(page_sequences, limit=20),
        "row_key_sets": _sorted_counts(row_key_sets),
        "goal_key_sets": _sorted_counts(goal_key_sets),
        "content_key_sets": _sorted_counts(content_key_sets),
        "files_with_done": files_with_done,
        "terminal_rewards": {
            **_numeric_summary(reward_values),
            "positive_count": sum(value > 0.0 for value in reward_values),
            "perfect_count": sum(value == 1.0 for value in reward_values),
        },
        "fixed_goal_indices": {
            "file_count": len(fixed_goal_indices),
            "unique_count": len(set(fixed_goal_indices)),
            "min": min(fixed_goal_indices) if fixed_goal_indices else None,
            "max": max(fixed_goal_indices) if fixed_goal_indices else None,
        },
    }
    report["archive_readable"] = (
        bool(trajectory_members)
        and not invalid_rows
        and not empty_trajectories
        and len(trajectory_lengths) == len(trajectory_members)
    )
    return report


def _key_set(payload: dict[str, object]) -> str:
    return ",".join(sorted(str(key) for key in payload)) or "<empty>"


def _invalid_row(filename: str, line_number: int, error: str) -> dict[str, object]:
    return {"file": filename, "line": line_number, "error": error}


def _numeric_summary(values: list[int] | list[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def _sorted_counts(counts: Counter[str]) -> dict[str, int]:
    return {key: counts[key] for key in sorted(counts)}


def _top_counts(counts: Counter[str], *, limit: int) -> list[dict[str, object]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:limit]
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
