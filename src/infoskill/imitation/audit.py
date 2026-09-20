from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from infoskill.integrations.webshop.demonstrations import (
    REGISTERED_HUMAN_DEMONSTRATIONS_SHA256,
    REGISTERED_HUMAN_GOALS_SHA256,
    REGISTERED_TRAIN_TRAJECTORY_COUNT,
)


_RESPONSE = re.compile(
    r"<think>.+</think>\n<action>(.+)</action>",
    re.DOTALL,
)
_ADMISSIBLE_BLOCK = re.compile(
    r"Your admissible actions of the current situation are:\s*\n?\[\n(.*?)\n\]\.",
    re.DOTALL,
)
_QUOTED_ACTION = re.compile(r"^'(.+)',$")


def audit_prepared_imitation_data(
    data_directory: str | Path,
    *,
    tokenizer: object,
    max_length: int,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(data_directory).expanduser().resolve()
    manifest_path = root / "manifest.json"
    train_path = root / "train.jsonl"
    validation_path = root / "validation.jsonl"
    report: dict[str, object]
    failure_stage = "load_manifest"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("WebShop imitation manifest must be an object")
        failure_stage = "load_train"
        train_rows = _read_jsonl(train_path)
        failure_stage = "load_validation"
        validation_rows = _read_jsonl(validation_path)
        failure_stage = "audit_rows"
        report = audit_imitation_rows(
            manifest=manifest,
            train_rows=train_rows,
            validation_rows=validation_rows,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        protocol_failures = registered_webshop_manifest_failures(manifest)
        if protocol_failures:
            report["failure_reasons"] = sorted(
                set(report["failure_reasons"]) | set(protocol_failures)
            )
            report["passed"] = False
        failure_stage = "checksum_sources"
        report["source_checksums"] = {
            "manifest": _sha256_file(manifest_path),
            "train": _sha256_file(train_path),
            "validation": _sha256_file(validation_path),
        }
    except Exception as error:
        report = {
            "schema_version": 1,
            "environment": "webshop",
            "passed": False,
            "failure_reasons": [f"{failure_stage}_failed"],
            "failure_stage": failure_stage,
            "error_type": type(error).__name__,
            "tokenizer": str(
                getattr(tokenizer, "name_or_path", type(tokenizer).__name__)
            ),
            "max_length": max_length,
        }
    report["data_directory"] = str(root)
    if output_path is not None:
        _atomic_json(Path(output_path), report)
    return report


def audit_imitation_rows(
    *,
    manifest: dict[str, object],
    train_rows: Sequence[dict[str, object]],
    validation_rows: Sequence[dict[str, object]],
    tokenizer: object,
    max_length: int,
) -> dict[str, object]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    failures: set[str] = set()
    if manifest.get("environment") != "webshop":
        failures.add("environment_not_webshop")

    split_rows = {
        "train": tuple(train_rows),
        "validation": tuple(validation_rows),
    }
    expected_counts = {
        "sample_count": len(train_rows) + len(validation_rows),
        "train_sample_count": len(train_rows),
        "validation_sample_count": len(validation_rows),
        "train_trajectory_count": len(_trajectory_ids(train_rows)),
        "validation_trajectory_count": len(_trajectory_ids(validation_rows)),
        "trajectory_count": len(
            _trajectory_ids(train_rows) | _trajectory_ids(validation_rows)
        ),
    }
    if any(manifest.get(key) != value for key, value in expected_counts.items()):
        failures.add("manifest_count_mismatch")

    train_ids = _trajectory_ids(train_rows)
    validation_ids = _trajectory_ids(validation_rows)
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        failures.add("cross_split_trajectory_leakage")

    action_families: Counter[str] = Counter()
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    combined_lengths: list[int] = []
    prompt_exhausts_response = 0
    over_limit = 0
    split_reports: dict[str, object] = {}
    row_content_locations: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    trajectory_content_locations: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, rows in split_rows.items():
        trajectory_steps: dict[str, list[int]] = defaultdict(list)
        trajectory_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
        seen_step_keys: set[tuple[str, int]] = set()
        for row in rows:
            task_id = str(row.get("task_id", "")).strip()
            step_index = row.get("step_index")
            prompt = str(row.get("prompt", ""))
            response = str(row.get("response", ""))
            if not task_id or not isinstance(step_index, int) or step_index < 0:
                failures.add("invalid_task_or_step_identity")
                continue
            if row.get("task_type") != "webshop":
                failures.add("row_environment_not_webshop")
            if not prompt.strip():
                failures.add("empty_prompt")
            if not response.strip():
                failures.add("empty_response")
            step_key = (task_id, step_index)
            if step_key in seen_step_keys:
                failures.add("duplicate_trajectory_step")
            seen_step_keys.add(step_key)
            trajectory_steps[task_id].append(step_index)
            trajectory_rows[task_id].append(row)
            row_content_locations[_row_content_sha256(row)].append(
                (split, task_id, step_index)
            )

            response_match = _RESPONSE.fullmatch(response)
            if response_match is None:
                failures.add("invalid_response_contract")
                continue
            action = response_match.group(1).strip()
            action_families[_action_family(action)] += 1
            if not _action_is_admissible(prompt, action):
                failures.add("demonstrated_action_not_admissible")

            prompt_count = len(_chat_ids(tokenizer, prompt))
            response_count = len(_token_ids(tokenizer, response)) + 1
            total = prompt_count + response_count
            prompt_lengths.append(prompt_count)
            response_lengths.append(response_count)
            combined_lengths.append(total)
            if prompt_count >= max_length:
                prompt_exhausts_response += 1
            if total > max_length:
                over_limit += 1

        trajectory_lengths = [len(steps) for steps in trajectory_steps.values()]
        for steps in trajectory_steps.values():
            if sorted(steps) != list(range(len(steps))):
                failures.add("non_contiguous_trajectory_steps")
        for task_id, task_rows in trajectory_rows.items():
            trajectory_content_locations[
                _trajectory_content_sha256(task_rows)
            ].append((split, task_id))
        split_reports[split] = {
            "sample_count": len(rows),
            "trajectory_count": len(trajectory_steps),
            "trajectory_lengths": _distribution(trajectory_lengths),
        }

    if prompt_exhausts_response:
        failures.add("prompt_exhausts_max_length")
    if over_limit:
        failures.add("examples_exceed_max_length")
    duplicate_row_groups = [
        locations
        for locations in row_content_locations.values()
        if len({(split, task_id) for split, task_id, _ in locations}) > 1
    ]
    duplicate_trajectory_groups = [
        locations
        for locations in trajectory_content_locations.values()
        if len(locations) > 1
    ]
    cross_split_row_content_groups = [
        locations
        for locations in duplicate_row_groups
        if len({split for split, _, _ in locations}) > 1
    ]
    cross_split_trajectory_content_groups = [
        locations
        for locations in duplicate_trajectory_groups
        if len({split for split, _ in locations}) > 1
    ]
    if duplicate_row_groups:
        failures.add("duplicate_row_content")
    if duplicate_trajectory_groups:
        failures.add("duplicate_trajectory_content")
    if cross_split_row_content_groups or cross_split_trajectory_content_groups:
        failures.add("cross_split_content_leakage")
    return {
        "schema_version": 1,
        "environment": "webshop",
        "passed": not failures,
        "failure_reasons": sorted(failures),
        "manifest_counts": expected_counts,
        "splits": split_reports,
        "cross_split_trajectory_count": len(overlap),
        "cross_split_trajectory_ids": overlap,
        "duplicate_row_content_group_count": len(duplicate_row_groups),
        "duplicate_trajectory_content_group_count": len(
            duplicate_trajectory_groups
        ),
        "cross_split_row_content_group_count": len(
            cross_split_row_content_groups
        ),
        "cross_split_trajectory_content_group_count": len(
            cross_split_trajectory_content_groups
        ),
        "cross_split_content_group_count": (
            len(cross_split_row_content_groups)
            + len(cross_split_trajectory_content_groups)
        ),
        "action_family_counts": dict(sorted(action_families.items())),
        "tokenizer": str(getattr(tokenizer, "name_or_path", type(tokenizer).__name__)),
        "max_length": max_length,
        "token_lengths": {
            "prompt": _distribution(prompt_lengths),
            "response_with_eos": _distribution(response_lengths),
            "combined": _distribution(combined_lengths),
            "prompt_exhausts_response_count": prompt_exhausts_response,
            "over_limit_count": over_limit,
        },
    }


def registered_webshop_manifest_failures(
    manifest: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    if manifest.get("provider") != "OfficialWebShopHumanDemonstrationProvider":
        failures.append("unregistered_demonstration_provider")
    if manifest.get("source_split") != "train":
        failures.append("source_split_not_train")
    if manifest.get("trajectory_count") != REGISTERED_TRAIN_TRAJECTORY_COUNT:
        failures.append("registered_trajectory_count_mismatch")
    if manifest.get("expected_trajectory_count") != REGISTERED_TRAIN_TRAJECTORY_COUNT:
        failures.append("expected_trajectory_count_mismatch")
    checksums = manifest.get("source_checksums")
    expected = {
        "human_demonstrations": REGISTERED_HUMAN_DEMONSTRATIONS_SHA256,
        "human_goals": REGISTERED_HUMAN_GOALS_SHA256,
    }
    if not isinstance(checksums, dict) or any(
        checksums.get(name) != digest for name, digest in expected.items()
    ):
        failures.append("registered_source_checksum_mismatch")
    return failures


def validate_webshop_audit_report(
    report: dict[str, object],
    *,
    data_directory: str | Path,
    source_checksums: dict[str, str],
    expected_tokenizer: str | None = None,
    expected_max_length: int | None = None,
) -> None:
    if report.get("passed") is not True:
        raise ValueError("WebShop imitation audit did not pass")
    expected_root = Path(data_directory).expanduser().resolve()
    recorded_root = Path(str(report.get("data_directory", ""))).expanduser().resolve()
    if recorded_root != expected_root:
        raise ValueError("WebShop audit data directory differs from current data")
    if report.get("source_checksums") != source_checksums:
        raise ValueError("WebShop audit source checksums differ from current data")
    if expected_tokenizer is not None and not _same_path_or_value(
        str(report.get("tokenizer", "")), expected_tokenizer
    ):
        raise ValueError("WebShop audit tokenizer differs from training tokenizer")
    if (
        expected_max_length is not None
        and report.get("max_length") != expected_max_length
    ):
        raise ValueError("WebShop audit max_length differs from training max_length")


def _trajectory_ids(rows: Sequence[dict[str, object]]) -> set[str]:
    return {str(row.get("task_id", "")).strip() for row in rows}


def _row_content_sha256(row: dict[str, object]) -> str:
    payload = {
        "task_type": row.get("task_type"),
        "step_index": row.get("step_index"),
        "prompt": row.get("prompt"),
        "response": row.get("response"),
    }
    return _canonical_sha256(payload)


def _trajectory_content_sha256(rows: Sequence[dict[str, object]]) -> str:
    ordered = sorted(rows, key=lambda row: int(row.get("step_index", -1)))
    return _canonical_sha256(
        [
            {
                "task_type": row.get("task_type"),
                "step_index": row.get("step_index"),
                "prompt": row.get("prompt"),
                "response": row.get("response"),
            }
            for row in ordered
        ]
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_path_or_value(left: str, right: str) -> bool:
    if left == right:
        return True
    left_path = Path(left).expanduser()
    right_path = Path(right).expanduser()
    return left_path.is_absolute() and right_path.is_absolute() and (
        left_path.resolve() == right_path.resolve()
    )


def _action_is_admissible(prompt: str, action: str) -> bool:
    matches = tuple(_ADMISSIBLE_BLOCK.finditer(prompt))
    if not matches:
        return False
    actions = {
        match.group(1)
        for line in matches[-1].group(1).splitlines()
        if (match := _QUOTED_ACTION.fullmatch(line.strip())) is not None
    }
    if action.startswith("search["):
        return "search[<your query>]" in actions
    return action in actions


def _action_family(action: str) -> str:
    if action.startswith("search["):
        return "search"
    if action.lower() == "click[buy now]":
        return "purchase"
    return "click"


def _chat_ids(tokenizer: object, prompt: str) -> list[int]:
    values = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _token_ids(tokenizer: object, text: str) -> list[int]:
    values = tokenizer(  # type: ignore[operator]
        text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _distribution(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _percentile(ordered: Sequence[int], quantile: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"imitation row must be an object at {path}:{line_number}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"imitation split is empty: {path}")
    return rows


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
