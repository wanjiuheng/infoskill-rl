from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path


TASK_TYPES = (
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
)


def build_planner_skill_bank(
    grounding_samples: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    source = Path(grounding_samples)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("planner skill bank requires successful grounding samples")
    trajectories = {str(row["task_id"]) for row in rows}
    action_verbs = Counter(str(row["expert_action"]).split(maxsplit=1)[0] for row in rows)
    trajectories_by_type: dict[str, set[str]] = {}
    for row in rows:
        trajectories_by_type.setdefault(str(row["task_type"]), set()).add(
            str(row["task_id"])
        )
    payload = {
        "metadata": {
            "schema_version": 2,
            "source": "verified_alfworld_planner_grounding",
            "category_gate": True,
            "source_samples_sha256": _sha256(source),
            "source_successful_trajectories": len(trajectories),
            "source_samples": len(rows),
            "action_verb_counts": dict(sorted(action_verbs.items())),
            "task_type_trajectory_counts": {
                key: len(value) for key, value in sorted(trajectories_by_type.items())
            },
        },
        "general_skills": _general_skills(),
        "task_specific_skills": _task_skills(rows),
        "common_mistakes": _mistakes(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    provenance = {
        "schema_version": 2,
        "artifact": "INFO-SKILL ALFWorld planner-derived phased skill library",
        "skill_bank_sha256_algorithm": "canonical-json-sha256-v1",
        "skill_bank_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_split": "train",
        "trajectory_count": len(trajectories),
        "generation_method": (
            "fixed task phase schema with deterministic action grammar extraction "
            "from verified planner grounding"
        ),
        "source_samples_sha256": _sha256(source),
    }
    _atomic_json(destination.with_suffix(".manifest.json"), provenance)
    return dict(payload["metadata"])


def _general_skills() -> list[dict[str, str]]:
    return [
        {
            "skill_id": "general_observe_then_act",
            "title": "Ground every action in the current state",
            "principle": "Choose only an exact admissible command and update the plan after every observation",
            "when_to_apply": "Every ALFWorld step",
            "phase": "all",
            "anti_loop_rule": "Do not repeat an action unless the observation changed",
        },
        {
            "skill_id": "general_inventory_control",
            "title": "Track the held object explicitly",
            "principle": "Never navigate as if an object was picked up until the observation or inventory confirms it",
            "when_to_apply": "Before transport or placement",
            "phase": "all",
        },
    ]


def _skill(skill_id: str, title: str, phase: str, principle: str, stop: str) -> dict[str, str]:
    return {
        "skill_id": skill_id,
        "title": title,
        "phase": phase,
        "principle": principle,
        "when_to_apply": f"During the {phase} phase",
        "stop_condition": stop,
        "anti_loop_rule": "If the same action does not change the observation, re-observe and choose another admissible command",
    }


def _task_skills(rows: list[dict[str, object]]) -> dict[str, list[dict[str, str]]]:
    simple = {
        "look_at_obj_in_light": [_skill("light_locate_and_use", "Locate object and activate the lamp", "locate_transform", "Find the target, carry it to the light if needed, and use the named lamp command", "The goal reports success")],
        "pick_and_place_simple": [_skill("place_pick_then_deliver", "Pick once and deliver once", "pick_deliver", "Locate the target, take it, navigate to the destination, then use the exact move/put command", "The object is in the target receptacle")],
        "pick_clean_then_place_in_recep": [_skill("clean_transform_then_deliver", "Clean before delivery", "transform_deliver", "Take the object, use the sink cleaning command, verify the transformed observation, then deliver", "The clean object is in the target receptacle")],
        "pick_cool_then_place_in_recep": [_skill("cool_transform_then_deliver", "Cool before delivery", "transform_deliver", "Take the object, use the fridge cooling command, verify the transformed observation, then deliver", "The cooled object is in the target receptacle")],
        "pick_heat_then_place_in_recep": [_skill("heat_transform_then_deliver", "Heat before delivery", "transform_deliver", "Take the object, use the microwave heating command, verify the transformed observation, then deliver", "The heated object is in the target receptacle")],
    }
    simple["pick_two_obj_and_place"] = [
        _skill("pick_two_first_object", "Acquire exactly one first object", "first_object", "Locate and take one matching object; retain the second object's likely location", "Inventory confirms the first object"),
        _skill("pick_two_first_delivery", "Deliver the first object", "first_delivery", "Navigate to the target receptacle and place the held first object", "Observation confirms one object delivered and inventory is empty"),
        _skill("pick_two_second_object", "Return for a distinct second object", "second_object", "Navigate back, locate a second instance, and take it; do not re-handle the delivered instance", "Inventory confirms the second object"),
        _skill("pick_two_second_delivery", "Deliver the second object and stop", "second_delivery", "Return to the same target and place the second object", "Goal success is reported; issue no further action"),
    ]
    templates_by_type: dict[str, Counter[str]] = {}
    for row in rows:
        task_type = str(row["task_type"])
        templates_by_type.setdefault(task_type, Counter())[
            _action_template(str(row["expert_action"]))
        ] += 1
    for task_type, skills in simple.items():
        observed = "; ".join(
            template
            for template, _ in templates_by_type.get(task_type, Counter()).most_common(12)
        )
        for skill in skills:
            skill["action_templates"] = observed
    return simple


def _action_template(action: str) -> str:
    """Keep planner command grammar while removing instance-specific IDs."""

    normalized = re.sub(r"\b\d+\b", "{id}", action.strip().lower())
    patterns = (
        (r"^go to .+$", "go to <receptacle>"),
        (r"^(open|close) .+$", r"\1 <receptacle>"),
        (r"^take .+ from .+$", "take <object> from <receptacle>"),
        (r"^(move|put) .+ (to|in|on) .+$", "move <object> to <receptacle>"),
        (r"^(clean|cool|heat) .+ with .+$", r"\1 <object> with <appliance>"),
        (r"^(switch on|switch off|use) .+$", r"\1 <object>"),
    )
    for pattern, replacement in patterns:
        if re.match(pattern, normalized):
            return re.sub(pattern, replacement, normalized)
    return normalized


def _mistakes() -> list[dict[str, str]]:
    return [
        {"mistake_id": "mistake_repeat_without_change", "description": "Repeating an action after an unchanged observation", "why_it_happens": "The agent ignores failed preconditions", "how_to_avoid": "Use the new admissible list and change the plan"},
        {"mistake_id": "mistake_pick_two_same_instance", "description": "Trying to deliver the same object instance twice", "why_it_happens": "The first delivery is not tracked", "how_to_avoid": "After first delivery, return and take a distinct numbered instance"},
        {"mistake_id": "mistake_transform_unverified", "description": "Delivering before the clean/cool/heat transform is confirmed", "why_it_happens": "The transformation stage is skipped", "how_to_avoid": "Require a transformed observation before delivery"},
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rewrite_grounding_skill_ids(
    grounding_directory: str | Path,
    skill_bank: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Derive a grounding run whose candidate IDs match the new skill bank."""

    source = Path(grounding_directory).expanduser().resolve()
    destination = Path(output_directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"derived grounding already exists: {destination}")
    bank = json.loads(Path(skill_bank).read_text(encoding="utf-8"))
    general = [str(item["skill_id"]) for item in bank["general_skills"]]
    mistakes = [
        str(item.get("mistake_id") or item["skill_id"])
        for item in bank["common_mistakes"]
    ]
    by_type = {
        task_type: [str(item["skill_id"]) for item in skills]
        for task_type, skills in bank["task_specific_skills"].items()
    }
    rows: list[dict[str, object]] = []
    for line in (source / "grounding_samples.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_type = str(row["task_type"])
        if task_type not in by_type:
            raise ValueError(f"planner skill bank has no task type: {task_type}")
        row["state"]["candidate_skill_ids"] = general + by_type[task_type] + mistakes
        rows.append(row)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    source_checksums = dict(manifest.get("source_checksums", {}))
    source_checksums.update(
        {
            "parent_grounding_manifest": _sha256(source / "manifest.json"),
            "parent_grounding_samples": _sha256(source / "grounding_samples.jsonl"),
            "skill_bank": _sha256(Path(skill_bank)),
        }
    )
    manifest["source_checksums"] = source_checksums
    manifest["derived_candidate_skill_ids"] = True
    manifest["parent_grounding_root"] = str(source)
    destination.mkdir(parents=True, exist_ok=False)
    _atomic_json(destination / "manifest.json", manifest)
    _atomic_jsonl(destination / "grounding_samples.jsonl", rows)
    quarantine = source / "quarantine.jsonl"
    if quarantine.is_file():
        (destination / "quarantine.jsonl").write_bytes(quarantine.read_bytes())
    return {
        "root": str(destination),
        "sample_count": len(rows),
        "trajectory_count": len({str(row["task_id"]) for row in rows}),
        "skill_bank_sha256": source_checksums["skill_bank"],
    }


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
