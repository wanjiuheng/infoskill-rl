from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class SkillRecord:
    skill_id: str
    kind: str
    category: str | None
    title: str
    text: str
    fields: Mapping[str, str]


class FixedSkillLibrary:
    """Validated immutable view of a SkillRL Claude-style skill bank."""

    def __init__(
        self,
        *,
        general: tuple[SkillRecord, ...],
        task_specific: tuple[SkillRecord, ...],
        mistakes: tuple[SkillRecord, ...],
        source_path: Path,
        source_sha256: str,
        metadata: Mapping[str, object],
    ) -> None:
        all_records = general + task_specific + mistakes
        ids = [record.skill_id for record in all_records]
        if len(ids) != len(set(ids)):
            raise ValueError("skill bank contains duplicate skill IDs")
        self.general = general
        self.task_specific = task_specific
        self.mistakes = mistakes
        self.source_path = source_path
        self.source_sha256 = source_sha256
        self.metadata = MappingProxyType(dict(metadata))
        self._by_id = MappingProxyType({record.skill_id: record for record in all_records})

    @classmethod
    def load(cls, path: str | Path) -> "FixedSkillLibrary":
        source = Path(path).expanduser().resolve()
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("skill bank root must be a JSON object")

        general = tuple(
            _parse_skill(item, kind="general", category=None)
            for item in _list(payload, "general_skills")
        )
        task_specific: list[SkillRecord] = []
        task_payload = payload.get("task_specific_skills", {})
        if not isinstance(task_payload, dict):
            raise ValueError("task_specific_skills must be a JSON object")
        for category, items in task_payload.items():
            if not isinstance(items, list):
                raise ValueError(f"task-specific category {category!r} must be a list")
            task_specific.extend(
                _parse_skill(item, kind="task_specific", category=str(category)) for item in items
            )
        mistakes = tuple(
            _parse_mistake(item, index=index)
            for index, item in enumerate(_list(payload, "common_mistakes"), start=1)
        )
        if not general or not task_specific or not mistakes:
            raise ValueError("skill bank must contain general, task-specific, and mistake entries")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("skill bank metadata must be a JSON object")
        return cls(
            general=general,
            task_specific=tuple(task_specific),
            mistakes=mistakes,
            source_path=source,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            metadata=metadata,
        )

    def get(self, skill_id: str) -> SkillRecord:
        try:
            return self._by_id[skill_id]
        except KeyError as error:
            raise KeyError(f"unknown skill ID: {skill_id}") from error


def _list(payload: Mapping[str, object], field: str) -> list[object]:
    value = payload.get(field, [])
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list")
    return value


def _string_fields(item: object) -> dict[str, str]:
    if not isinstance(item, dict):
        raise ValueError("each skill must be a JSON object")
    return {str(key): str(value).strip() for key, value in item.items() if value is not None}


def _parse_skill(item: object, *, kind: str, category: str | None) -> SkillRecord:
    fields = _string_fields(item)
    skill_id = fields.get("skill_id", "")
    title = fields.get("title", "")
    components = [fields.get(name, "") for name in ("title", "principle", "when_to_apply")]
    text = ". ".join(component for component in components if component)
    if not skill_id or not text:
        raise ValueError("skill entries require skill_id and semantic text")
    return SkillRecord(skill_id, kind, category, title, text, MappingProxyType(fields))


def _parse_mistake(item: object, *, index: int) -> SkillRecord:
    fields = _string_fields(item)
    skill_id = fields.get("mistake_id") or f"mistake_{index:03d}"
    components = [fields.get(name, "") for name in ("description", "why_it_happens", "how_to_avoid")]
    text = ". ".join(component for component in components if component)
    if not text:
        raise ValueError("mistake entries require semantic text")
    return SkillRecord(
        skill_id,
        "common_mistake",
        None,
        fields.get("description", skill_id),
        text,
        MappingProxyType(fields),
    )
