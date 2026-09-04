from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from .library import FixedSkillLibrary, SkillRecord


class TextEmbeddingEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True, slots=True)
class RetrievedSkill:
    record: SkillRecord
    score: float | None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    mode: str
    query: str
    skills: tuple[RetrievedSkill, ...]

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(item.record.skill_id for item in self.skills)


class SentenceTransformerEncoder:
    """Lazy local SentenceTransformer adapter used by embedding retrieval."""

    def __init__(self, model_path: str, *, device: str | None = None) -> None:
        self._model_path = model_path
        self._device = device
        self._model: object | None = None

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError("embedding retrieval requires sentence-transformers") from error
            self._model = SentenceTransformer(self._model_path, device=self._device)
        result = self._model.encode(  # type: ignore[attr-defined]
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return result.tolist()


class EmbeddingRetriever:
    """SkillRL-compatible cross-category cosine ranking with deterministic ties."""

    def __init__(
        self,
        library: FixedSkillLibrary,
        encoder: TextEmbeddingEncoder,
        *,
        general_top_k: int = 6,
        task_top_k: int = 6,
        mistake_count: int = 5,
    ) -> None:
        if min(general_top_k, task_top_k, mistake_count) < 0:
            raise ValueError("retrieval counts cannot be negative")
        self._library = library
        self._encoder = encoder
        self._general_top_k = general_top_k
        self._task_top_k = task_top_k
        self._mistake_count = mistake_count
        ranked_pool = library.general + library.task_specific
        self._pool = ranked_pool
        self._embeddings = tuple(_unit(vector) for vector in encoder.encode([item.text for item in ranked_pool]))
        if len(self._embeddings) != len(ranked_pool):
            raise ValueError("embedding encoder returned an unexpected number of vectors")

    def retrieve(self, query: str) -> RetrievalResult:
        query_vector = _unit(self._encoder.encode([query])[0])
        scored = [
            (sum(left * right for left, right in zip(vector, query_vector)), index, record)
            for index, (record, vector) in enumerate(zip(self._pool, self._embeddings))
        ]
        general = _top_kind(scored, "general", self._general_top_k)
        task = _top_kind(scored, "task_specific", self._task_top_k)
        mistakes = tuple(RetrievedSkill(record, None) for record in self._library.mistakes[: self._mistake_count])
        return RetrievalResult("embedding", query, general + task + mistakes)


class TemplateRetriever:
    def __init__(
        self,
        library: FixedSkillLibrary,
        *,
        general_count: int = 6,
        task_count: int = 6,
        mistake_count: int = 5,
    ) -> None:
        self._library = library
        self._general_count = general_count
        self._task_count = task_count
        self._mistake_count = mistake_count

    def retrieve(self, query: str) -> RetrievalResult:
        category = _detect_alfworld_category(query)
        general = tuple(RetrievedSkill(item, None) for item in self._library.general[: self._general_count])
        task = tuple(
            RetrievedSkill(item, None)
            for item in self._library.task_specific
            if item.category == category
        )[: self._task_count]
        mistakes = tuple(RetrievedSkill(item, None) for item in self._library.mistakes[: self._mistake_count])
        return RetrievalResult("template", query, general + task + mistakes)


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in values))
    if not values or norm == 0.0:
        raise ValueError("embedding vectors must be non-empty and non-zero")
    return tuple(value / norm for value in values)


def _top_kind(
    scored: Sequence[tuple[float, int, SkillRecord]], kind: str, count: int
) -> tuple[RetrievedSkill, ...]:
    selected = sorted(
        (item for item in scored if item[2].kind == kind),
        key=lambda item: (-item[0], item[1]),
    )[:count]
    return tuple(RetrievedSkill(record, score) for score, _, record in selected)


def _detect_alfworld_category(goal: str) -> str:
    normalized = goal.lower()
    if "look at" in normalized and "under" in normalized:
        return "look_at_obj_in_light"
    if "clean" in normalized:
        return "clean"
    if "heat" in normalized:
        return "heat"
    if "cool" in normalized:
        return "cool"
    if "examine" in normalized or "find" in normalized:
        return "examine"
    return "pick_and_place"
