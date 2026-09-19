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


class EmptyRetriever:
    """Return an explicit empty result for conditioning-control diagnostics."""

    def retrieve(self, query: str) -> RetrievalResult:
        if not query.strip():
            raise ValueError("empty retrieval requires a non-empty query")
        return RetrievalResult(mode="empty", query=query, skills=())


class SentenceTransformerEncoder:
    """Lazy local SentenceTransformer adapter used by embedding retrieval."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = None,
        show_progress_bar: bool = False,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._show_progress_bar = show_progress_bar
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
            show_progress_bar=self._show_progress_bar,
            convert_to_numpy=True,
        )
        return result.tolist()

    def close(self) -> None:
        """Release the lazy model and any CUDA allocator cache before Ray starts."""

        self._model = None
        if self._device is not None and self._device.startswith("cuda"):
            import gc

            gc.collect()
            try:
                import torch
            except ImportError:
                return
            torch.cuda.empty_cache()


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
        self._category_gate = library.metadata.get("category_gate") is True
        ranked_pool = library.general + library.task_specific
        self._pool = ranked_pool
        self._embeddings = tuple(_unit(vector) for vector in encoder.encode([item.text for item in ranked_pool]))
        if len(self._embeddings) != len(ranked_pool):
            raise ValueError("embedding encoder returned an unexpected number of vectors")

    def retrieve(self, query: str) -> RetrievalResult:
        query_vector = _unit(self._encoder.encode([query])[0])
        return _embedding_result(
            query=query,
            query_vector=query_vector,
            pool=self._pool,
            embeddings=self._embeddings,
            mistakes=self._library.mistakes,
            general_top_k=self._general_top_k,
            task_top_k=self._task_top_k,
            mistake_count=self._mistake_count,
            task_category=(
                _detect_alfworld_category(query) if self._category_gate else None
            ),
        )


class PrecomputedEmbeddingRetriever:
    """Batch all registered queries once, then expose immutable lookup retrieval."""

    def __init__(
        self,
        library: FixedSkillLibrary,
        encoder: TextEmbeddingEncoder,
        *,
        queries: Sequence[str],
        general_top_k: int = 6,
        task_top_k: int = 6,
        mistake_count: int = 5,
    ) -> None:
        if min(general_top_k, task_top_k, mistake_count) < 0:
            raise ValueError("retrieval counts cannot be negative")
        unique_queries = tuple(dict.fromkeys(queries))
        if not unique_queries or any(not query.strip() for query in unique_queries):
            raise ValueError("precomputed retrieval requires non-empty queries")
        pool = library.general + library.task_specific
        skill_embeddings = tuple(
            _unit(vector) for vector in encoder.encode([item.text for item in pool])
        )
        if len(skill_embeddings) != len(pool):
            raise ValueError(
                "embedding encoder returned an unexpected number of skill vectors"
            )
        query_embeddings = tuple(
            _unit(vector) for vector in encoder.encode(unique_queries)
        )
        if len(query_embeddings) != len(unique_queries):
            raise ValueError(
                "embedding encoder returned an unexpected number of query vectors"
            )
        self._results = {
            query: _embedding_result(
                query=query,
                query_vector=query_vector,
                pool=pool,
                embeddings=skill_embeddings,
                mistakes=library.mistakes,
                general_top_k=general_top_k,
                task_top_k=task_top_k,
                mistake_count=mistake_count,
                task_category=(
                    _detect_alfworld_category(query)
                    if library.metadata.get("category_gate") is True
                    else None
                ),
            )
            for query, query_vector in zip(unique_queries, query_embeddings)
        }

    def retrieve(self, query: str) -> RetrievalResult:
        try:
            return self._results[query]
        except KeyError as error:
            raise KeyError(
                f"retrieval query was not precomputed: {query!r}"
            ) from error


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
        if not task:
            legacy_category = _detect_legacy_alfworld_category(query)
            task = tuple(
                RetrievedSkill(item, None)
                for item in self._library.task_specific
                if item.category == legacy_category
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


def _embedding_result(
    *,
    query: str,
    query_vector: Sequence[float],
    pool: Sequence[SkillRecord],
    embeddings: Sequence[Sequence[float]],
    mistakes: Sequence[SkillRecord],
    general_top_k: int,
    task_top_k: int,
    mistake_count: int,
    task_category: str | None = None,
) -> RetrievalResult:
    scored = [
        (
            sum(left * right for left, right in zip(vector, query_vector)),
            index,
            record,
        )
        for index, (record, vector) in enumerate(zip(pool, embeddings))
    ]
    general = _top_kind(scored, "general", general_top_k)
    task_scored = (
        [item for item in scored if item[2].category == task_category]
        if task_category is not None
        else scored
    )
    task = _top_kind(task_scored, "task_specific", task_top_k)
    selected_mistakes = tuple(
        RetrievedSkill(record, None) for record in mistakes[:mistake_count]
    )
    return RetrievalResult(
        "embedding",
        query,
        general + task + selected_mistakes,
    )


def _detect_alfworld_category(goal: str) -> str:
    normalized = goal.lower()
    if "look at" in normalized and "under" in normalized:
        return "look_at_obj_in_light"
    if any(token in normalized for token in ("two ", "2 ", "two of")):
        return "pick_two_obj_and_place"
    if "clean" in normalized:
        return "pick_clean_then_place_in_recep"
    if "heat" in normalized:
        return "pick_heat_then_place_in_recep"
    if "cool" in normalized:
        return "pick_cool_then_place_in_recep"
    if "examine" in normalized or "find" in normalized:
        return "examine"
    return "pick_and_place_simple"


def _detect_legacy_alfworld_category(goal: str) -> str:
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
