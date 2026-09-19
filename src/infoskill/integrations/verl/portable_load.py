from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path


def load_portable_state_after_base_sync(
    *,
    worker_group: object,
    actor_directory: Path,
    actor_learning_rate_override: float | None = None,
) -> tuple[Mapping[str, object], ...]:
    """Make dummy-loaded vLLM base-ready before restoring a portable LoRA."""

    reports = tuple(
        worker_group.prepare_infoskill_portable_checkpoint_load()  # type: ignore[attr-defined]
    )
    if not reports:
        raise RuntimeError("portable checkpoint base preparation returned no workers")
    if any(report.get("base_sync_done_after") is not True for report in reports):
        raise RuntimeError("vLLM base synchronization did not complete on every rank")
    load_reports = tuple(
        worker_group.load_portable_checkpoint(  # type: ignore[attr-defined]
            str(actor_directory),
            actor_learning_rate_override,
        )
    )
    if not load_reports:
        raise RuntimeError("portable checkpoint load returned no workers")
    prepared_by_rank = _reports_by_rank(reports, stage="base preparation")
    loaded_by_rank = _reports_by_rank(load_reports, stage="checkpoint load")
    if set(prepared_by_rank) != set(loaded_by_rank):
        raise RuntimeError("portable checkpoint preparation/load rank set differs")
    if actor_learning_rate_override is not None:
        expected = float(actor_learning_rate_override)
        if any(
            not math.isclose(
                float(report.get("actor_learning_rate", float("nan"))),
                expected,
                rel_tol=1e-9,
            )
            for report in load_reports
        ):
            raise RuntimeError(
                "portable checkpoint actor learning-rate override did not apply "
                "on every rank"
            )
    return tuple(
        {**prepared_by_rank[rank], **loaded_by_rank[rank]}
        for rank in sorted(prepared_by_rank)
    )


def _reports_by_rank(
    reports: tuple[Mapping[str, object], ...],
    *,
    stage: str,
) -> dict[int, Mapping[str, object]]:
    by_rank: dict[int, Mapping[str, object]] = {}
    for report in reports:
        rank = report.get("rank")
        if not isinstance(rank, int) or rank < 0 or rank in by_rank:
            raise RuntimeError(f"portable checkpoint {stage} returned invalid ranks")
        by_rank[rank] = report
    return by_rank
