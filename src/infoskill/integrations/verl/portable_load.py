from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def load_portable_state_after_base_sync(
    *,
    worker_group: object,
    actor_directory: Path,
) -> tuple[Mapping[str, object], ...]:
    """Make dummy-loaded vLLM base-ready before restoring a portable LoRA."""

    reports = tuple(
        worker_group.prepare_infoskill_portable_checkpoint_load()  # type: ignore[attr-defined]
    )
    if not reports:
        raise RuntimeError("portable checkpoint base preparation returned no workers")
    if any(report.get("base_sync_done_after") is not True for report in reports):
        raise RuntimeError("vLLM base synchronization did not complete on every rank")
    worker_group.load_portable_checkpoint(  # type: ignore[attr-defined]
        str(actor_directory)
    )
    return reports
