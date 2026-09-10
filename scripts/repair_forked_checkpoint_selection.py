from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from infoskill.evaluation import inherit_forked_checkpoint_selection


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge valid_seen checkpoint-selection history from a source "
            "checkpoint into an already completed forked-resume run"
        )
    )
    parser.add_argument("source_checkpoint", type=Path)
    parser.add_argument("destination_run", type=Path)
    args = parser.parse_args()

    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    destination_run = args.destination_run.expanduser().resolve()
    if source_checkpoint.parent.name != "checkpoints":
        raise RuntimeError(
            "source checkpoint must be RUN/checkpoints/step-NNNNNN"
        )
    source_run = source_checkpoint.parent.parent
    trainer_state_path = source_checkpoint / "trainer_state.json"
    destination_provenance_path = destination_run / "provenance.json"
    for required in (trainer_state_path, destination_provenance_path):
        if not required.is_file():
            raise RuntimeError(f"required metadata does not exist: {required}")

    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    max_source_step = int(trainer_state["global_update"])
    provenance = json.loads(
        destination_provenance_path.read_text(encoding="utf-8")
    )
    manifest = provenance.get("valid_seen_task_manifest_sha256")
    if not isinstance(manifest, str) or not manifest:
        raise RuntimeError(
            "destination provenance has no valid_seen task manifest identity"
        )

    selection_path = destination_run / "checkpoint_selection.json"
    backup_path = destination_run / "checkpoint_selection.pre-fork-merge.json"
    if selection_path.is_file() and not backup_path.exists():
        shutil.copy2(selection_path, backup_path)

    payload = inherit_forked_checkpoint_selection(
        source_run=source_run,
        destination_run=destination_run,
        max_source_step=max_source_step,
        task_manifest_sha256=manifest,
    )
    report = {
        "source_checkpoint": str(source_checkpoint),
        "destination_run": str(destination_run),
        "backup": str(backup_path) if backup_path.is_file() else None,
        "inherited_through_step": max_source_step,
        "evaluated_steps": [
            record["step"] for record in payload["evaluations"]
        ],
        "best_valid": payload["best_valid"],
        "last": payload["last"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
