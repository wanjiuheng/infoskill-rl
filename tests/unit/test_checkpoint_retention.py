from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infoskill.persistence.checkpoint import (
    CheckpointManager,
    TrainerCheckpointState,
)
from infoskill.evaluation import EvaluationCheckpointScore
from infoskill.training import TaskScheduleState
from infoskill.training.m0 import (
    _checkpoint_is_permanent,
    _sync_best_valid_checkpoint,
)


class _Runtime:
    def save_portable_state(self, directory: Path) -> dict[str, object]:
        directory.mkdir(parents=True)
        (directory / "state.bin").write_bytes(b"portable")
        return {"portable": True}


class CheckpointRetentionTests(unittest.TestCase):
    def test_best_valid_policy_replaces_periodic_permanent_milestones(self) -> None:
        self.assertTrue(
            _checkpoint_is_permanent(
                update=25,
                max_updates=445,
                keep_best_valid=False,
            )
        )
        self.assertFalse(
            _checkpoint_is_permanent(
                update=25,
                max_updates=445,
                keep_best_valid=True,
            )
        )
        self.assertTrue(
            _checkpoint_is_permanent(
                update=0,
                max_updates=445,
                keep_best_valid=True,
            )
        )
        self.assertTrue(
            _checkpoint_is_permanent(
                update=445,
                max_updates=445,
                keep_best_valid=True,
            )
        )

    def test_keeps_recent_five_current_best_and_final_with_audit(self) -> None:
        Path(".test-tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=".test-tmp") as temporary:
            root = Path(temporary) / "checkpoints"
            manager = CheckpointManager(root, keep_recent=5)
            runtime = _Runtime()

            for step in range(5, 60, 5):
                checkpoint = manager.save(
                    state=_state(step),
                    runtime=runtime,
                    resolved_config={},
                    provenance={},
                )
                if step == 25:
                    manager.set_best_checkpoint(checkpoint)

            manager.save(
                state=_state(445),
                runtime=runtime,
                resolved_config={},
                provenance={},
                permanent=True,
            )

            self.assertEqual(
                sorted(path.name for path in root.glob("step-*")),
                [
                    "step-000025",
                    "step-000035",
                    "step-000040",
                    "step-000045",
                    "step-000050",
                    "step-000055",
                    "step-000445",
                ],
            )
            audit = [
                json.loads(line)
                for line in (root / "retention-audit.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [
                    row["checkpoint"]
                    for row in audit
                    if row["event"] == "checkpoint_deleted"
                ],
                [
                    "step-000005",
                    "step-000010",
                    "step-000015",
                    "step-000020",
                    "step-000030",
                ],
            )
            self.assertTrue(
                all(row["reason"] == "outside_recent_and_best" for row in audit)
            )
            self.assertEqual(
                [row["event"] for row in audit],
                [
                    event
                    for _ in range(5)
                    for event in ("checkpoint_delete_intent", "checkpoint_deleted")
                ],
            )

    def test_delete_intent_is_durable_before_checkpoint_removal(self) -> None:
        Path(".test-tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=".test-tmp") as temporary:
            manager = CheckpointManager(
                Path(temporary) / "checkpoints",
                keep_recent=1,
            )
            runtime = _Runtime()
            manager.save(
                state=_state(1),
                runtime=runtime,
                resolved_config={},
                provenance={},
            )
            events: list[str] = []

            with (
                patch.object(
                    manager,
                    "_append_retention_audit",
                    side_effect=lambda **row: events.append(str(row["event"])),
                ),
                patch(
                    "infoskill.persistence.checkpoint.shutil.rmtree",
                    side_effect=lambda _path: events.append("directory_removed"),
                ),
            ):
                manager.save(
                    state=_state(2),
                    runtime=runtime,
                    resolved_config={},
                    provenance={},
                )

            self.assertEqual(
                events,
                [
                    "checkpoint_delete_intent",
                    "directory_removed",
                    "checkpoint_deleted",
                ],
            )

    def test_best_checkpoint_must_be_a_committed_direct_child(self) -> None:
        Path(".test-tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=".test-tmp") as temporary:
            root = Path(temporary) / "checkpoints"
            external = Path(temporary) / "source" / "step-000050"
            external.mkdir(parents=True)
            manager = CheckpointManager(root, keep_recent=5)

            with self.assertRaisesRegex(ValueError, "direct child"):
                manager.set_best_checkpoint(external)

            self.assertTrue(external.is_dir())

    def test_selection_wiring_protects_local_best_and_ignores_source_run(self) -> None:
        Path(".test-tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=".test-tmp") as temporary:
            run = Path(temporary) / "run"
            manager = CheckpointManager(run / "checkpoints", keep_recent=2)
            runtime = _Runtime()
            best = manager.save(
                state=_state(5),
                runtime=runtime,
                resolved_config={},
                provenance={},
            )
            _sync_best_valid_checkpoint(
                checkpoint_manager=manager,
                run_directory=run,
                scores=[_score(5, "checkpoints/step-000005")],
            )
            for step in (10, 15, 20):
                manager.save(
                    state=_state(step),
                    runtime=runtime,
                    resolved_config={},
                    provenance={},
                )
            self.assertTrue(best.is_dir())

            external = (
                Path(temporary) / "source" / "checkpoints" / "step-000050"
            )
            external.mkdir(parents=True)
            _sync_best_valid_checkpoint(
                checkpoint_manager=manager,
                run_directory=run,
                scores=[_score(50, str(external))],
            )
            self.assertTrue(external.is_dir())


def _state(step: int) -> TrainerCheckpointState:
    return TrainerCheckpointState(
        global_update=step,
        schedule=TaskScheduleState(cursor=step, ordered_task_ids=("task",)),
        semantic_counters={},
    )


def _score(step: int, checkpoint: str) -> EvaluationCheckpointScore:
    return EvaluationCheckpointScore(
        step=step,
        macro_success=0.25,
        overall_success=0.25,
        invalid_action_rate=0.1,
        checkpoint=checkpoint,
    )


if __name__ == "__main__":
    unittest.main()
