from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infoskill.episode import TaskSpec, Trajectory, TrajectoryGroup
from infoskill.persistence import TrainingTaskOutcomeWriter


def _group(
    task_id: str,
    task_type: str,
    wins: tuple[bool, ...],
) -> TrajectoryGroup:
    task = TaskSpec(
        task_id=task_id,
        split="train",
        task_type=task_type,
        goal=f"goal for {task_id}",
        environment_path=f"games/{task_id}.tw-pddl",
        trajectory_path=f"expert/{task_id}.json",
    )
    trajectories = tuple(
        Trajectory(
            task=task,
            rollout_id=index,
            steps=(),
            won=won,
            environment_done=won,
            horizon_exhausted=not won,
            invalid_action_count=index if not won else 0,
            reward=1.0 if won else -0.01 * index,
        )
        for index, won in enumerate(wins)
    )
    return TrajectoryGroup(task=task, trajectories=trajectories)


class TrainingTaskOutcomeWriterTests(unittest.TestCase):
    def test_writes_compact_outcomes_and_classifies_success_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            trace = run / "traces/train-update-000051-rank-000.jsonl.zst"
            trace.parent.mkdir()
            trace.touch()
            writer = TrainingTaskOutcomeWriter(
                run,
                expected_rollouts_per_task=8,
            )

            result = writer.write_training_update(
                global_update=51,
                groups=(
                    _group("hard", "type-a", (False,) * 8),
                    _group("partial", "type-a", (True,) + (False,) * 7),
                    _group("mastered", "type-b", (True,) * 8),
                ),
                trace_path=trace,
            )

            path = run / "task-outcomes/train-update-000051.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                [row["difficulty_band"] for row in rows],
                ["hard_failed", "partial", "mastered"],
            )
            self.assertEqual(rows[0]["failed_rollout_ids"], list(range(8)))
            self.assertEqual(rows[1]["success_count"], 1)
            self.assertEqual(rows[1]["failure_count"], 7)
            self.assertEqual(rows[2]["failed_rollout_ids"], [])
            self.assertEqual(rows[0]["trace"], trace.relative_to(run).as_posix())
            self.assertEqual(len(rows[0]["rollouts"]), 8)
            self.assertNotIn("steps", rows[0]["rollouts"][0])
            self.assertEqual(result["hard_failed_groups"], 1)
            self.assertEqual(result["partial_groups"], 1)
            self.assertEqual(result["mastered_groups"], 1)
            self.assertFalse(path.with_name(f".{path.name}.tmp").exists())

    def test_rewriting_an_update_is_idempotent_and_refreshes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            trace = run / "traces/train-update-000001-rank-000.jsonl.zst"
            trace.parent.mkdir()
            trace.touch()
            writer = TrainingTaskOutcomeWriter(run, expected_rollouts_per_task=8)

            writer.write_training_update(
                global_update=1,
                groups=(_group("task", "type-a", (False,) * 8),),
                trace_path=trace,
            )
            writer.write_training_update(
                global_update=1,
                groups=(_group("task", "type-a", (True,) * 8),),
                trace_path=trace,
            )

            rows = (run / "task-outcomes/train-update-000001.jsonl").read_text()
            self.assertEqual(len(rows.splitlines()), 1)
            summary = json.loads(
                (run / "task-outcomes-summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["updates"], [1])
            self.assertEqual(summary["total_task_groups"], 1)
            self.assertEqual(summary["difficulty_counts"]["mastered"], 1)
            self.assertEqual(summary["difficulty_counts"]["hard_failed"], 0)

    def test_unexpected_rollout_count_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            trace = run / "traces/train-update-000002-rank-000.jsonl.zst"
            trace.parent.mkdir()
            trace.touch()
            writer = TrainingTaskOutcomeWriter(run, expected_rollouts_per_task=8)

            result = writer.write_training_update(
                global_update=2,
                groups=(_group("short", "type-a", (False,) * 7),),
                trace_path=trace,
            )

            row = json.loads(
                (run / "task-outcomes/train-update-000002.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(row["difficulty_band"], "incomplete")
            self.assertEqual(result["incomplete_groups"], 1)

    def test_resume_quarantines_outcomes_newer_than_the_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            trace_one = run / "traces/train-update-000001-rank-000.jsonl.zst"
            trace_two = run / "traces/train-update-000002-rank-000.jsonl.zst"
            trace_one.parent.mkdir()
            trace_one.write_bytes(b"trace-one")
            trace_two.write_bytes(b"stale-trace-two")
            initial = TrainingTaskOutcomeWriter(
                run,
                expected_rollouts_per_task=8,
            )
            initial.write_training_update(
                global_update=1,
                groups=(_group("one", "type-a", (False,) * 8),),
                trace_path=trace_one,
            )
            initial.write_training_update(
                global_update=2,
                groups=(_group("stale-two", "type-a", (False,) * 8),),
                trace_path=trace_two,
            )

            resumed = TrainingTaskOutcomeWriter(
                run,
                expected_rollouts_per_task=8,
                committed_through_update=1,
            )

            stale = (
                run
                / "task-outcomes/stale-after-resume-000001"
                / "train-update-000002.jsonl"
            )
            self.assertTrue(stale.is_file())
            stale_row = json.loads(stale.read_text(encoding="utf-8").strip())
            stale_trace = run / stale_row["trace"]
            self.assertTrue(stale_trace.is_file())
            self.assertEqual(stale_trace.read_bytes(), b"stale-trace-two")
            self.assertFalse(trace_two.exists())
            self.assertFalse(
                (run / "task-outcomes/train-update-000002.jsonl").exists()
            )
            summary = json.loads(
                (run / "task-outcomes-summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["updates"], [1])

            trace_two.write_bytes(b"replacement-trace-two")
            resumed.write_training_update(
                global_update=2,
                groups=(_group("new-two", "type-a", (True,) * 8),),
                trace_path=trace_two,
            )
            self.assertEqual(stale_trace.read_bytes(), b"stale-trace-two")
            self.assertEqual(trace_two.read_bytes(), b"replacement-trace-two")
            summary = json.loads(
                (run / "task-outcomes-summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["updates"], [1, 2])
            self.assertEqual(summary["difficulty_counts"]["hard_failed"], 1)
            self.assertEqual(summary["difficulty_counts"]["mastered"], 1)

    def test_quarantine_retry_survives_failure_before_outcome_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            trace = run / "traces/train-update-000002-rank-000.jsonl.zst"
            trace.parent.mkdir()
            trace.write_bytes(b"stale-trace")
            initial = TrainingTaskOutcomeWriter(
                run,
                expected_rollouts_per_task=8,
            )
            initial.write_training_update(
                global_update=2,
                groups=(_group("stale-two", "type-a", (False,) * 8),),
                trace_path=trace,
            )

            with patch(
                "infoskill.persistence.task_outcomes._atomic_write_jsonl",
                side_effect=OSError("injected archive commit failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    TrainingTaskOutcomeWriter(
                        run,
                        expected_rollouts_per_task=8,
                        committed_through_update=1,
                    )

            active = run / "task-outcomes/train-update-000002.jsonl"
            self.assertTrue(active.is_file())
            self.assertEqual(trace.read_bytes(), b"stale-trace")

            TrainingTaskOutcomeWriter(
                run,
                expected_rollouts_per_task=8,
                committed_through_update=1,
            )

            stale_directory = (
                run / "task-outcomes/stale-after-resume-000001"
            )
            stale_outcomes = list(
                stale_directory.glob("train-update-000002.jsonl*")
            )
            self.assertEqual(len(stale_outcomes), 1)
            stale_row = json.loads(
                stale_outcomes[0].read_text(encoding="utf-8").strip()
            )
            stale_trace = run / stale_row["trace"]
            self.assertEqual(stale_trace.read_bytes(), b"stale-trace")
            self.assertEqual(
                stale_row["trace_sha256"],
                "20665ceeb144b0305d9646e5b1dba4f0"
                "a73c3a3c8279c0edc4197a9d0356f6c1",
            )
            self.assertFalse(active.exists())
            self.assertFalse(trace.exists())


if __name__ == "__main__":
    unittest.main()
