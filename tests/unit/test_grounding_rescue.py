from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from infoskill.episode import TaskSpec
from infoskill.integrations.alfworld import (
    ExpertReplayResult,
    GroundingWorkItem,
    build_grounding_manifest,
    load_available_committed_grounding_results,
    load_committed_grounding_results,
    merge_timeout_grounding_results,
    run_bounded_grounding,
    select_timeout_work_items,
)


def _item(index: int) -> GroundingWorkItem:
    return GroundingWorkItem(
        task=TaskSpec(
            task_id=f"task-{index}",
            split="train",
            task_type="pick_two_obj_and_place",
            goal="put two objects somewhere",
        ),
        candidate_skill_ids=("skill-1",),
        seed=index,
    )


def _result(index: int, *, succeeded: bool) -> ExpertReplayResult:
    return ExpertReplayResult(
        task_id=f"task-{index}",
        succeeded=succeeded,
        samples=(),
        total_steps=5 if succeeded else 0,
        quarantine_reason=None if succeeded else "expert_wall_timeout",
    )


class GroundingRescueTests(unittest.TestCase):
    def test_eight_rescues_cross_the_unchanged_formal_coverage_gate(self) -> None:
        source = [
            (
                "pick_two_obj_and_place",
                _result(index, succeeded=index < 3510),
            )
            for index in range(3553)
        ]
        rescue = [
            ("pick_two_obj_and_place", _result(index, succeeded=True))
            for index in range(3510, 3518)
        ]

        merged = merge_timeout_grounding_results(source, rescue)
        manifest = build_grounding_manifest(
            results=merged.results,
            source_checksums={"test": "abc"},
            code_revision="test",
            max_replay_steps=150,
            persist_horizon=30,
            expert_type="planner",
            expert_binding={
                "requested_expert_type": "planner",
                "effective_expert_type": "planner",
                "compatibility_guard_active": True,
                "positional_binding_corrected": True,
                "module_within_configured_source": True,
            },
        )

        self.assertEqual(manifest.successful_games, 3518)
        self.assertGreaterEqual(manifest.success_coverage, 0.99)
        self.assertTrue(manifest.formal_gate_passed)

    def test_loads_only_checksum_validated_committed_shards(self) -> None:
        work_items = tuple(_item(index) for index in range(3))

        def worker(
            items,
            config_path,
            temporary,
            max_steps,
            horizon,
            expert_type,
            replay_backend,
            native_batch_size,
            callback,
        ):
            del (
                config_path,
                temporary,
                max_steps,
                horizon,
                expert_type,
                replay_backend,
                native_batch_size,
            )
            if callback is not None:
                callback(len(items))
            return [
                (item.task.task_type, _result(int(item.task.task_id[-1]), succeeded=True))
                for item in items
            ]

        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.yaml"
            config.write_text("test: true\n", encoding="utf-8")
            expected, _ = run_bounded_grounding(
                work_items=work_items,
                config_path=config,
                run_directory=temporary,
                worker_batch_size=2,
                max_replay_steps=150,
                persist_horizon=30,
                expert_type="planner",
                worker_runner=worker,
            )
            loaded = load_committed_grounding_results(temporary, work_items)
            plan_path = Path(temporary) / "grounding-resume.json"
            original_plan = plan_path.read_text(encoding="utf-8")
            plan_path.write_text(
                original_plan.replace('"task_count": 3', '"task_count": 4'),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "task_count|plan_sha256"):
                load_committed_grounding_results(temporary, work_items)
            plan_path.write_text(original_plan, encoding="utf-8")
            results_path = (
                Path(temporary)
                / "grounding-shards"
                / "shard-0002"
                / "results.jsonl"
            )
            results_path.write_text("corrupt\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "results_sha256"):
                load_committed_grounding_results(temporary, work_items)

        self.assertEqual(loaded, expected)

    def test_loads_a_checksum_validated_snapshot_of_committed_shards(self) -> None:
        work_items = tuple(_item(index) for index in range(3))

        def worker(
            items,
            config_path,
            temporary,
            max_steps,
            horizon,
            expert_type,
            replay_backend,
            native_batch_size,
            callback,
        ):
            del (
                config_path,
                temporary,
                max_steps,
                horizon,
                expert_type,
                replay_backend,
                native_batch_size,
            )
            if callback is not None:
                callback(len(items))
            return [
                (
                    item.task.task_type,
                    _result(int(item.task.task_id[-1]), succeeded=True),
                )
                for item in items
            ]

        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.yaml"
            config.write_text("test: true\n", encoding="utf-8")
            run_bounded_grounding(
                work_items=work_items,
                config_path=config,
                run_directory=temporary,
                worker_batch_size=1,
                max_replay_steps=150,
                persist_horizon=30,
                expert_type="planner",
                worker_runner=worker,
            )
            marker = (
                Path(temporary)
                / "grounding-shards"
                / "shard-0002"
                / "complete.json"
            )
            marker.unlink()

            snapshot = load_available_committed_grounding_results(
                temporary,
                work_items,
            )

        self.assertEqual(
            [result.task_id for _, result in snapshot],
            ["task-0", "task-2"],
        )

    def test_selects_only_timeout_work_items_in_original_order(self) -> None:
        work_items = tuple(_item(index) for index in range(4))
        source = [
            (item.task.task_type, _result(index, succeeded=index in {0, 2}))
            for index, item in enumerate(work_items)
        ]

        selected = select_timeout_work_items(work_items, source)

        self.assertEqual(
            [item.task.task_id for item in selected],
            ["task-1", "task-3"],
        )

        changed_type = list(source)
        changed_type[1] = ("pick_and_place_simple", changed_type[1][1])
        with self.assertRaisesRegex(ValueError, "changed task type"):
            select_timeout_work_items(work_items, changed_type)

    def test_merge_replaces_only_successful_timeout_rescues(self) -> None:
        source = [
            ("pick_two_obj_and_place", _result(0, succeeded=True)),
            ("pick_two_obj_and_place", _result(1, succeeded=False)),
            ("pick_two_obj_and_place", _result(2, succeeded=False)),
        ]
        rescue = [
            ("pick_two_obj_and_place", _result(1, succeeded=True)),
            ("pick_two_obj_and_place", _result(2, succeeded=False)),
        ]

        merged = merge_timeout_grounding_results(source, rescue)

        self.assertEqual(
            [result.task_id for _, result in merged.results],
            ["task-0", "task-1", "task-2"],
        )
        self.assertTrue(merged.results[1][1].succeeded)
        self.assertEqual(
            merged.results[2][1].quarantine_reason,
            "expert_wall_timeout",
        )
        self.assertEqual(merged.attempted_task_ids, ("task-1", "task-2"))
        self.assertEqual(merged.rescued_task_ids, ("task-1",))
        self.assertEqual(merged.remaining_timeout_task_ids, ("task-2",))

    def test_merge_rejects_non_timeout_or_unknown_rescue_targets(self) -> None:
        non_timeout = ExpertReplayResult(
            "task-1",
            False,
            (),
            3,
            "terminated_without_win",
        )
        source = [
            ("pick_two_obj_and_place", _result(0, succeeded=True)),
            ("pick_two_obj_and_place", non_timeout),
        ]

        with self.assertRaisesRegex(ValueError, "not an expert timeout"):
            merge_timeout_grounding_results(
                source,
                [("pick_two_obj_and_place", _result(1, succeeded=True))],
            )
        with self.assertRaisesRegex(ValueError, "unknown source task"):
            merge_timeout_grounding_results(
                source,
                [("pick_two_obj_and_place", _result(9, succeeded=True))],
            )


if __name__ == "__main__":
    unittest.main()
