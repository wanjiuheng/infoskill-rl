from __future__ import annotations

import unittest

from infoskill.integrations.alfworld.planner_loop_diagnostic import (
    TracingPlannerExpert,
    build_planner_loop_report,
    select_loop_diagnostic_rows,
    summarize_planner_trace,
)


class PlannerLoopDiagnosticTests(unittest.TestCase):
    def test_trace_fingerprint_prefers_world_facts_over_visible_text(self) -> None:
        expert = TracingPlannerExpert()
        shared = {
            "feedback": "Nothing happens.",
            "admissible_commands": ("look", "inventory"),
            "extra.expert_plan": ("look",),
            "won": False,
        }

        expert.act(dict(shared, facts=("at apple counter",)), 0.0, False, "")
        expert.act(dict(shared, facts=("at apple table",)), 0.0, False, "look")

        self.assertNotEqual(
            expert.trace[0]["state_fingerprint"],
            expert.trace[1]["state_fingerprint"],
        )
        self.assertEqual(expert.trace[0]["state_fingerprint_source"], "facts")

    def test_selection_keeps_all_failures_and_longest_two_object_controls(self) -> None:
        rows = (
            _row("failed-cool", "pick_cool_then_place_in_recep", False, 150),
            _row("failed-two", "pick_two_obj_and_place", False, 150),
            _row("short-two", "pick_two_obj_and_place", True, 20),
            _row("long-two-a", "pick_two_obj_and_place", True, 80),
            _row("long-two-b", "pick_two_obj_and_place", True, 120),
            _row("long-simple", "pick_and_place_simple", True, 90),
        )

        selected = select_loop_diagnostic_rows(
            rows=rows,
            successful_two_object_controls=1,
            persist_horizon=30,
        )

        self.assertEqual(
            [row["task_id"] for row in selected],
            ["failed-cool", "failed-two", "long-two-b"],
        )
        self.assertEqual(selected[-1]["diagnostic_role"], "long_success_control")

    def test_trace_summary_separates_historical_revisit_from_terminal_cycle(self) -> None:
        trace = (
            _step(1, "state-a", "go to desk 1"),
            _step(2, "state-b", "go to shelf 1"),
            _step(3, "state-a", "go to desk 1"),
            _step(5, "state-a", "go to desk 1"),
            _step(6, "state-c", "take apple 1"),
            _step(7, "state-d", "go to fridge 1"),
            _step(8, "state-e", "open fridge 1"),
        )

        summary = summarize_planner_trace(trace)

        self.assertTrue(summary["repeated_state_action_detected"])
        self.assertFalse(summary["terminal_cycle_detected"])
        self.assertEqual(summary["max_state_action_occurrences"], 3)
        self.assertEqual(summary["first_cycle_step"], 5)
        self.assertEqual(summary["unique_state_fingerprints"], 5)

    def test_trace_summary_detects_exact_periodic_terminal_suffix(self) -> None:
        trace = (
            _step(1, "progress", "take apple 1"),
            _step(2, "state-a", "go to desk 1"),
            _step(3, "state-b", "go to shelf 1"),
            _step(4, "state-a", "go to desk 1"),
            _step(5, "state-b", "go to shelf 1"),
            _step(6, "state-a", "go to desk 1"),
            _step(7, "state-b", "go to shelf 1"),
        )

        summary = summarize_planner_trace(trace)

        self.assertTrue(summary["terminal_cycle_detected"])
        self.assertEqual(summary["terminal_cycle_period"], 2)
        self.assertEqual(summary["terminal_cycle_repetitions"], 3)

    def test_trace_summary_does_not_call_one_revisit_a_cycle(self) -> None:
        trace = (
            _step(1, "state-a", "open drawer 1"),
            _step(2, "state-b", "take pen 1 from drawer 1"),
            _step(3, "state-a", "close drawer 1"),
        )

        summary = summarize_planner_trace(trace)

        self.assertFalse(summary["repeated_state_action_detected"])
        self.assertFalse(summary["terminal_cycle_detected"])
        self.assertEqual(summary["max_state_action_occurrences"], 1)

    def test_report_names_rescues_failures_cycles_and_control_regressions(self) -> None:
        rows = (
            _diagnostic_row("rescued", "source_failure", True, False),
            _diagnostic_row("looping", "source_failure", False, True),
            _diagnostic_row("control", "long_success_control", False, False),
        )

        report = build_planner_loop_report(
            rows=rows,
            source_pilot_run="pilot",
            source_pilot_sha256="pilot-sha",
            source_results_sha256="rows-sha",
            train_task_manifest_sha256="tasks-sha",
            code_revision="revision-1234567890",
            source_max_replay_steps=150,
            diagnostic_max_replay_steps=300,
            successful_two_object_controls=1,
            temporary_directory_cleaned=True,
            minimum_free_disk_bytes=123,
        )

        self.assertEqual(report["rescued_task_ids"], ["rescued"])
        self.assertEqual(report["still_failed_task_ids"], ["looping"])
        self.assertEqual(report["terminal_cycle_task_ids"], ["looping"])
        self.assertEqual(report["control_regression_task_ids"], ["control"])


def _row(
    task_id: str,
    task_type: str,
    succeeded: bool,
    total_steps: int,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "task_type": task_type,
        "goal": "goal",
        "seed": 7,
        "succeeded": succeeded,
        "total_steps": total_steps,
        "reason": None if succeeded else "terminated_without_win",
    }


def _step(step: int, fingerprint: str, action: str) -> dict[str, object]:
    return {
        "step_index": step,
        "state_fingerprint": fingerprint,
        "action": action,
    }


def _diagnostic_row(
    task_id: str,
    role: str,
    succeeded: bool,
    cycle: bool,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "task_type": "pick_two_obj_and_place",
        "diagnostic_role": role,
        "diagnostic_result": {
            "succeeded": succeeded,
            "total_steps": 50 if succeeded else 300,
            "reason": None if succeeded else "expert_replay_limit",
        },
        "trace_summary": {
            "repeated_state_action_detected": cycle,
            "terminal_cycle_detected": cycle,
        },
    }


if __name__ == "__main__":
    unittest.main()
