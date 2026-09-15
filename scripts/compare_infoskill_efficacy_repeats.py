#!/usr/bin/env python3
"""Aggregate repeated fixed-evaluation pairs for an INFO-SKILL candidate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

try:
    from scripts.compare_infoskill_optimization_efficacy import compare_evaluations
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from compare_infoskill_optimization_efficacy import compare_evaluations


def compare_repeated_evaluations(
    pairs: tuple[tuple[Path, Path], ...],
    *,
    minimum_macro_delta: float = 0.03,
    minimum_overall_delta: float = 0.0,
    expected_task_count: int = 140,
) -> dict[str, object]:
    if len(pairs) < 2:
        raise ValueError("at least two evaluation pairs are required")

    pair_reports = tuple(
        compare_evaluations(
            control,
            candidate,
            expected_task_count=expected_task_count,
        )
        for control, candidate in pairs
    )
    first_control, first_candidate = pairs[0]
    repeated_checkpoint_checks: list[dict[str, object]] = []
    for repeat_index, (control, candidate) in enumerate(pairs[1:], start=2):
        control_check = compare_evaluations(
            first_control,
            control,
            expected_task_count=expected_task_count,
            require_same_checkpoint=True,
        )
        candidate_check = compare_evaluations(
            first_candidate,
            candidate,
            expected_task_count=expected_task_count,
            require_same_checkpoint=True,
        )
        repeated_checkpoint_checks.append(
            {
                "repeat": repeat_index,
                "control_same_checkpoint_and_protocol": control_check[
                    "controls_valid"
                ],
                "candidate_same_checkpoint_and_protocol": candidate_check[
                    "controls_valid"
                ],
            }
        )

    controls_valid = all(
        report["controls_valid"] is True for report in pair_reports
    ) and all(
        check["control_same_checkpoint_and_protocol"] is True
        and check["candidate_same_checkpoint_and_protocol"] is True
        for check in repeated_checkpoint_checks
    )
    control_macro = statistics.fmean(
        float(report["baseline"]["macro_success"]) for report in pair_reports
    )
    candidate_macro = statistics.fmean(
        float(report["candidate"]["macro_success"]) for report in pair_reports
    )
    control_overall = statistics.fmean(
        float(report["baseline"]["overall_success"]) for report in pair_reports
    )
    candidate_overall = statistics.fmean(
        float(report["candidate"]["overall_success"]) for report in pair_reports
    )
    macro_delta = candidate_macro - control_macro
    overall_delta = candidate_overall - control_overall

    if not controls_valid:
        classification = "invalid_controls"
        exit_code = 2
    elif (
        macro_delta >= minimum_macro_delta
        and overall_delta >= minimum_overall_delta
    ):
        classification = "candidate_selected"
        exit_code = 0
    else:
        classification = "candidate_rejected_after_repeat"
        exit_code = 4

    return {
        "schema_version": 1,
        "decision_rule": (
            "mean_macro_delta_at_least_threshold_and_mean_overall_not_lower"
        ),
        "repeat_count": len(pairs),
        "controls_valid": controls_valid,
        "repeated_checkpoint_checks": repeated_checkpoint_checks,
        "minimum_macro_delta": minimum_macro_delta,
        "minimum_overall_delta": minimum_overall_delta,
        "repeats": [
            {
                "repeat": index,
                "control_run": report["baseline"]["run"],
                "candidate_run": report["candidate"]["run"],
                "control_macro_success": report["baseline"]["macro_success"],
                "candidate_macro_success": report["candidate"]["macro_success"],
                "macro_delta": report["candidate_minus_baseline"][
                    "macro_success"
                ],
                "control_overall_success": report["baseline"][
                    "overall_success"
                ],
                "candidate_overall_success": report["candidate"][
                    "overall_success"
                ],
                "overall_delta": report["candidate_minus_baseline"][
                    "overall_success"
                ],
            }
            for index, report in enumerate(pair_reports, start=1)
        ],
        "aggregate": {
            "control_macro_success": control_macro,
            "candidate_macro_success": candidate_macro,
            "macro_delta": macro_delta,
            "control_overall_success": control_overall,
            "candidate_overall_success": candidate_overall,
            "overall_delta": overall_delta,
        },
        "classification": classification,
        "exit_code": exit_code,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("CONTROL", "CANDIDATE"),
        type=Path,
        required=True,
    )
    parser.add_argument("--minimum-macro-delta", type=float, default=0.03)
    parser.add_argument("--minimum-overall-delta", type=float, default=0.0)
    parser.add_argument("--expected-task-count", type=int, default=140)
    args = parser.parse_args()
    report = compare_repeated_evaluations(
        tuple((control, candidate) for control, candidate in args.pair),
        minimum_macro_delta=args.minimum_macro_delta,
        minimum_overall_delta=args.minimum_overall_delta,
        expected_task_count=args.expected_task_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
