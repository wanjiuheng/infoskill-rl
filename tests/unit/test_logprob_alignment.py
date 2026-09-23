from __future__ import annotations

import math
import unittest

from infoskill.learning import (
    LogprobAlignmentError,
    LogprobAlignmentThresholds,
    alignment_passes,
    collect_logprob_alignment_offenders,
    require_logprob_alignment,
    summarize_logprob_alignment,
)


class LogprobAlignmentTests(unittest.TestCase):
    def test_masked_rollout_recompute_alignment_reports_ratio_deviation(self) -> None:
        summary = summarize_logprob_alignment(
            rollout=((-1.0, -2.0, 99.0), (-0.5, 99.0, 99.0)),
            recomputed=((-1.0, -1.9, -50.0), (-0.6, -50.0, -50.0)),
            mask=((True, True, False), (True, False, False)),
        )

        self.assertEqual(summary["token_count"], 3)
        self.assertAlmostEqual(summary["logprob_abs_error_max"], 0.1)
        self.assertAlmostEqual(summary["ratio_max_abs_deviation"], math.exp(0.1) - 1.0)
        self.assertAlmostEqual(
            summary["ratio_mean"],
            (1.0 + math.exp(0.1) + math.exp(-0.1)) / 3.0,
        )

    def test_non_finite_alignment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            summarize_logprob_alignment(
                rollout=((float("nan"),),),
                recomputed=((-1.0,),),
                mask=((True,),),
            )

    def test_alignment_reports_tail_and_worst_token_diagnostics(self) -> None:
        summary = summarize_logprob_alignment(
            rollout=((-1.0, -6.0, -2.0, 99.0),),
            recomputed=((-1.0, -5.0, -5.0, -50.0),),
            mask=((True, True, True, False),),
            token_ids=((11, 22, 33, 0),),
        )

        self.assertEqual(summary["token_count"], 3)
        self.assertAlmostEqual(summary["logprob_abs_error_median"], 1.0)
        self.assertAlmostEqual(summary["logprob_abs_error_p95"], 2.8)
        self.assertAlmostEqual(summary["logprob_abs_error_p99"], 2.96)
        self.assertAlmostEqual(summary["logprob_abs_error_gt_1_rate"], 1 / 3)
        self.assertEqual(summary["max_sample_index"], 0)
        self.assertEqual(summary["max_token_position"], 2)
        self.assertEqual(summary["max_token_id"], 33)
        self.assertEqual(summary["max_at_first_active_token"], 0)
        self.assertEqual(summary["max_at_last_active_token"], 1)
        self.assertAlmostEqual(summary["max_signed_logprob_delta"], -3.0)
        self.assertAlmostEqual(summary["max_rollout_logprob"], -2.0)
        self.assertAlmostEqual(summary["max_recomputed_logprob"], -5.0)
        self.assertEqual(summary["rollout_ge_neg1_count"], 1)
        self.assertEqual(summary["rollout_neg5_to_neg1_count"], 1)
        self.assertEqual(summary["rollout_lt_neg5_count"], 1)
        self.assertAlmostEqual(summary["rollout_lt_neg5_error_mean"], 1.0)

    def test_observed_post_fix_alignment_passes_the_default_gate(self) -> None:
        require_logprob_alignment(
            {
                "logprob_abs_error_mean": 0.021211,
                "logprob_abs_error_median": 0.001749,
                "logprob_abs_error_p95": 0.102767,
                "logprob_abs_error_p99": 0.182578,
                "logprob_abs_error_gt_1_rate": 0.0,
                "logprob_abs_error_gt_5_rate": 0.0,
                "ratio_mean": 1.000495,
            }
        )

    def test_observed_raw_full_alignment_passes_the_calibrated_gate(self) -> None:
        require_logprob_alignment(
            {
                "logprob_abs_error_mean": 0.030098398605059914,
                "logprob_abs_error_median": 0.002674724906682968,
                "logprob_abs_error_p95": 0.14057102799415588,
                "logprob_abs_error_p99": 0.2843205928802492,
                "logprob_abs_error_gt_1_rate": 0.00021199701185164248,
                "logprob_abs_error_gt_5_rate": 0.0,
                "ratio_mean": 0.9997580686460814,
            }
        )

    def test_prefixed_padding_failure_is_blocked_before_update(self) -> None:
        with self.assertRaisesRegex(
            LogprobAlignmentError,
            "alignment gate failed before policy update",
        ) as raised:
            require_logprob_alignment(
                {
                    "logprob_abs_error_mean": 0.377112,
                    "logprob_abs_error_median": 0.001976,
                    "logprob_abs_error_p95": 0.115651,
                    "logprob_abs_error_p99": 22.203129,
                    "logprob_abs_error_gt_1_rate": 0.015819,
                    "logprob_abs_error_gt_5_rate": 0.015819,
                    "ratio_mean": 0.984668,
                }
            )

        message = str(raised.exception)
        self.assertIn("logprob_abs_error_mean", message)
        self.assertIn("logprob_abs_error_p99", message)
        self.assertIn("logprob_abs_error_gt_5_rate", message)
        diagnostic = raised.exception.as_dict()
        self.assertFalse(diagnostic["passed"])
        self.assertEqual(
            diagnostic["summary"]["logprob_abs_error_p99"],
            22.203129,
        )
        self.assertEqual(diagnostic["thresholds"]["error_p99_max"], 0.30)
        self.assertEqual(len(diagnostic["failures"]), 4)

    def test_missing_gate_metric_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required metric"):
            require_logprob_alignment({"ratio_mean": 1.0})

    def test_offenders_are_sorted_and_bound_to_row_metadata(self) -> None:
        offenders = collect_logprob_alignment_offenders(
            rollout=((-0.5, -0.5), (-0.5, -0.5)),
            recomputed=((-0.5, -3.0), (-7.0, -0.5)),
            mask=((True, True), (True, True)),
            token_ids=((10, 11), (12, 13)),
            row_metadata=({"task_id": "a"}, {"task_id": "b"}),
            decode_token=lambda token_id: f"token-{token_id}",
            minimum_abs_error=1.0,
        )

        self.assertEqual([item["sample_index"] for item in offenders], [1, 0])
        self.assertEqual(offenders[0]["token_text"], "token-12")
        self.assertEqual(offenders[0]["row"], {"task_id": "b"})
        self.assertTrue(offenders[0]["at_first_active_token"])
        self.assertFalse(offenders[0]["at_last_active_token"])

    def test_alignment_passes_reports_gate_result_without_hiding_bad_schema(self) -> None:
        passing = {
            "logprob_abs_error_mean": 0.0,
            "logprob_abs_error_median": 0.0,
            "logprob_abs_error_p95": 0.0,
            "logprob_abs_error_p99": 0.0,
            "logprob_abs_error_gt_1_rate": 0.0,
            "logprob_abs_error_gt_5_rate": 0.0,
            "ratio_mean": 1.0,
        }
        self.assertTrue(alignment_passes(passing))
        self.assertFalse(
            alignment_passes({**passing, "logprob_abs_error_gt_5_rate": 0.1})
        )
        with self.assertRaisesRegex(ValueError, "missing required metric"):
            alignment_passes({"ratio_mean": 1.0})

    def test_alignment_error_persists_counterfactual_diagnostics(self) -> None:
        error = LogprobAlignmentError(
            summary={"token_count": 2},
            thresholds=LogprobAlignmentThresholds(),
            failures=("failed",),
            diagnostics={"counterfactual_classification": "prefix"},
        )

        self.assertEqual(
            error.as_dict()["diagnostics"]["counterfactual_classification"],
            "prefix",
        )


if __name__ == "__main__":
    unittest.main()
