from __future__ import annotations

import math
import unittest

from infoskill.learning import summarize_logprob_alignment


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


if __name__ == "__main__":
    unittest.main()
