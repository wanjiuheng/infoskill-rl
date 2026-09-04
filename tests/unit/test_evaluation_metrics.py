from __future__ import annotations

import unittest

from infoskill.config import EvaluationConfig
from infoskill.evaluation import EpisodeEvaluation, aggregate_valid_seen


class EvaluationMetricsTests(unittest.TestCase):
    def test_complete_manifest_reports_micro_and_unweighted_macro_success(self) -> None:
        config = EvaluationConfig()
        records: list[EpisodeEvaluation] = []
        for denominator in config.denominators:
            for index in range(denominator.count):
                records.append(
                    EpisodeEvaluation(
                        task_id=f"{denominator.task_type}-{index}",
                        task_type=denominator.task_type,
                        won=index == 0,
                        steps=1,
                        invalid_action_count=1 if index == 0 else 0,
                    )
                )

        result = aggregate_valid_seen(records, config=config)

        self.assertTrue(result.is_complete)
        self.assertEqual(result.evaluated, 140)
        self.assertAlmostEqual(result.overall_success, 6 / 140)
        expected_macro = sum(1 / item.count for item in config.denominators) / 6
        self.assertAlmostEqual(result.macro_success, expected_macro)
        self.assertAlmostEqual(result.invalid_action_rate, 6 / 140)
        self.assertEqual(result.mean_steps, 1.0)


if __name__ == "__main__":
    unittest.main()
