from __future__ import annotations

import unittest

from infoskill.evaluation import EvaluationCheckpointScore, select_best_valid


class CheckpointSelectionTests(unittest.TestCase):
    def test_registered_ties_prefer_overall_then_invalid_then_earlier(self) -> None:
        scores = [
            EvaluationCheckpointScore(50, 0.4, 0.5, 0.1),
            EvaluationCheckpointScore(25, 0.4, 0.5, 0.1),
            EvaluationCheckpointScore(75, 0.4, 0.49, 0.01),
        ]

        self.assertEqual(select_best_valid(scores).step, 25)


if __name__ == "__main__":
    unittest.main()
