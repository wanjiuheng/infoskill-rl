from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree

from infoskill.training import m0
from infoskill.training import (
    load_training_rollout_step_scores,
    write_training_rollout_steps_curve,
)


def _write_metrics(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class TrainingRolloutStepsCurveTests(unittest.TestCase):
    def test_monitor_failure_is_isolated_from_the_training_callback(self) -> None:
        with mock.patch.object(
            m0,
            "load_training_rollout_step_scores",
            side_effect=OSError("disk unavailable"),
        ):
            logger = mock.Mock()
            refreshed = m0._refresh_training_rollout_steps_curve(
                Path("curve.svg"),
                metric_paths=(Path("metrics.jsonl"),),
                max_step=51,
                logger=logger,
            )

        self.assertFalse(refreshed)
        logger.warning.assert_called_once()
        self.assertTrue(logger.warning.call_args.kwargs["exc_info"])

    def test_curve_merges_resume_metrics_and_renders_available_series(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            current = root / "current.jsonl"
            output = root / "training_rollout_steps_curve.svg"
            _write_metrics(
                source,
                [
                    {
                        "step": 50,
                        "phase": "train",
                        "rollout/mean_steps": 24.0,
                    },
                    {
                        "step": 50,
                        "phase": "evaluation",
                        "rollout/mean_steps": 99.0,
                    },
                ],
            )
            _write_metrics(
                current,
                [
                    {
                        "step": 51,
                        "phase": "train",
                        "rollout/mean_steps": 20.0,
                        "rollout/successful_trajectories": 16.0,
                        "rollout/failed_trajectories": 48.0,
                        "rollout/mean_steps_successful": 12.0,
                        "rollout/mean_steps_failed": 22.6666666667,
                        "rollout/horizon_exhaustion_rate": 0.5,
                    },
                    {
                        "step": 51,
                        "phase": "train",
                        "rollout/mean_steps": 19.0,
                        "rollout/successful_trajectories": 20.0,
                        "rollout/failed_trajectories": 44.0,
                        "rollout/mean_steps_successful": 11.0,
                        "rollout/mean_steps_failed": 22.6363636364,
                        "rollout/horizon_exhaustion_rate": 0.4,
                    },
                ],
            )

            scores = load_training_rollout_step_scores(
                (source, current),
                max_step=51,
            )
            written = write_training_rollout_steps_curve(output, scores=scores)

            self.assertEqual([score.step for score in scores], [50, 51])
            self.assertEqual(scores[-1].mean_steps, 19.0)
            self.assertEqual(scores[-1].mean_successful_steps, 11.0)
            self.assertEqual(scores[-1].horizon_exhaustion_rate, 0.4)
            self.assertEqual(written, output)
            svg = output.read_text(encoding="utf-8")
            ElementTree.fromstring(svg)
            self.assertIn("training rollout step curve", svg)
            self.assertIn("All rollouts", svg)
            self.assertIn("Successful rollouts", svg)
            self.assertIn("Failed rollouts", svg)
            self.assertIn("Horizon exhausted", svg)
            self.assertIn("Latest update 51", svg)
            self.assertFalse(output.with_name(f".{output.name}.tmp").exists())

    def test_curve_does_not_invent_means_for_empty_outcome_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "metrics.jsonl"
            output = root / "training_rollout_steps_curve.svg"
            _write_metrics(
                metrics,
                [
                    {
                        "step": 1,
                        "phase": "train",
                        "rollout/mean_steps": 30.0,
                        "rollout/successful_trajectories": 0.0,
                        "rollout/failed_trajectories": 64.0,
                        "rollout/mean_steps_successful": 0.0,
                        "rollout/mean_steps_failed": 30.0,
                        "rollout/horizon_exhaustion_rate": 1.0,
                    }
                ],
            )

            scores = load_training_rollout_step_scores((metrics,))
            write_training_rollout_steps_curve(output, scores=scores)

            self.assertIsNone(scores[0].mean_successful_steps)
            self.assertEqual(scores[0].mean_failed_steps, 30.0)
            svg = output.read_text(encoding="utf-8")
            self.assertIn('class="dot-all"', svg)
            self.assertIn('class="dot-failure"', svg)
            self.assertIn('class="dot-horizon"', svg)
            self.assertNotIn("Latest successful mean", svg)
            self.assertIn("Latest failed mean 30.00", svg)


if __name__ == "__main__":
    unittest.main()
