from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from infoskill.evaluation import (
    EvaluationCheckpointScore,
    inherit_forked_checkpoint_selection,
    select_best_valid,
    write_checkpoint_selection,
    write_valid_seen_learning_curve,
)


class CheckpointSelectionTests(unittest.TestCase):
    def test_selection_records_the_evaluation_comparison_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint_selection.json"

            payload = write_checkpoint_selection(
                path,
                scores=[EvaluationCheckpointScore(0, 0.20, 0.25, 0.10)],
                task_manifest_sha256="registered-manifest",
                eval_batch_size=12,
                comparison_role="nonregistered_monitoring_curve",
            )

            self.assertEqual(payload["eval_batch_size"], 12)
            self.assertEqual(
                payload["comparison_role"],
                "nonregistered_monitoring_curve",
            )

    def test_learning_curve_is_atomically_refreshed_with_monitoring_protocol(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "valid_seen_learning_curve.svg"
            scores = [
                EvaluationCheckpointScore(
                    0,
                    0.20,
                    0.25,
                    0.10,
                    eval_batch_size=12,
                    execution_mode="eager",
                ),
                EvaluationCheckpointScore(
                    25,
                    0.30,
                    0.35,
                    0.08,
                    eval_batch_size=12,
                    execution_mode="eager",
                ),
            ]

            written = write_valid_seen_learning_curve(
                path,
                scores=scores,
                eval_batch_size=12,
                monitoring_only=True,
            )

            svg = path.read_text(encoding="utf-8")
            ElementTree.fromstring(svg)
            self.assertEqual(written, path)
            self.assertIn("<svg", svg)
            self.assertIn("Macro success", svg)
            self.assertIn("Overall success", svg)
            self.assertIn("batch=12", svg)
            self.assertIn("monitoring curve", svg)
            self.assertIn("update 25", svg)
            self.assertIn('class="point-label-macro"', svg)
            self.assertIn('class="point-label-overall"', svg)
            self.assertIn("M 20.00%", svg)
            self.assertIn("O 25.00%", svg)
            self.assertIn("M 30.00%", svg)
            self.assertIn("O 35.00%", svg)
            self.assertFalse(path.with_name(f".{path.name}.tmp").exists())

    def test_learning_curve_expands_and_marks_a_protocol_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "valid_seen_learning_curve.svg"
            scores = [
                EvaluationCheckpointScore(
                    step=step,
                    macro_success=0.20 + index / 100,
                    overall_success=0.22 + index / 100,
                    invalid_action_rate=0.10,
                    eval_batch_size=12 if step <= 50 else 64,
                    execution_mode="eager" if step <= 50 else "cuda_graph",
                )
                for index, step in enumerate(range(0, 451, 25))
            ]

            write_valid_seen_learning_curve(
                path,
                scores=scores,
                eval_batch_size=64,
                monitoring_only=True,
            )

            svg = path.read_text(encoding="utf-8")
            ElementTree.fromstring(svg)
            self.assertIn('viewBox="0 0 1810 640"', svg)
            self.assertEqual(svg.count('class="point-label-macro"'), 19)
            self.assertEqual(svg.count('class="point-label-overall"'), 19)
            self.assertIn("after update 50", svg)
            self.assertIn("batch 12 / eager → batch 64 / CUDA Graph", svg)

    def test_curve_labels_stay_separated_at_boundaries_and_nearby_steps(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "valid_seen_learning_curve.svg"
            scores = [
                EvaluationCheckpointScore(0, 0.0, 0.0, 0.1),
                EvaluationCheckpointScore(1, 1.0, 1.0, 0.1),
                EvaluationCheckpointScore(445, 0.5, 0.5, 0.1),
            ]

            write_valid_seen_learning_curve(
                path,
                scores=scores,
                eval_batch_size=64,
                monitoring_only=True,
            )

            root = ElementTree.fromstring(path.read_text(encoding="utf-8"))
            macro = [
                node
                for node in root.iter()
                if node.attrib.get("class") == "point-label-macro"
            ]
            overall = [
                node
                for node in root.iter()
                if node.attrib.get("class") == "point-label-overall"
            ]
            self.assertEqual(len(macro), 3)
            self.assertEqual(len(overall), 3)
            for macro_label, overall_label in zip(macro, overall, strict=True):
                self.assertGreaterEqual(
                    abs(
                        float(macro_label.attrib["y"])
                        - float(overall_label.attrib["y"])
                    ),
                    16.0,
                )
            x_positions = [float(label.attrib["x"]) for label in macro]
            self.assertTrue(
                all(
                    right - left >= 80
                    for left, right in zip(x_positions, x_positions[1:])
                )
            )

    def test_curve_escapes_historical_protocol_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "valid_seen_learning_curve.svg"
            scores = [
                EvaluationCheckpointScore(
                    0,
                    0.2,
                    0.25,
                    0.1,
                    eval_batch_size=12,
                    execution_mode="eager & audit",
                ),
                EvaluationCheckpointScore(
                    25,
                    0.3,
                    0.35,
                    0.1,
                    eval_batch_size=64,
                    execution_mode="cuda<graph",
                ),
            ]

            write_valid_seen_learning_curve(
                path,
                scores=scores,
                eval_batch_size=64,
                monitoring_only=True,
            )

            svg = path.read_text(encoding="utf-8")
            ElementTree.fromstring(svg)
            self.assertIn("eager &amp; audit", svg)
            self.assertIn("cuda&lt;graph", svg)

    def test_registered_ties_prefer_overall_then_invalid_then_earlier(self) -> None:
        scores = [
            EvaluationCheckpointScore(50, 0.4, 0.5, 0.1),
            EvaluationCheckpointScore(25, 0.4, 0.5, 0.1),
            EvaluationCheckpointScore(75, 0.4, 0.49, 0.01),
        ]

        self.assertEqual(select_best_valid(scores).step, 25)

    def test_forked_resume_inherits_source_scores_and_preserves_checkpoint_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            manifest = "registered-manifest"
            (source / "checkpoint_selection.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "task_manifest_sha256": manifest,
                        "eval_batch_size": 12,
                        "comparison_role": "nonregistered_monitoring_curve",
                        "evaluations": [
                            {
                                "step": 0,
                                "macro_success": 0.30,
                                "overall_success": 0.25,
                                "invalid_action_rate": 0.10,
                                "checkpoint": "checkpoints/step-000000",
                            },
                            {
                                "step": 10,
                                "macro_success": 0.90,
                                "overall_success": 0.90,
                                "invalid_action_rate": 0.01,
                                "checkpoint": "checkpoints/step-000010",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (source / "resolved_config.json").write_text(
                json.dumps(
                    {
                        "runtime_options": {
                            "hybrid_prefix_cuda_graph": False,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (destination / "checkpoint_selection.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "task_manifest_sha256": manifest,
                        "eval_batch_size": 64,
                        "comparison_role": "nonregistered_monitoring_curve",
                        "execution_mode": "cuda_graph",
                        "evaluations": [
                            {
                                "step": 25,
                                "macro_success": 0.20,
                                "overall_success": 0.24,
                                "invalid_action_rate": 0.08,
                                "checkpoint": "checkpoints/step-000025",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            inherit_forked_checkpoint_selection(
                source_run=source,
                destination_run=destination,
                max_source_step=5,
                task_manifest_sha256=manifest,
                eval_batch_size=64,
                execution_mode="cuda_graph",
            )

            payload = json.loads(
                (destination / "checkpoint_selection.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [record["step"] for record in payload["evaluations"]],
                [0, 25],
            )
            self.assertEqual(payload["best_valid"]["step"], 0)
            self.assertEqual(
                payload["evaluations"][0]["checkpoint"],
                str((source / "checkpoints/step-000000").resolve()),
            )
            self.assertEqual(
                payload["evaluations"][1]["checkpoint"],
                "checkpoints/step-000025",
            )
            self.assertEqual(payload["evaluations"][0]["eval_batch_size"], 12)
            self.assertEqual(payload["evaluations"][0]["execution_mode"], "eager")
            self.assertEqual(payload["evaluations"][1]["eval_batch_size"], 64)
            self.assertEqual(
                payload["evaluations"][1]["execution_mode"],
                "cuda_graph",
            )


if __name__ == "__main__":
    unittest.main()
