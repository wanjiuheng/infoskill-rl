from __future__ import annotations

import unittest
import json
from unittest.mock import patch
from pathlib import Path

from infoskill.persistence import resolve_portable_checkpoint


class EvaluationCheckpointTests(unittest.TestCase):
    def test_committed_portable_checkpoint_resolves_runtime_and_update(self) -> None:
        checkpoint = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "portable_checkpoint"
            / "step-000025"
        )

        resolved = resolve_portable_checkpoint(checkpoint)

        self.assertEqual(resolved.global_update, 25)
        self.assertEqual(resolved.runtime_directory, checkpoint / "runtime")

    def test_incomplete_checkpoint_is_rejected(self) -> None:
        missing = Path(__file__).resolve().parents[1] / "fixtures" / "missing-checkpoint"
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            resolve_portable_checkpoint(missing)

    def test_m1_checkpoint_requires_portable_infoskill_manifest(self) -> None:
        checkpoint = Path.cwd() / "synthetic-m1-checkpoint"
        completion = {
            "global_update": 1,
            "portable": True,
            "files": ["runtime/actor/actor_manifest.json"],
            "runtime_manifest": {"infoskill_modules_included": True},
        }
        actor_manifest = checkpoint / "runtime" / "actor" / "actor_manifest.json"
        with (
            patch(
                "infoskill.persistence.checkpoint._validated_checkpoint_payload",
                return_value=completion,
            ),
            patch.object(Path, "is_dir", return_value=True),
            patch.object(
                Path,
                "is_file",
                side_effect=lambda path: path == actor_manifest,
                autospec=True,
            ),
            patch.object(
                Path,
                "read_text",
                return_value=json.dumps({"global_step": 1}),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "INFO-SKILL state"):
                resolve_portable_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
