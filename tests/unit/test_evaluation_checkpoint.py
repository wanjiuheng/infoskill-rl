from __future__ import annotations

import unittest
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


if __name__ == "__main__":
    unittest.main()
