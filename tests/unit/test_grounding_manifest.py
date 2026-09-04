from __future__ import annotations

import unittest

from infoskill.integrations.alfworld import ExpertReplayResult, build_grounding_manifest


class GroundingManifestTests(unittest.TestCase):
    def test_formal_gate_reports_low_coverage_and_long_replays(self) -> None:
        results = [
            (
                "pick_and_place_simple",
                ExpertReplayResult("ok", True, (), 31, None),
            ),
            (
                "pick_and_place_simple",
                ExpertReplayResult("bad", False, (), 1, "expert_action_not_admissible"),
            ),
        ]

        manifest = build_grounding_manifest(
            results=results,
            source_checksums={"data": "abc"},
            code_revision="test",
            max_replay_steps=150,
            persist_horizon=30,
        )

        self.assertFalse(manifest.formal_gate_passed)
        self.assertEqual(manifest.success_coverage, 0.5)
        self.assertEqual(manifest.over_persist_horizon_rate, 0.5)
        self.assertEqual(manifest.quarantine_reasons["expert_action_not_admissible"], 1)


if __name__ == "__main__":
    unittest.main()
