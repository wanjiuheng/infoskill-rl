from __future__ import annotations

import unittest

from infoskill.integrations.verl.worker_options import policy_gradient_clip_mode


class VerlWorkerOptionsTests(unittest.TestCase):
    def test_policy_gradient_clip_mode_crosses_the_worker_model_config_seam(self) -> None:
        self.assertEqual(policy_gradient_clip_mode({}), "joint")
        self.assertEqual(
            policy_gradient_clip_mode(
                {"infoskill_policy_gradient_clip_mode": "separate"}
            ),
            "separate",
        )

    def test_unknown_policy_gradient_clip_mode_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "gradient clip mode"):
            policy_gradient_clip_mode(
                {"infoskill_policy_gradient_clip_mode": "shared"}
            )


if __name__ == "__main__":
    unittest.main()
