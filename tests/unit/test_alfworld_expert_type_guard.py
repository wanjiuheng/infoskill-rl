from __future__ import annotations

import unittest

from infoskill.integrations.alfworld.expert_type_guard import (
    install_alfworld_expert_type_guard,
    verify_alfworld_expert_type_binding,
)


class _LegacyAlfredExpert:
    def __init__(self, env=None, expert_type: str = "handcoded") -> None:
        self.env = env
        self.expert_type = expert_type


class AlfworldExpertTypeGuardTests(unittest.TestCase):
    def test_legacy_positional_planner_argument_is_rebound_to_expert_type(self) -> None:
        class Expert(_LegacyAlfredExpert):
            pass

        self.assertTrue(install_alfworld_expert_type_guard(Expert))

        instance = Expert("planner")

        self.assertIsNone(instance.env)
        self.assertEqual(instance.expert_type, "planner")
        self.assertEqual(instance.__infoskill_requested_expert_type__, "planner")
        self.assertTrue(instance.__infoskill_positional_binding_corrected__)

    def test_guard_is_idempotent_and_preserves_keyword_binding(self) -> None:
        class Expert(_LegacyAlfredExpert):
            pass

        self.assertTrue(install_alfworld_expert_type_guard(Expert))
        guarded = Expert.__init__
        self.assertFalse(install_alfworld_expert_type_guard(Expert))
        self.assertIs(Expert.__init__, guarded)

        instance = Expert(env="real-env", expert_type="planner")

        self.assertEqual(instance.env, "real-env")
        self.assertEqual(instance.expert_type, "planner")
        self.assertFalse(instance.__infoskill_positional_binding_corrected__)

    def test_binding_probe_fails_closed_without_the_guard(self) -> None:
        class Expert(_LegacyAlfredExpert):
            pass

        with self.assertRaisesRegex(RuntimeError, "effective expert type"):
            verify_alfworld_expert_type_binding(
                Expert,
                requested_expert_type="planner",
                module_path="alfred_tw_env.py",
            )

    def test_binding_probe_reports_actual_corrected_instance(self) -> None:
        class Expert(_LegacyAlfredExpert):
            pass

        install_alfworld_expert_type_guard(Expert)

        report = verify_alfworld_expert_type_binding(
            Expert,
            requested_expert_type="planner",
            module_path="alfred_tw_env.py",
        )

        self.assertEqual(report["requested_expert_type"], "planner")
        self.assertEqual(report["effective_expert_type"], "planner")
        self.assertTrue(report["compatibility_guard_active"])
        self.assertTrue(report["positional_binding_corrected"])


if __name__ == "__main__":
    unittest.main()
