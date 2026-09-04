from __future__ import annotations

import unittest

from infoskill.domain.actions import resolve_action


class ResolveActionTests(unittest.TestCase):
    def test_complete_action_tag_resolves_to_environment_command(self) -> None:
        result = resolve_action(
            "<think>Inspect the cabinet.</think>\n<action>  OPEN   CABINET 1 </action>",
            ("look", "open cabinet 1"),
        )

        self.assertEqual(result.resolved_action, "open cabinet 1")
        self.assertEqual(result.executed_action, "open cabinet 1")
        self.assertEqual(result.extraction_method, "action_tag")
        self.assertTrue(result.is_executable)

    def test_conflicting_action_tags_are_rejected_as_ambiguous(self) -> None:
        result = resolve_action(
            "<action>look</action>\n<action>open cabinet 1</action>",
            ("look", "open cabinet 1"),
        )

        self.assertIsNone(result.resolved_action)
        self.assertEqual(result.executed_action, "__invalid_action__")
        self.assertEqual(result.failure_reason, "conflicting_action_tags")

    def test_last_line_fallback_removes_only_allowed_formatting(self) -> None:
        result = resolve_action(
            "I should inspect the cabinet first.\n- Action: `OPEN   CABINET 1`",
            ("look", "open cabinet 1"),
        )

        self.assertEqual(result.resolved_action, "open cabinet 1")
        self.assertEqual(result.extraction_method, "last_line")
        self.assertTrue(result.is_executable)
        self.assertFalse(result.format_compliant)


if __name__ == "__main__":
    unittest.main()
