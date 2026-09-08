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

    def test_square_action_marker_resolves_without_counting_as_format_compliant(self) -> None:
        result = resolve_action(
            "<think>Go to the desk.</think>\n[action] GO TO DESK 1 [/action]",
            ("look", "go to desk 1"),
        )

        self.assertEqual(result.resolved_action, "go to desk 1")
        self.assertEqual(result.extraction_method, "action_tag")
        self.assertTrue(result.is_executable)
        self.assertTrue(result.had_action_tag)
        self.assertFalse(result.format_compliant)

    def test_malformed_square_action_opener_resolves_only_on_final_line(self) -> None:
        result = resolve_action(
            "<think>Take the bowl.</think>\n[action> take bowl 1 from desk 1 </action>",
            ("look", "take bowl 1 from desk 1"),
        )

        self.assertEqual(result.resolved_action, "take bowl 1 from desk 1")
        self.assertTrue(result.is_executable)
        self.assertFalse(result.format_compliant)

    def test_unclosed_square_action_marker_resolves_on_final_line(self) -> None:
        result = resolve_action(
            "<think>Examine the desk.</think>\n[action] examine desk 1",
            ("look", "examine desk 1"),
        )

        self.assertEqual(result.resolved_action, "examine desk 1")
        self.assertTrue(result.is_executable)
        self.assertFalse(result.format_compliant)

    def test_square_action_text_inside_reasoning_is_not_scanned(self) -> None:
        result = resolve_action(
            "<think>\n[action] look\nI should reconsider.\n</think>\nNo final action.",
            ("look", "open cabinet 1"),
        )

        self.assertIsNone(result.resolved_action)
        self.assertEqual(result.executed_action, "__invalid_action__")
        self.assertFalse(result.is_executable)


if __name__ == "__main__":
    unittest.main()
