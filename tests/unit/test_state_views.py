from __future__ import annotations

import unittest

from infoskill.domain.state import AgentHistoryEntry, CanonicalAgentState, render_state_views


class StateViewsTests(unittest.TestCase):
    def test_views_share_one_state_but_expose_only_registered_fields(self) -> None:
        state = CanonicalAgentState(
            task_id="game-1",
            split="train",
            task_type="pick_and_place_simple",
            goal="put the apple in the fridge",
            step_index=3,
            observation="The fridge is closed.",
            history=(
                AgentHistoryEntry(0, "You are in the kitchen.", "look"),
                AgentHistoryEntry(1, "A fridge is nearby.", "go to fridge 1"),
                AgentHistoryEntry(2, "The fridge is closed.", "take apple 1 from table 1"),
            ),
            admissible_commands=("look", "open fridge 1"),
        )

        views = render_state_views(state, history_limit=2)

        self.assertEqual(views.retrieval_view, "put the apple in the fridge")
        self.assertIn("go to fridge 1", views.compression_view)
        self.assertIn("take apple 1 from table 1", views.compression_view)
        self.assertNotIn("You are in the kitchen.", views.compression_view)
        self.assertNotIn("open fridge 1", views.compression_view)
        self.assertIn("'look'", views.policy_view)
        self.assertIn("'open fridge 1'", views.policy_view)
        self.assertIn("<think> </think>", views.policy_view)
        self.assertIn("<action> </action>", views.policy_view)


if __name__ == "__main__":
    unittest.main()
