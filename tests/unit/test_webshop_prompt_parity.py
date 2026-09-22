from __future__ import annotations

import unittest

from infoskill.integrations.webshop.policy import render_webshop_policy_message
from infoskill.integrations.webshop.prompt_parity import compare_initial_step


class WebShopPromptParityTests(unittest.TestCase):
    def test_rich_online_search_page_matches_imitation_first_step(self) -> None:
        goal = "find red walking shoes"
        observation = "[button] Search [button_]"
        offline_prompt = render_webshop_policy_message(
            task_description=goal,
            current_observation=observation,
            available_actions=("search[<your query>]",),
        )
        result = compare_initial_step(
            offline_prompt=offline_prompt,
            offline_goal=goal,
            offline_observation=observation,
            offline_actions=("search[<your query>]",),
            online_goal=goal + ", and price lower than 50.00 dollars",
            online_observation=(
                "WebShop\nInstruction:\n"
                + goal
                + ", and price lower than 50.00 dollars\n[button] Search [button_]"
            ),
            online_available_actions={
                "has_search_bar": True,
                "clickables": ["search"],
            },
        )
        self.assertTrue(result["passed"])
        self.assertFalse(result["full_task_text_match"])
        self.assertEqual(result["online_observation"], observation)
        self.assertEqual(result["online_actions"], ["search[<your query>]"])

    def test_wrong_online_action_fails_closed(self) -> None:
        goal = "find red walking shoes"
        observation = "[button] Search [button_]"
        result = compare_initial_step(
            offline_prompt=render_webshop_policy_message(
                task_description=goal,
                current_observation=observation,
                available_actions=("search[<your query>]",),
            ),
            offline_goal=goal,
            offline_observation=observation,
            offline_actions=("search[<your query>]",),
            online_goal=goal,
            online_observation=(
                "WebShop\nInstruction:\n" + goal + "\n[button] Search [button_]"
            ),
            online_available_actions={
                "has_search_bar": False,
                "clickables": ["buy now"],
            },
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["actions_match"])

    def test_wrong_online_goal_fails_closed(self) -> None:
        goal = "find red walking shoes"
        observation = "[button] Search [button_]"
        result = compare_initial_step(
            offline_prompt=render_webshop_policy_message(
                task_description=goal,
                current_observation=observation,
                available_actions=("search[<your query>]",),
            ),
            offline_goal=goal,
            offline_observation=observation,
            offline_actions=("search[<your query>]",),
            online_goal="find blue travel bag",
            online_observation=(
                "WebShop\nInstruction:\nfind blue travel bag\n"
                "[button] Search [button_]"
            ),
            online_available_actions={"has_search_bar": True, "clickables": ["search"]},
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["goal_match"])


if __name__ == "__main__":
    unittest.main()
