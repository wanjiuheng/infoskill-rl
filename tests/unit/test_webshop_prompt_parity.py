from __future__ import annotations

import unittest

from infoskill.integrations.webshop.policy import render_webshop_policy_message
from infoskill.integrations.webshop.prompt_parity import (
    compare_initial_step,
    compare_replayed_step,
    replay_demonstration,
)


def _two_step_fixture() -> tuple[
    str, list[dict[str, object]], dict[str, object]
]:
    goal = "find red walking shoes"
    observations = ("[button] Search [button_]", "[button] Buy Now [button_]")
    actions = ("search[red walking shoes]", "click[Buy Now]")
    admissible = (("search[<your query>]",), ("click[Buy Now]",))
    history: list[tuple[str, str]] = []
    rows: list[dict[str, object]] = []
    for step_index, (observation, action, available) in enumerate(
        zip(observations, actions, admissible, strict=True)
    ):
        rows.append(
            {
                "step_index": step_index,
                "prompt": render_webshop_policy_message(
                    task_description=goal,
                    current_observation=observation,
                    available_actions=available,
                    history=history,
                ),
                "response": f"<think>x</think>\n<action>{action}</action>",
            }
        )
        history.append((observation, action))
    demonstration = {
        "states": [
            f"WebShop\nInstruction:\n{goal}\n{observations[0]}",
            observations[1],
        ],
        "available_actions": [list(items) for items in admissible],
        "action_idxs": [-1, 0],
        "actions": list(actions),
    }
    return goal, rows, demonstration


class _TwoStepEnvironment:
    def __init__(self, goal: str, *, done_after_first: bool, done_after_last: bool):
        self.instruction_text = goal
        self.done_after_first = done_after_first
        self.done_after_last = done_after_last
        self.page = 0

    def reset(self, *, session: int) -> tuple[str, None]:
        self.page = 0
        return (
            f"WebShop\nInstruction:\n{self.instruction_text}\n"
            "[button] Search [button_]",
            None,
        )

    def get_available_actions(self) -> dict[str, object]:
        if self.page == 0:
            return {"has_search_bar": True, "clickables": ["search"]}
        return {"has_search_bar": False, "clickables": ["Buy Now"]}

    def step(self, chosen: str) -> tuple[str, float, bool, None]:
        self.page += 1
        if self.page == 1:
            return "[button] Buy Now [button_]", 0.0, self.done_after_first, None
        return "done", 1.0, self.done_after_last, None


class WebShopPromptParityTests(unittest.TestCase):
    def test_replay_fails_closed_at_configured_step_limit(self) -> None:
        goal, rows, demonstration = _two_step_fixture()
        report = replay_demonstration(
            environment=_TwoStepEnvironment(
                goal, done_after_first=False, done_after_last=True
            ),
            demonstration=demonstration,
            prepared_rows=rows,
            runtime_position=42,
            max_steps=1,
        )
        self.assertEqual(report["stopped_reason"], "max_steps_reached")
        self.assertEqual(report["status"], "failed")

    def test_replay_fails_when_environment_terminates_before_demo(self) -> None:
        goal, rows, demonstration = _two_step_fixture()
        report = replay_demonstration(
            environment=_TwoStepEnvironment(
                goal, done_after_first=True, done_after_last=True
            ),
            demonstration=demonstration,
            prepared_rows=rows,
            runtime_position=42,
            max_steps=8,
        )
        self.assertEqual(
            report["stopped_reason"], "environment_terminated_before_demonstration"
        )
        self.assertEqual(report["status"], "failed")

    def test_replay_fails_when_demo_ends_before_environment(self) -> None:
        goal, rows, demonstration = _two_step_fixture()
        report = replay_demonstration(
            environment=_TwoStepEnvironment(
                goal, done_after_first=False, done_after_last=False
            ),
            demonstration=demonstration,
            prepared_rows=rows,
            runtime_position=42,
            max_steps=8,
        )
        self.assertEqual(
            report["stopped_reason"], "demonstration_ended_before_environment"
        )
        self.assertEqual(report["status"], "failed")

    def test_replay_stops_at_non_executable_demonstrated_action(self) -> None:
        goal = "find red walking shoes"
        observation = "[button] Search [button_]"

        class DivergedEnvironment:
            instruction_text = goal

            def reset(self, *, session: int) -> tuple[str, None]:
                return f"WebShop\nInstruction:\n{goal}\n{observation}", None

            def get_available_actions(self) -> dict[str, object]:
                return {"has_search_bar": False, "clickables": ["Back"]}

            def step(self, chosen: str) -> None:
                raise AssertionError("a non-executable action must not be submitted")

        action = "search[red walking shoes]"
        report = replay_demonstration(
            environment=DivergedEnvironment(),
            demonstration={
                "states": [f"WebShop\nInstruction:\n{goal}\n{observation}"],
                "available_actions": [[action]],
                "action_idxs": [-1],
                "actions": [action],
            },
            prepared_rows=[
                {
                    "step_index": 0,
                    "prompt": render_webshop_policy_message(
                        task_description=goal,
                        current_observation=observation,
                        available_actions=("search[<your query>]",),
                    ),
                    "response": f"<think>x</think>\n<action>{action}</action>",
                }
            ],
            runtime_position=42,
            max_steps=8,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            report["stopped_reason"], "demonstrated_action_not_executable"
        )
        self.assertEqual(report["checked_steps"], 1)

    def test_replay_checks_search_then_product_decision(self) -> None:
        goal = "find red walking shoes"
        first = "[button] Search [button_]"
        results = "Results [button] ASIN1 [button_]"
        product = "Product [clicked button] red [clicked button_] [button] Buy Now [button_]"
        selected = "Product red [button] Buy Now [button_]"
        actions = (
            "search[red walking shoes]",
            "click[ASIN1]",
            "click[red]",
            "click[Buy Now]",
        )

        class SearchThenProductEnvironment:
            instruction_text = goal

            def __init__(self) -> None:
                self.page = 0

            def reset(self, *, session: int) -> tuple[str, None]:
                self.page = 0
                return f"WebShop\nInstruction:\n{goal}\n{first}", None

            def get_available_actions(self) -> dict[str, object]:
                if self.page == 0:
                    return {"has_search_bar": True, "clickables": ["search"]}
                if self.page == 1:
                    return {"has_search_bar": False, "clickables": ["ASIN1"]}
                if self.page == 2:
                    return {
                        "has_search_bar": False,
                        "clickables": ["red", "Buy Now"],
                    }
                return {"has_search_bar": False, "clickables": ["Buy Now"]}

            def step(self, chosen: str) -> tuple[str, float, bool, None]:
                expected = actions[self.page]
                if chosen != expected:
                    raise AssertionError("unexpected replay action")
                self.page += 1
                if self.page == 1:
                    return results, 0.0, False, None
                if self.page == 2:
                    return product, 0.0, False, None
                if self.page == 3:
                    return selected, 0.0, False, None
                if self.page == 4:
                    return "done", 1.0, True, None
                raise AssertionError("unexpected replay action")

        observations = (first, results, product, selected)
        admissible = (
            ("search[<your query>]",),
            ("click[ASIN1]",),
            ("click[red]", "click[Buy Now]"),
            ("click[Buy Now]",),
        )
        history: list[tuple[str, str]] = []
        rows = []
        for step_index, (observation, action, available) in enumerate(
            zip(observations, actions, admissible, strict=True)
        ):
            rows.append(
                {
                    "step_index": step_index,
                    "prompt": render_webshop_policy_message(
                        task_description=goal,
                        current_observation=observation,
                        available_actions=available,
                        history=history,
                    ),
                    "response": f"<think>x</think>\n<action>{action}</action>",
                }
            )
            history.append((observation, action))
        demonstration = {
            "states": [f"WebShop\nInstruction:\n{goal}\n{first}", *observations[1:]],
            "available_actions": [list(items) for items in admissible],
            "action_idxs": [-1, 0, 0, 0],
            "actions": list(actions),
        }
        report = replay_demonstration(
            environment=SearchThenProductEnvironment(),
            demonstration=demonstration,
            prepared_rows=rows,
            runtime_position=42,
            max_steps=8,
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["checked_steps"], 4)
        self.assertTrue(report["exact_online_prompt_match"])
        self.assertEqual(
            report["phase_coverage"],
            ["options", "product", "purchase", "results", "search"],
        )

    def test_replayed_product_step_reports_conditional_and_actual_prompt_parity(self) -> None:
        goal = "find red walking shoes"
        live_goal = goal + ", and price lower than 50.00 dollars"
        first_observation = "[button] Search [button_]"
        product_observation = "[button] Buy Now [button_]"
        history = ((first_observation, "search[red walking shoes]"),)
        offline_prompt = render_webshop_policy_message(
            task_description=goal,
            current_observation=product_observation,
            available_actions=("click[Buy Now]",),
            history=history,
        )
        result = compare_replayed_step(
            offline_prompt=offline_prompt,
            offline_goal=goal,
            offline_observation=product_observation,
            offline_actions=("click[Buy Now]",),
            offline_history=history,
            online_goal=live_goal,
            online_observation=product_observation,
            online_available_actions={
                "has_search_bar": False,
                "clickables": ["Buy Now"],
            },
            online_history=history,
            demonstrated_action="click[Buy Now]",
        )
        self.assertTrue(result["structural_passed"])
        self.assertFalse(result["exact_online_prompt_match"])
        self.assertTrue(result["demonstrated_action_executable"])
        self.assertNotIn("online_observation", result)

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
