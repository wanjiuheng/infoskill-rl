from __future__ import annotations

import unittest

from infoskill.integrations.webshop.action_adapter import DemonstrationActionAdapter


class _RawEnvironment:
    observation_mode = "text"
    instruction_text = "find a red shoe"

    class _Server:
        product_item_dict = {
            "B000000001": {"Title": "Red Walking Shoe"},
            "B000000002": {"Title": "Blue Travel Bag"},
        }

    server = _Server()

    def __init__(self) -> None:
        self.submitted: list[str] = []

    def get_available_actions(self) -> dict[str, object]:
        return {
            "has_search_bar": False,
            "clickables": ["b000000001", "buy now", "B000000002"],
        }

    def step(self, action: str) -> tuple[str, float, bool, None]:
        self.submitted.append(action)
        return "next", 0.0, False, None


class DemonstrationActionAdapterTests(unittest.TestCase):
    def test_translates_asin_candidates_to_official_item_title_actions(self) -> None:
        raw = _RawEnvironment()
        adapter = DemonstrationActionAdapter(raw)

        self.assertEqual(
            adapter.get_available_actions(),
            {
                "has_search_bar": False,
                "clickables": [
                    "item - red walking shoe",
                    "buy now",
                    "item - blue travel bag",
                ],
            },
        )

    def test_translates_selected_item_title_back_to_asin(self) -> None:
        raw = _RawEnvironment()
        adapter = DemonstrationActionAdapter(raw)

        result = adapter.step("click[item - red walking shoe]")

        self.assertEqual(raw.submitted, ["click[b000000001]"])
        self.assertEqual(result, ("next", 0.0, False, None))

    def test_passes_non_product_actions_through(self) -> None:
        raw = _RawEnvironment()
        adapter = DemonstrationActionAdapter(raw)

        adapter.step("search[red walking shoe]")
        adapter.step("click[buy now]")

        self.assertEqual(
            raw.submitted,
            ["search[red walking shoe]", "click[buy now]"],
        )

    def test_observation_mode_and_other_attributes_delegate(self) -> None:
        raw = _RawEnvironment()
        adapter = DemonstrationActionAdapter(raw)

        adapter.observation_mode = "text_rich"

        self.assertEqual(raw.observation_mode, "text_rich")
        self.assertEqual(adapter.instruction_text, "find a red shoe")


if __name__ == "__main__":
    unittest.main()
