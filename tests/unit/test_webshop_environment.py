from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from infoskill.domain.actions import INVALID_ACTION_SENTINEL
from infoskill.episode import TaskSpec
from infoskill.integrations.webshop.environment import WebShopEnvironment
from infoskill.integrations.webshop.factory import (
    WebShopEnvironmentFactory,
    _required_index,
)


class _Server:
    product_item_dict = {
        "ASIN1": {"Title": "Blue Running Shoe"},
    }


class _RawWebShopEnvironment:
    def __init__(self) -> None:
        self.server = _Server()
        self.instruction_text = "Instruction: buy blue running shoes under 50 dollars"
        self.actions: list[str] = []
        self.closed = False

    def reset(self, session=None):
        self.session = session
        return "Search\n[button] Search [button_]", None

    def get_available_actions(self):
        return {"has_search_bar": True, "clickables": []}

    def step(self, action):
        self.actions.append(action)
        if action == INVALID_ACTION_SENTINEL:
            return "Search\n[button] Search [button_]", 0.0, False, {"invalid": True}
        return "Done", 0.75, True, {"purchase": True}

    def close(self):
        self.closed = True


class WebShopEnvironmentTests(unittest.TestCase):
    def test_reset_and_step_preserve_goal_session_actions_and_reward(self) -> None:
        raw = _RawWebShopEnvironment()
        task = TaskSpec(
            task_id="webshop-paper128-000",
            split="paper128",
            task_type="webshop",
            goal="buy blue running shoes under 50 dollars",
        )
        environment = WebShopEnvironment(raw, task=task, session_index=319)

        initial = environment.reset()
        transition = environment.step("search[blue running shoes]")

        self.assertEqual(raw.session, 319)
        self.assertEqual(initial.admissible_commands, ("search[<your query>]",))
        self.assertEqual(raw.actions, ["search[blue running shoes]"])
        self.assertEqual(transition.raw_reward, 0.75)
        self.assertTrue(transition.raw_done)
        self.assertFalse(transition.raw_won)
        self.assertEqual(transition.next_state.history[-1].executed_action, "search[blue running shoes]")

    def test_goal_mismatch_fails_closed(self) -> None:
        raw = _RawWebShopEnvironment()
        task = TaskSpec(
            task_id="webshop-paper128-000",
            split="paper128",
            task_type="webshop",
            goal="a different goal",
        )

        with self.assertRaisesRegex(RuntimeError, "different goal"):
            WebShopEnvironment(raw, task=task, session_index=319).reset()

    def test_invalid_action_is_a_local_noop(self) -> None:
        raw = _RawWebShopEnvironment()
        task = TaskSpec(
            task_id="webshop-paper128-000",
            split="paper128",
            task_type="webshop",
            goal="buy blue running shoes under 50 dollars",
        )
        environment = WebShopEnvironment(raw, task=task, session_index=319)
        initial = environment.reset()

        transition = environment.step(INVALID_ACTION_SENTINEL)

        self.assertEqual(raw.actions, [])
        self.assertEqual(transition.raw_reward, 0.0)
        self.assertFalse(transition.raw_done)
        self.assertEqual(transition.pre_world_state_checksum, transition.post_world_state_checksum)
        self.assertEqual(transition.next_state.observation, initial.observation)
        self.assertEqual(
            transition.next_state.history[-1].executed_action,
            INVALID_ACTION_SENTINEL,
        )

    def test_factory_rejects_tasks_outside_the_frozen_manifest(self) -> None:
        raw = _RawWebShopEnvironment()
        task = TaskSpec(
            task_id="webshop-paper128-000",
            split="paper128",
            task_type="webshop",
            goal="buy blue running shoes under 50 dollars",
        )
        factory = WebShopEnvironmentFactory(
            task_sessions={task.task_id: (task, 319)},
            raw_environment_factory=lambda task_id, rollout_id, seed: raw,
        )

        created = factory.create(task, rollout_id=2, seed=7)
        self.assertIsInstance(created, WebShopEnvironment)
        with self.assertRaisesRegex(ValueError, "frozen paper128"):
            factory.create(
                TaskSpec("unknown", "paper128", "webshop", "goal"),
                rollout_id=0,
                seed=0,
            )

    def test_small_index_must_be_bound_to_the_frozen_sources(self) -> None:
        sources = {
            name: {"sha256": f"sha-{name}", "bytes": index}
            for index, name in enumerate(
                ("products", "attributes", "human_instructions"),
                start=1,
            )
        }
        report = {
            "status": "complete",
            "catalog": "paper1000",
            "document_count": 1000,
            "sources": {name: dict(identity) for name, identity in sources.items()},
        }
        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(report)),
        ):
            index = _required_index(Path("webshop"), {"source_files": sources})
        self.assertEqual(index, Path("webshop/search_engine/indexes_1k"))

        report["sources"]["products"] = {"sha256": "different", "bytes": 1}
        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(report)),
        ):
            with self.assertRaisesRegex(RuntimeError, "source differs: products"):
                _required_index(Path("webshop"), {"source_files": sources})


if __name__ == "__main__":
    unittest.main()
