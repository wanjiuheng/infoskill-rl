from __future__ import annotations

import unittest

from infoskill.episode import TaskSpec
from infoskill.integrations.alfworld import AlfworldEnvironment


class _RawBatchSizeOneEnvironment:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def reset(self):
        return (
            ["You are in a kitchen.\nYour task is to: Put an apple in the fridge."],
            {
                "admissible_commands": [["look", "open fridge 1"]],
                "won": [False],
                "facts": [["fridge_closed", "apple_on_table"]],
                "extra.gamefile": ["/data/game.tw-pddl"],
            },
        )

    def step(self, actions):
        self.actions.extend(actions)
        return (
            ["I don't understand that command."],
            [0.0],
            [False],
            {
                "admissible_commands": [["look", "open fridge 1"]],
                "won": [False],
                "facts": [["fridge_closed", "apple_on_table"]],
                "extra.gamefile": ["/data/game.tw-pddl"],
            },
        )

    def close(self) -> None:
        return None


class AlfworldEnvironmentTests(unittest.TestCase):
    def test_batch_one_output_becomes_canonical_state_and_raw_transition(self) -> None:
        raw = _RawBatchSizeOneEnvironment()
        task = TaskSpec(
            task_id="game-1",
            split="train",
            task_type="pick_and_place_simple",
            goal="dataset annotation wording",
            environment_path="/data/game.tw-pddl",
        )
        environment = AlfworldEnvironment(raw, task=task)

        initial = environment.reset()
        transition = environment.step("__invalid_action__")

        self.assertEqual(initial.goal, "Put an apple in the fridge.")
        self.assertEqual(initial.observation, "You are in a kitchen.")
        self.assertEqual(raw.actions, ["__invalid_action__"])
        self.assertEqual(transition.raw_observation, "I don't understand that command.")
        self.assertEqual(transition.next_state.history[-1].executed_action, "__invalid_action__")
        self.assertEqual(transition.pre_world_state_checksum, transition.post_world_state_checksum)


if __name__ == "__main__":
    unittest.main()
