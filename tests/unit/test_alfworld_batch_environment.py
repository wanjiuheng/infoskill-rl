from __future__ import annotations

import unittest
from dataclasses import replace

from infoskill.episode import TaskSpec
from infoskill.integrations.alfworld.batch_environment import AlfworldEnvironmentBatch


class _RawBatchEnvironment:
    def __init__(self, gamefiles: tuple[str, ...]) -> None:
        self.gamefiles = list(gamefiles)
        self._gamefiles_iterator = iter(reversed(self.gamefiles))
        self.loaded_gamefiles: list[str] = []
        self.actions: list[tuple[str, ...]] = []

    def reset(self):
        self.loaded_gamefiles = [
            next(self._gamefiles_iterator) for _ in range(len(self.gamefiles))
        ]
        observations = [
            f"Kitchen {index}.\nYour task is to: Goal {index}."
            for index in range(len(self.gamefiles))
        ]
        return observations, self._infos(won=False)

    def step(self, actions):
        self.actions.append(tuple(actions))
        observations = [f"After {action}." for action in actions]
        return (
            observations,
            [0.0] * len(actions),
            [False] * len(actions),
            self._infos(won=False),
        )

    def _infos(self, *, won: bool):
        return {
            "admissible_commands": [
                ["look", f"open cabinet {index}"]
                for index in range(len(self.gamefiles))
            ],
            "won": [won] * len(self.gamefiles),
            "facts": [[f"slot-{index}"] for index in range(len(self.gamefiles))],
            "extra.gamefile": list(self.loaded_gamefiles),
        }

    def close(self) -> None:
        return None


class AlfworldEnvironmentBatchTests(unittest.TestCase):
    def test_batch_preserves_slot_order_and_returns_canonical_transitions(self) -> None:
        tasks = tuple(
            TaskSpec(
                task_id=f"game-{index}",
                split="train",
                task_type="pick_and_place_simple",
                goal=f"dataset goal {index}",
                environment_path=f"/data/game-{index}.tw-pddl",
            )
            for index in range(2)
        )
        raw = _RawBatchEnvironment(
            tuple(task.environment_path for task in tasks if task.environment_path)
        )
        batch = AlfworldEnvironmentBatch(raw, tasks=tasks)

        states = batch.reset()
        transitions = batch.step(("look", "open cabinet 1"))

        self.assertEqual(raw.loaded_gamefiles, [task.environment_path for task in tasks])
        self.assertEqual([state.task_id for state in states], ["game-0", "game-1"])
        self.assertEqual([state.goal for state in states], ["Goal 0.", "Goal 1."])
        self.assertEqual(raw.actions, [("look", "open cabinet 1")])
        self.assertEqual(
            [transition.next_state.task_id for transition in transitions if transition],
            ["game-0", "game-1"],
        )

    def test_inactive_slots_are_stepped_but_not_returned(self) -> None:
        tasks = tuple(
            TaskSpec(
                task_id=f"game-{index}",
                split="train",
                task_type="pick_and_place_simple",
                goal="goal",
                environment_path=f"/data/game-{index}.tw-pddl",
            )
            for index in range(2)
        )
        raw = _RawBatchEnvironment(tuple(task.environment_path for task in tasks))
        batch = AlfworldEnvironmentBatch(raw, tasks=tasks)
        batch.reset()
        batch._states[0] = replace(batch._states[0], done=True)  # type: ignore[attr-defined]

        transitions = batch.step((None, "look"))

        self.assertIsNone(transitions[0])
        self.assertIsNotNone(transitions[1])
        self.assertEqual(raw.actions, [("look", "look")])


if __name__ == "__main__":
    unittest.main()
