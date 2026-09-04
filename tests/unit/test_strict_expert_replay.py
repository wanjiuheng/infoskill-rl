from __future__ import annotations

import unittest

from infoskill.domain.state import AgentHistoryEntry, CanonicalAgentState
from infoskill.episode import EnvironmentTransition, TaskSpec
from infoskill.integrations.alfworld import StrictExpertReplay


class _ExpertReplayEnvironment:
    def __init__(self) -> None:
        self.state = CanonicalAgentState(
            task_id="game-1",
            split="train",
            task_type="pick_and_place_simple",
            goal="put an apple in the fridge",
            step_index=0,
            observation="Kitchen.",
            history=(),
            admissible_commands=("look", "open fridge 1"),
        )

    def reset(self) -> CanonicalAgentState:
        return self.state

    def expert_payload(self):
        return {"feedback": self.state.observation, "admissible_commands": self.state.admissible_commands}

    def step(self, action: str) -> EnvironmentTransition:
        won = action == "open fridge 1"
        self.state = CanonicalAgentState(
            task_id=self.state.task_id,
            split=self.state.split,
            task_type=self.state.task_type,
            goal=self.state.goal,
            step_index=self.state.step_index + 1,
            observation="Done." if won else "Kitchen after looking.",
            history=self.state.history
            + (AgentHistoryEntry(self.state.step_index, self.state.observation, action),),
            admissible_commands=("look", "open fridge 1"),
            done=won,
            won=won,
        )
        return EnvironmentTransition(self.state, self.state.observation, float(won), won, won, {})

    def close(self) -> None:
        return None


class _Expert:
    def __init__(self, next_action: str) -> None:
        self.next_action = next_action

    def reset(self, gamefile: str) -> None:
        return None

    def observe(self, feedback: str) -> None:
        return None

    def act(self, game_state, reward: float, done: bool, last_action: str) -> str:
        return self.next_action


class StrictExpertReplayTests(unittest.TestCase):
    def test_non_admissible_expert_action_quarantines_the_whole_game(self) -> None:
        task = TaskSpec(
            task_id="game-1",
            split="train",
            task_type="pick_and_place_simple",
            goal="put an apple in the fridge",
            environment_path="/data/game.tw-pddl",
        )

        result = StrictExpertReplay(max_replay_steps=150, persist_horizon=30).run(
            task=task,
            environment=_ExpertReplayEnvironment(),
            expert=_Expert("dance"),
        )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.quarantine_reason, "expert_action_not_admissible")
        self.assertEqual(result.samples, ())
        self.assertEqual(result.total_steps, 1)


if __name__ == "__main__":
    unittest.main()
