from __future__ import annotations

import unittest

from infoskill.conditioning import NoSkillConditioner
from infoskill.domain.state import AgentHistoryEntry, CanonicalAgentState
from infoskill.episode import EnvironmentTransition, TaskSpec, TrajectoryCollector
from infoskill.rollout import GenerationRequest, GenerationResult


class _FakeEnvironment:
    def __init__(self, task: TaskSpec, rollout_id: int) -> None:
        self.task = task
        self.rollout_id = rollout_id
        self.step_index = 0
        self.history: tuple[AgentHistoryEntry, ...] = ()

    def _state(self, *, done: bool = False, won: bool = False) -> CanonicalAgentState:
        return CanonicalAgentState(
            task_id=self.task.task_id,
            split=self.task.split,
            task_type=self.task.task_type,
            goal=self.task.goal,
            step_index=self.step_index,
            observation="The fridge is closed.",
            history=self.history,
            admissible_commands=("look", "open fridge 1"),
            done=done,
            won=won,
        )

    def reset(self) -> CanonicalAgentState:
        return self._state()

    def step(self, action: str) -> EnvironmentTransition:
        before = self._state()
        won = self.rollout_id == 0 and action == "open fridge 1"
        self.history += (AgentHistoryEntry(self.step_index, before.observation, action),)
        self.step_index += 1
        after = self._state(done=won, won=won)
        return EnvironmentTransition(
            next_state=after,
            raw_observation=after.observation,
            raw_reward=float(won),
            raw_done=won,
            raw_won=won,
            info={},
        )

    def close(self) -> None:
        return None


class _FakeEnvironmentFactory:
    def create(self, task: TaskSpec, *, rollout_id: int, seed: int) -> _FakeEnvironment:
        return _FakeEnvironment(task, rollout_id)


class _FakeRolloutBackend:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def generate(self, requests: tuple[GenerationRequest, ...]) -> tuple[GenerationResult, ...]:
        self.batch_sizes.append(len(requests))
        results = []
        for request in requests:
            text = (
                "<action>open fridge 1</action>"
                if request.rollout_id == 0
                else "I might open fridge 1 after thinking."
            )
            results.append(
                GenerationResult(
                    request_id=request.request_id,
                    text=text,
                    finish_reason="stop",
                    token_ids=(1,),
                    token_logprobs=(-0.1,),
                    prompt_token_count=10,
                )
            )
        return tuple(results)


class TrajectoryCollectorTests(unittest.TestCase):
    def test_group_collection_batches_active_envs_and_preserves_invalid_failures(self) -> None:
        backend = _FakeRolloutBackend()
        collector = TrajectoryCollector(
            environment_factory=_FakeEnvironmentFactory(),
            conditioner=NoSkillConditioner(),
            rollout_backend=backend,
            max_steps=2,
            history_limit=2,
            invalid_action_penalty=0.01,
        )
        task = TaskSpec(
            task_id="game-1",
            split="train",
            task_type="pick_and_place_simple",
            goal="put the apple in the fridge",
        )

        group = collector.collect_task_group(task, rollouts_per_task=2, master_seed=0)

        self.assertEqual(backend.batch_sizes, [2, 1])
        self.assertEqual(len(group.trajectories), 2)
        self.assertTrue(group.trajectories[0].won)
        self.assertEqual(group.trajectories[0].reward, 1.0)
        self.assertFalse(group.trajectories[1].won)
        self.assertEqual(group.trajectories[1].invalid_action_count, 2)
        self.assertEqual(group.trajectories[1].reward, -0.02)
        self.assertEqual(group.trajectories[1].steps[-1].action.executed_action, "__invalid_action__")

    def test_multiple_task_groups_share_one_global_generation_batch(self) -> None:
        backend = _FakeRolloutBackend()
        collector = TrajectoryCollector(
            environment_factory=_FakeEnvironmentFactory(),
            conditioner=NoSkillConditioner(),
            rollout_backend=backend,
            max_steps=1,
            history_limit=2,
            invalid_action_penalty=0.01,
        )
        tasks = tuple(
            TaskSpec(
                task_id=f"game-{index}",
                split="train",
                task_type="pick_and_place_simple",
                goal="put the apple in the fridge",
            )
            for index in range(2)
        )

        groups = collector.collect_task_groups(tasks, rollouts_per_task=2, master_seed=0)

        self.assertEqual(len(groups), 2)
        self.assertEqual(backend.batch_sizes, [4])


if __name__ == "__main__":
    unittest.main()
