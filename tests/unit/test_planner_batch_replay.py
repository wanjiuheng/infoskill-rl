from __future__ import annotations

import unittest
from dataclasses import replace

from infoskill.domain.state import CanonicalAgentState
from infoskill.episode import EnvironmentTransition, TaskSpec
from infoskill.integrations.alfworld import StrictPlannerBatchReplay


class _ScriptedBatch:
    def __init__(self, tasks: tuple[TaskSpec, ...]) -> None:
        self.tasks = tasks
        self.states = [self._state(task, step=0) for task in tasks]
        self.actions: list[tuple[str | None, ...]] = []
        self.closed = False
        self.forced_worker_terminations = 0

    @staticmethod
    def _state(
        task: TaskSpec,
        *,
        step: int,
        done: bool = False,
        won: bool = False,
    ) -> CanonicalAgentState:
        return CanonicalAgentState(
            task_id=task.task_id,
            split="train",
            task_type=task.task_type,
            goal=task.goal,
            step_index=step,
            observation=f"observation-{task.task_id}-{step}",
            history=(),
            admissible_commands=(
                "look",
                f"advance {task.task_id}",
                f"finish {task.task_id}",
            ),
            done=done,
            won=won,
        )

    def reset(self) -> tuple[CanonicalAgentState, ...]:
        return tuple(self.states)

    def expert_payloads(self):
        return tuple(
            {
                "feedback": state.observation,
                "admissible_commands": state.admissible_commands,
                "extra.expert_plan": [
                    (
                        f"advance {state.task_id}"
                        if state.task_id == "task-1" and state.step_index == 1
                        else f"finish {state.task_id}"
                    )
                ],
            }
            for state in self.states
        )

    def step(self, actions: tuple[str | None, ...]):
        self.actions.append(actions)
        transitions = []
        for index, (task, action) in enumerate(zip(self.tasks, actions)):
            if action is None:
                transitions.append(None)
                continue
            before = self.states[index]
            won = action == f"finish {task.task_id}"
            after = replace(
                self._state(task, step=before.step_index + 1, done=won, won=won),
                candidate_skill_ids=before.candidate_skill_ids,
            )
            self.states[index] = after
            transitions.append(
                EnvironmentTransition(
                    next_state=after,
                    raw_observation=after.observation,
                    raw_reward=float(won),
                    raw_done=won,
                    raw_won=won,
                    info={},
                )
            )
        return tuple(transitions)

    def close(self) -> None:
        self.closed = True


class PlannerBatchReplayTests(unittest.TestCase):
    def test_replays_planner_slots_in_lockstep_without_cross_contamination(self) -> None:
        tasks = tuple(
            TaskSpec(
                task_id=f"task-{index}",
                split="train",
                task_type="pick_and_place_simple",
                goal=f"goal-{index}",
                environment_path=f"/data/task-{index}/game.tw-pddl",
            )
            for index in range(3)
        )
        environment = _ScriptedBatch(tasks)

        results = StrictPlannerBatchReplay(
            max_replay_steps=10,
            persist_horizon=10,
        ).run(
            tasks=tasks,
            environment=environment,
            candidate_skill_ids=(("skill-a",), ("skill-b",), ("skill-c",)),
        )

        self.assertTrue(environment.closed)
        self.assertEqual(environment.actions[0], ("look", "look", "look"))
        self.assertEqual(
            environment.actions[1],
            ("finish task-0", "advance task-1", "finish task-2"),
        )
        self.assertEqual(
            environment.actions[2],
            (None, "finish task-1", None),
        )
        self.assertEqual([result.task_id for result in results], [task.task_id for task in tasks])
        self.assertTrue(all(result.succeeded for result in results))
        self.assertEqual([result.total_steps for result in results], [2, 3, 2])
        self.assertEqual(
            [sample.state.candidate_skill_ids for sample in results[0].samples],
            [("skill-a",), ("skill-a",)],
        )


if __name__ == "__main__":
    unittest.main()
