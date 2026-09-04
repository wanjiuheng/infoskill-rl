from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Protocol

from infoskill.domain.actions import resolve_action
from infoskill.domain.state import CanonicalAgentState
from infoskill.episode import Environment, TaskSpec


class ExpertPolicy(Protocol):
    def reset(self, gamefile: str) -> None: ...

    def observe(self, feedback: str) -> None: ...

    def act(
        self,
        game_state: Mapping[str, object],
        reward: float,
        done: bool,
        last_action: str,
    ) -> str: ...


class ExpertReplayEnvironment(Environment, Protocol):
    def expert_payload(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class GroundingSample:
    state: CanonicalAgentState
    expert_action: str


@dataclass(frozen=True, slots=True)
class ExpertReplayResult:
    task_id: str
    succeeded: bool
    samples: tuple[GroundingSample, ...]
    total_steps: int
    quarantine_reason: str | None


class StrictExpertReplay:
    def __init__(self, *, max_replay_steps: int = 150, persist_horizon: int = 30) -> None:
        if max_replay_steps <= 0 or persist_horizon <= 0:
            raise ValueError("expert replay limits must be positive")
        if persist_horizon > max_replay_steps:
            raise ValueError("persist_horizon cannot exceed max_replay_steps")
        self._max_replay_steps = max_replay_steps
        self._persist_horizon = persist_horizon

    def run(
        self,
        *,
        task: TaskSpec,
        environment: ExpertReplayEnvironment,
        expert: ExpertPolicy,
        candidate_skill_ids: tuple[str, ...] = (),
    ) -> ExpertReplayResult:
        samples: list[GroundingSample] = []
        total_steps = 0
        try:
            if task.environment_path is None:
                return self._quarantine(task, total_steps, "missing_gamefile")
            state = environment.reset()
            state = replace(state, candidate_skill_ids=candidate_skill_ids)
            payload = environment.expert_payload()
            expert.reset(task.environment_path)
            expert.observe(str(payload.get("feedback", state.observation)))
            last_action = ""
            last_reward = 0.0

            for decision_index in range(self._max_replay_steps):
                if decision_index == 0:
                    proposed_action = "look"
                else:
                    proposed_action = expert.act(payload, last_reward, state.done, last_action)
                resolution = resolve_action(
                    f"<action>{proposed_action}</action>",
                    state.admissible_commands,
                )
                if not resolution.is_executable or resolution.resolved_action is None:
                    return self._quarantine(task, total_steps, "expert_action_not_admissible")

                samples.append(GroundingSample(state=state, expert_action=resolution.resolved_action))
                transition = environment.step(resolution.resolved_action)
                total_steps += 1
                state = transition.next_state
                state = replace(state, candidate_skill_ids=candidate_skill_ids)
                payload = environment.expert_payload()
                last_action = resolution.resolved_action
                last_reward = transition.raw_reward

                if state.won:
                    return ExpertReplayResult(
                        task_id=task.task_id,
                        succeeded=True,
                        samples=tuple(samples[: self._persist_horizon]),
                        total_steps=total_steps,
                        quarantine_reason=None,
                    )
                if state.done:
                    return self._quarantine(task, total_steps, "terminated_without_win")
            return self._quarantine(task, total_steps, "expert_replay_limit")
        except Exception as error:
            return self._quarantine(task, total_steps, f"expert_exception:{type(error).__name__}")
        finally:
            environment.close()

    @staticmethod
    def _quarantine(task: TaskSpec, total_steps: int, reason: str) -> ExpertReplayResult:
        return ExpertReplayResult(
            task_id=task.task_id,
            succeeded=False,
            samples=(),
            total_steps=total_steps,
            quarantine_reason=reason,
        )
