from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from infoskill.conditioning import ConditionedPolicyInput
from infoskill.domain.actions import ActionResolution
from infoskill.domain.state import CanonicalAgentState
from infoskill.rollout import GenerationResult


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    split: str
    task_type: str
    goal: str
    environment_path: str | None = None
    trajectory_path: str | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentTransition:
    next_state: CanonicalAgentState
    raw_observation: str
    raw_reward: float
    raw_done: bool
    raw_won: bool
    info: Mapping[str, object]
    pre_world_state_checksum: str | None = None
    post_world_state_checksum: str | None = None


class Environment(Protocol):
    def reset(self) -> CanonicalAgentState: ...

    def step(self, action: str) -> EnvironmentTransition: ...

    def close(self) -> None: ...


class EnvironmentFactory(Protocol):
    def create(self, task: TaskSpec, *, rollout_id: int, seed: int) -> Environment: ...


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    state_before: CanonicalAgentState
    conditioned_input: ConditionedPolicyInput
    generation: GenerationResult
    action: ActionResolution
    transition: EnvironmentTransition


@dataclass(frozen=True, slots=True)
class Trajectory:
    task: TaskSpec
    rollout_id: int
    steps: tuple[TrajectoryStep, ...]
    won: bool
    environment_done: bool
    horizon_exhausted: bool
    invalid_action_count: int
    reward: float


@dataclass(frozen=True, slots=True)
class TrajectoryGroup:
    task: TaskSpec
    trajectories: tuple[Trajectory, ...]
