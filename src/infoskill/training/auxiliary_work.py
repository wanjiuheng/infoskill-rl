from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from infoskill.conditioning import InfoSkillReplayTrace
from infoskill.episode import TrajectoryGroup
from infoskill.integrations.alfworld import GroundingSample


@dataclass(frozen=True, slots=True)
class OnlineAuxiliaryExample:
    """One rollout state with its trajectory-balanced fidelity weight."""

    replay_trace: InfoSkillReplayTrace
    candidate_skill_ids: tuple[str, ...]
    fidelity_target: float
    trajectory_index: int
    trajectory_weight: float
    eligible: bool = True


@dataclass(frozen=True, slots=True)
class OfflineAuxiliaryExample:
    """One deterministic expert state used by grounding and offline rate."""

    sample: GroundingSample
    epsilon_seed: int
    eligible: bool = True


@dataclass(frozen=True, slots=True)
class AuxiliaryRankWork:
    """Equal-call work assigned to one DDP rank."""

    online: tuple[OnlineAuxiliaryExample, ...]
    offline: tuple[OfflineAuxiliaryExample, ...]
    global_trajectory_count: int
    global_offline_count: int
    micro_batch_size: int


def collect_online_auxiliary_examples(
    groups: Sequence[TrajectoryGroup],
    fidelity_targets: Sequence[Sequence[float]],
) -> tuple[OnlineAuxiliaryExample, ...]:
    """Flatten trajectories without losing per-trajectory normalization."""

    if len(groups) != len(fidelity_targets):
        raise ValueError("groups and fidelity targets must have equal length")
    examples: list[OnlineAuxiliaryExample] = []
    trajectory_index = 0
    for group, targets in zip(groups, fidelity_targets):
        if len(group.trajectories) != len(targets):
            raise ValueError(
                "each group requires one fidelity target per trajectory"
            )
        for trajectory, target in zip(group.trajectories, targets):
            if not trajectory.steps:
                raise ValueError("online auxiliary trajectories must not be empty")
            trajectory_weight = 1.0 / len(trajectory.steps)
            for step in trajectory.steps:
                trace = step.conditioned_input.conditioning_trace
                if not isinstance(trace, InfoSkillReplayTrace):
                    raise TypeError(
                        "online INFO-SKILL steps require InfoSkillReplayTrace data"
                    )
                skill_ids = step.conditioned_input.candidate_skill_ids
                if not skill_ids:
                    raise ValueError(
                        "online INFO-SKILL step has no candidate skills"
                    )
                examples.append(
                    OnlineAuxiliaryExample(
                        replay_trace=trace,
                        candidate_skill_ids=skill_ids,
                        fidelity_target=float(target),
                        trajectory_index=trajectory_index,
                        trajectory_weight=trajectory_weight,
                    )
                )
            trajectory_index += 1
    if not examples:
        raise ValueError("online auxiliary work must not be empty")
    return tuple(examples)


def build_offline_auxiliary_examples(
    samples: Sequence[GroundingSample],
    epsilon_seeds: Sequence[int],
) -> tuple[OfflineAuxiliaryExample, ...]:
    if not samples or len(samples) != len(epsilon_seeds):
        raise ValueError(
            "grounding samples and epsilon seeds must be non-empty and equal length"
        )
    if any(seed < 0 for seed in epsilon_seeds):
        raise ValueError("grounding epsilon seeds must be non-negative")
    return tuple(
        OfflineAuxiliaryExample(sample=sample, epsilon_seed=int(seed))
        for sample, seed in zip(samples, epsilon_seeds)
    )


def partition_auxiliary_work(
    *,
    online: Sequence[OnlineAuxiliaryExample],
    offline: Sequence[OfflineAuxiliaryExample],
    world_size: int,
    micro_batch_size: int,
) -> tuple[AuxiliaryRankWork, ...]:
    """Create equal micro-batch counts while marking padding ineligible."""

    if world_size <= 0 or micro_batch_size <= 0:
        raise ValueError("world_size and micro_batch_size must be positive")
    if not online or not offline:
        raise ValueError("online and offline auxiliary work must not be empty")
    trajectory_count = len({item.trajectory_index for item in online})
    if trajectory_count <= 0:
        raise ValueError("online work requires at least one trajectory")
    online_by_rank = _partition_equal_calls(
        online,
        world_size=world_size,
        micro_batch_size=micro_batch_size,
    )
    offline_by_rank = _partition_equal_calls(
        offline,
        world_size=world_size,
        micro_batch_size=micro_batch_size,
    )
    return tuple(
        AuxiliaryRankWork(
            online=online_by_rank[rank],
            offline=offline_by_rank[rank],
            global_trajectory_count=trajectory_count,
            global_offline_count=len(offline),
            micro_batch_size=micro_batch_size,
        )
        for rank in range(world_size)
    )


def _partition_equal_calls(
    examples: Sequence[OnlineAuxiliaryExample] | Sequence[OfflineAuxiliaryExample],
    *,
    world_size: int,
    micro_batch_size: int,
) -> tuple[tuple, ...]:
    local_size = math.ceil(
        len(examples) / (world_size * micro_batch_size)
    ) * micro_batch_size
    assigned = [list(examples[rank::world_size]) for rank in range(world_size)]
    fallback = examples[0]
    for rank, items in enumerate(assigned):
        template = items[-1] if items else fallback
        while len(items) < local_size:
            items.append(replace(template, eligible=False))
        if len(items) != local_size:
            raise RuntimeError(
                f"auxiliary partition overflow on rank {rank}: {len(items)}"
            )
    return tuple(tuple(items) for items in assigned)
