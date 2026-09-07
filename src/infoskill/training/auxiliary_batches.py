from __future__ import annotations

from typing import Sequence, cast

import torch
from torch import Tensor

from infoskill.conditioning import InfoSkillReplayTrace
from infoskill.domain.actions import resolve_action
from infoskill.domain.state import render_state_views
from infoskill.episode import TrajectoryGroup
from infoskill.integrations.alfworld import GroundingSample
from infoskill.learning import (
    AuxiliaryTrainingBatch,
    CompressionReplayBatch,
    OfflineGroundingBatch,
    OnlineAuxiliaryBatch,
)
from infoskill.semantic import FrozenSemanticEncoder, SemanticFeatureCache
from infoskill.skills import FixedSkillLibrary, SkillRecord


class AuxiliaryBatchBuilder:
    """Turn online trajectories and expert samples into one replayable aux batch."""

    def __init__(
        self,
        *,
        library: FixedSkillLibrary,
        semantic_encoder: FrozenSemanticEncoder,
        feature_cache: SemanticFeatureCache,
        history_length: int,
        latent_dim: int,
    ) -> None:
        if history_length < 0 or latent_dim <= 0:
            raise ValueError("history_length must be non-negative and latent_dim positive")
        self.library = library
        self.semantic_encoder = semantic_encoder
        self.feature_cache = feature_cache
        self.history_length = history_length
        self.latent_dim = latent_dim

    def build(
        self,
        *,
        groups: Sequence[TrajectoryGroup],
        fidelity_targets: Sequence[Sequence[float]],
        grounding_samples: Sequence[GroundingSample],
        grounding_epsilon_seeds: Sequence[int],
    ) -> AuxiliaryTrainingBatch:
        online = self._online(groups, fidelity_targets)
        offline = self._offline(grounding_samples, grounding_epsilon_seeds)
        batch = AuxiliaryTrainingBatch(online=online, offline=offline)
        batch.validate()
        return batch

    def _online(
        self,
        groups: Sequence[TrajectoryGroup],
        fidelity_targets: Sequence[Sequence[float]],
    ) -> OnlineAuxiliaryBatch:
        if len(groups) != len(fidelity_targets):
            raise ValueError("groups and fidelity targets must have equal length")
        traces: list[InfoSkillReplayTrace] = []
        skill_groups: list[tuple[SkillRecord, ...]] = []
        targets: list[float] = []
        trajectory_indices: list[int] = []
        trajectory_index = 0
        for group, group_targets in zip(groups, fidelity_targets):
            if len(group.trajectories) != len(group_targets):
                raise ValueError(
                    "each group requires one fidelity target per trajectory"
                )
            for trajectory, target in zip(group.trajectories, group_targets):
                for step in trajectory.steps:
                    trace = step.conditioned_input.conditioning_trace
                    if not isinstance(trace, InfoSkillReplayTrace):
                        raise TypeError(
                            "online INFO-SKILL steps require InfoSkillReplayTrace data"
                        )
                    skill_ids = step.conditioned_input.candidate_skill_ids
                    if not skill_ids:
                        raise ValueError("online INFO-SKILL step has no candidate skills")
                    traces.append(trace)
                    skill_groups.append(
                        tuple(self.library.get(skill_id) for skill_id in skill_ids)
                    )
                    targets.append(float(target))
                    trajectory_indices.append(trajectory_index)
                trajectory_index += 1
        if not traces:
            raise ValueError("online auxiliary batch requires at least one trajectory step")
        state_tokens, state_valid = _pad_state_features(
            traces,
            device=self.semantic_encoder.device,
        )
        skill_tokens, skill_valid, skill_kind_ids = (
            self.feature_cache.heterogeneous_skill_batch(skill_groups)
        )
        return OnlineAuxiliaryBatch(
            replay=CompressionReplayBatch(
                state_tokens=state_tokens,
                state_valid=state_valid,
                skill_tokens=skill_tokens,
                skill_valid=skill_valid,
                skill_kind_ids=skill_kind_ids,
                epsilon=torch.stack(
                    [cast(Tensor, trace.epsilon) for trace in traces]
                ).to(device=state_tokens.device),
            ),
            trajectory_index=torch.tensor(
                trajectory_indices,
                device=state_tokens.device,
                dtype=torch.long,
            ),
            fidelity_target=torch.tensor(
                targets,
                device=state_tokens.device,
                dtype=torch.float32,
            ),
        )

    def _offline(
        self,
        samples: Sequence[GroundingSample],
        epsilon_seeds: Sequence[int],
    ) -> OfflineGroundingBatch:
        if not samples:
            raise ValueError("offline auxiliary batch requires grounding samples")
        if len(samples) != len(epsilon_seeds):
            raise ValueError("grounding samples and epsilon seeds must have equal length")
        if any(seed < 0 for seed in epsilon_seeds):
            raise ValueError("grounding epsilon seeds must be non-negative")
        features = self.semantic_encoder.encode_tokens(
            [
                render_state_views(
                    sample.state,
                    history_limit=self.history_length,
                ).compression_view
                for sample in samples
            ]
        )
        skill_groups = [
            tuple(
                self.library.get(skill_id)
                for skill_id in sample.state.candidate_skill_ids
            )
            for sample in samples
        ]
        skill_tokens, skill_valid, skill_kind_ids = (
            self.feature_cache.heterogeneous_skill_batch(skill_groups)
        )
        command_groups = [sample.state.admissible_commands for sample in samples]
        command_embeddings, command_valid = self.feature_cache.command_batch(
            command_groups
        )
        grounding_targets = [
            _grounding_target(sample) for sample in samples
        ]
        epsilon = torch.stack(
            [
                torch.randn(
                    self.latent_dim,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                )
                for seed in epsilon_seeds
            ]
        ).to(device=features.tokens.device)
        return OfflineGroundingBatch(
            replay=CompressionReplayBatch(
                state_tokens=features.tokens.detach(),
                state_valid=features.valid.detach(),
                skill_tokens=skill_tokens,
                skill_valid=skill_valid,
                skill_kind_ids=skill_kind_ids,
                epsilon=epsilon,
            ),
            command_embeddings=command_embeddings,
            command_valid=command_valid,
            grounding_target=torch.tensor(
                grounding_targets,
                device=features.tokens.device,
                dtype=torch.long,
            ),
        )


def _pad_state_features(
    traces: Sequence[InfoSkillReplayTrace],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    rows = [cast(Tensor, trace.state_tokens) for trace in traces]
    if any(row.ndim != 2 for row in rows):
        raise ValueError("online replay state tokens must be [tokens, width]")
    widths = {int(row.shape[-1]) for row in rows}
    devices = {row.device for row in rows}
    dtypes = {row.dtype for row in rows}
    if len(widths) != 1 or len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("online replay state features must share width, device, and dtype")
    max_tokens = max(int(row.shape[0]) for row in rows)
    output = torch.zeros(
        (len(rows), max_tokens, next(iter(widths))),
        device=device,
        dtype=rows[0].dtype,
    )
    valid = torch.zeros(
        (len(rows), max_tokens),
        device=device,
        dtype=torch.bool,
    )
    for index, row in enumerate(rows):
        length = row.shape[0]
        output[index, :length] = row.to(device=device)
        valid[index, :length] = True
    return output, valid


def _grounding_target(sample: GroundingSample) -> int:
    resolution = resolve_action(
        f"<action>{sample.expert_action}</action>",
        sample.state.admissible_commands,
    )
    if not resolution.is_executable or resolution.resolved_action is None:
        raise ValueError("expert grounding action is not uniquely admissible")
    return sample.state.admissible_commands.index(resolution.resolved_action)
