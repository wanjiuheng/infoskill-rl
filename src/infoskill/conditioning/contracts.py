from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from infoskill.domain.state import CanonicalAgentState, StateViews
from infoskill.skills import RetrievalResult


@dataclass(frozen=True, slots=True)
class ConditioningContext:
    candidate_skill_ids: tuple[str, ...] = ()
    retrieval: RetrievalResult | None = None


@dataclass(frozen=True, slots=True)
class ConditioningRequest:
    state: CanonicalAgentState
    views: StateViews
    rollout_id: int
    global_update: int
    latent_seed: int
    history_limit: int | None = None

    def __post_init__(self) -> None:
        if self.rollout_id < 0 or self.global_update < 0 or self.latent_seed < 0:
            raise ValueError("conditioning identities and seed must be non-negative")
        if self.history_limit is not None and self.history_limit < 0:
            raise ValueError("history_limit must be non-negative")


@dataclass(frozen=True, slots=True)
class ConditionedPolicyInput:
    user_message: str
    candidate_skill_ids: tuple[str, ...]
    soft_prefix: object | None = None
    conditioning_trace: object | None = None
    history_entries_omitted: int = 0
    history_entries_omitted_by_window: int = 0
    history_entries_omitted_for_prompt_budget: int = 0


@dataclass(frozen=True, slots=True)
class InfoSkillReplayTrace:
    latent_seed: int
    state_summary: object
    state_tokens: object
    posterior_mu: object
    posterior_logvar: object
    latent: object
    epsilon: object


@dataclass(frozen=True, slots=True)
class InfoSkillConditioningResult:
    soft_prefix: object
    replay_trace: InfoSkillReplayTrace


@dataclass(frozen=True, slots=True)
class InfoSkillConditioningWorkItem:
    compression_view: str
    candidate_skill_ids: tuple[str, ...]
    latent_seed: int
    latent_mode: Literal["sample", "mean"]

    def __post_init__(self) -> None:
        if not self.compression_view.strip():
            raise ValueError("INFO-SKILL compression view must not be empty")
        if not self.candidate_skill_ids:
            raise ValueError("INFO-SKILL conditioning requires candidate skills")
        if self.latent_seed < 0:
            raise ValueError("INFO-SKILL latent seed must be non-negative")
        if self.latent_mode not in {"sample", "mean"}:
            raise ValueError(f"unsupported INFO-SKILL latent mode: {self.latent_mode}")


class SkillConditioner(Protocol):
    def prepare_group(self, initial_state: CanonicalAgentState) -> ConditioningContext: ...

    def condition_batch(
        self,
        requests: tuple[ConditioningRequest, ...],
        context: ConditioningContext,
    ) -> tuple[ConditionedPolicyInput, ...]: ...


class InfoSkillConditioningRuntime(Protocol):
    def condition_infoskill(
        self,
        requests: tuple[ConditioningRequest, ...],
        candidate_skill_ids: tuple[str, ...],
        *,
        latent_mode: Literal["sample", "mean"],
    ) -> tuple[InfoSkillConditioningResult, ...]: ...
