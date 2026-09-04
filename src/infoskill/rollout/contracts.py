from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class GenerationParameters:
    do_sample: bool
    temperature: float
    top_p: float
    max_new_tokens: int

    @classmethod
    def training(cls) -> "GenerationParameters":
        return cls(do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=256)

    @classmethod
    def evaluation(cls) -> "GenerationParameters":
        return cls(do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=256)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    request_id: str
    task_id: str
    rollout_id: int
    env_step: int
    user_message: str
    parameters: GenerationParameters
    soft_prefix: object | None = None
    seed: int = 0


@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    text: str
    finish_reason: str
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    prompt_token_count: int

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_logprobs):
            raise ValueError("token_ids and token_logprobs must have equal length")
        if self.prompt_token_count < 0:
            raise ValueError("prompt_token_count must be non-negative")


class RolloutBackend(Protocol):
    def generate(self, requests: tuple[GenerationRequest, ...]) -> tuple[GenerationResult, ...]: ...
