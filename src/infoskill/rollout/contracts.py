from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class PromptLengthError(RuntimeError):
    """A complete policy input exceeded the registered prompt budget."""

    def __init__(
        self,
        *,
        request_id: str,
        token_count: int,
        max_prompt_tokens: int,
        user_message: str,
    ) -> None:
        self.request_id = request_id
        self.token_count = token_count
        self.max_prompt_tokens = max_prompt_tokens
        self.user_message = user_message
        super().__init__(
            f"policy prompt for {request_id} has {token_count} tokens, exceeding "
            f"{max_prompt_tokens}; no silent truncation"
        )

    def as_dict(self) -> dict[str, int | str]:
        return {
            "request_id": self.request_id,
            "token_count": self.token_count,
            "max_prompt_tokens": self.max_prompt_tokens,
            "user_message": self.user_message,
        }


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
