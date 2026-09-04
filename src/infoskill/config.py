from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SkillMode(str, Enum):
    NO_SKILL = "no_skill"
    RAW_SKILL_PROMPT = "raw_skill_prompt"
    INFO_SKILL = "infoskill"


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    max_steps: int = 30
    history_length: int = 2
    invalid_action_penalty: float = 0.01


@dataclass(frozen=True, slots=True)
class BatchConfig:
    task_groups_per_update: int = 8
    rollouts_per_task: int = 8
    action_minibatch_size: int = 256
    max_tokens_per_gpu: int = 16_384

    @property
    def trajectories_per_update(self) -> int:
        return self.task_groups_per_update * self.rollouts_per_task


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_prompt_tokens: int = 4096
    max_response_tokens: int = 256
    train_do_sample: bool = True
    train_temperature: float = 1.0
    train_top_p: float = 1.0
    eval_do_sample: bool = False
    eval_temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class TaskDenominator:
    task_type: str
    count: int


_VALID_SEEN_DENOMINATORS = (
    TaskDenominator("pick_and_place_simple", 35),
    TaskDenominator("pick_two_obj_and_place", 24),
    TaskDenominator("look_at_obj_in_light", 13),
    TaskDenominator("pick_clean_then_place_in_recep", 27),
    TaskDenominator("pick_cool_then_place_in_recep", 25),
    TaskDenominator("pick_heat_then_place_in_recep", 16),
)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    split: str = "valid_seen"
    every_updates: int = 25
    infrastructure_retries: int = 2
    denominators: tuple[TaskDenominator, ...] = _VALID_SEEN_DENOMINATORS

    @property
    def total_tasks(self) -> int:
        return sum(item.count for item in self.denominators)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    mode: SkillMode
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    master_seed: int = 0

    @classmethod
    def formal(cls, *, mode: SkillMode) -> "ExperimentConfig":
        config = cls(mode=mode)
        config.validate()
        return config

    def validate(self) -> None:
        if self.episode.max_steps <= 0:
            raise ValueError("episode.max_steps must be positive")
        if self.episode.history_length < 0:
            raise ValueError("episode.history_length must be non-negative")
        if self.episode.invalid_action_penalty < 0:
            raise ValueError("episode.invalid_action_penalty must be non-negative")
        if self.batch.task_groups_per_update <= 0 or self.batch.rollouts_per_task <= 1:
            raise ValueError("formal GRPO requires positive task groups and rollouts_per_task > 1")
        if self.batch.action_minibatch_size <= 0 or self.batch.max_tokens_per_gpu <= 0:
            raise ValueError("batch limits must be positive")
        if self.generation.max_prompt_tokens <= 0 or self.generation.max_response_tokens <= 0:
            raise ValueError("generation token limits must be positive")
        if not self.generation.train_do_sample or self.generation.train_temperature <= 0:
            raise ValueError("training rollout must use stochastic decoding with positive temperature")
        if self.generation.train_top_p != 1.0:
            raise ValueError("formal training requires train_top_p=1.0")
        if self.generation.eval_do_sample or self.generation.eval_temperature != 0.0:
            raise ValueError("formal evaluation must use deterministic decoding")
        expected = {item.task_type: item.count for item in _VALID_SEEN_DENOMINATORS}
        actual = {item.task_type: item.count for item in self.evaluation.denominators}
        if actual != expected or self.evaluation.total_tasks != 140:
            raise ValueError("valid_seen denominators must match the registered 140-task manifest")
