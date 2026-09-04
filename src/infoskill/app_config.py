from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    policy_model: str
    policy_adapter: str | None
    semantic_model: str
    skillrl_source: str
    alfworld_source: str
    alfworld_data: str
    alfworld_config: str
    skill_bank: str
    output_root: str
    infoskill_checkpoint: str | None


@dataclass(frozen=True, slots=True)
class AppConfig:
    paths: RuntimePaths
    max_steps: int = 30
    history_length: int = 2
    max_prompt_tokens: int = 4096
    max_response_tokens: int = 256
    retrieval_mode: str = "embedding"
    general_top_k: int = 6
    task_top_k: int = 6
    mistake_count: int = 5
    master_seed: int = 0
    eval_batch_size: int = 8

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        source = Path(path)
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("configuration loading requires PyYAML") from error
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("paths"), dict):
            raise ValueError("config requires a paths mapping")
        path_values = dict(payload.pop("paths"))
        paths = RuntimePaths(**path_values)
        config = cls(paths=paths, **payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.retrieval_mode not in {"embedding", "template"}:
            raise ValueError("retrieval_mode must be embedding or template")
        if min(self.max_steps, self.max_prompt_tokens, self.max_response_tokens, self.eval_batch_size) <= 0:
            raise ValueError("runtime limits must be positive")
        if min(self.general_top_k, self.task_top_k, self.mistake_count) < 0:
            raise ValueError("retrieval counts cannot be negative")

    def as_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        return asdict(self)
