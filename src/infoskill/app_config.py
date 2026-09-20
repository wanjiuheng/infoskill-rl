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
    skill_bank: str
    output_root: str
    infoskill_checkpoint: str | None = None
    skill_bank_manifest: str | None = None
    grounding_data: str | None = None
    alfworld_source: str | None = None
    alfworld_data: str | None = None
    alfworld_config: str | None = None
    webshop_source: str | None = None
    webshop_data: str | None = None
    webshop_human_demonstrations: str | None = None
    webshop_human_goals: str | None = None


@dataclass(frozen=True, slots=True)
class AppConfig:
    paths: RuntimePaths
    environment: str = "alfworld"
    policy_model_id: str | None = None
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
        if self.environment not in {"alfworld", "webshop", "search"}:
            raise ValueError("environment must be alfworld, webshop, or search")
        if self.retrieval_mode not in {"embedding", "template"}:
            raise ValueError("retrieval_mode must be embedding or template")
        if min(self.max_steps, self.max_prompt_tokens, self.max_response_tokens, self.eval_batch_size) <= 0:
            raise ValueError("runtime limits must be positive")
        if min(self.general_top_k, self.task_top_k, self.mistake_count) < 0:
            raise ValueError("retrieval counts cannot be negative")
        if self.policy_model_id is not None:
            if not self.policy_model_id.strip():
                raise ValueError("policy_model_id cannot be empty")
            if self.policy_model_id != self.policy_model_id.strip():
                raise ValueError("policy_model_id cannot have surrounding whitespace")
        if self.environment == "alfworld":
            self._require_paths("alfworld_source", "alfworld_data", "alfworld_config")
        if self.environment == "webshop":
            self._require_paths("webshop_source", "webshop_data")

    def _require_paths(self, *names: str) -> None:
        missing = [name for name in names if not getattr(self.paths, name)]
        if missing:
            raise ValueError(
                f"{self.environment} config requires paths: {', '.join(missing)}"
            )

    def as_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        return asdict(self)
