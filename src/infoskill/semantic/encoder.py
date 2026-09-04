from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from infoskill.skills import SkillRecord


@dataclass(frozen=True)
class FeatureBatch:
    tokens: Tensor
    valid: Tensor

    def pooled_last_token(self) -> Tensor:
        lengths = self.valid.long().sum(dim=-1).clamp_min(1) - 1
        rows = torch.arange(self.tokens.shape[0], device=self.tokens.device)
        return F.normalize(self.tokens[rows, lengths], dim=-1)


class FrozenSemanticEncoder:
    """One local Qwen embedding model shared by retrieval, compression, and grounding."""

    def __init__(self, tokenizer: object, model: object, *, device: torch.device) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model.eval()  # type: ignore[attr-defined]
        for parameter in self.model.parameters():  # type: ignore[attr-defined]
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "FrozenSemanticEncoder":
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("semantic encoder requires transformers") from error
        target = torch.device(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(target)
        return cls(tokenizer, model, device=target)

    @property
    def hidden_size(self) -> int:
        config = self.model.config  # type: ignore[attr-defined]
        width = getattr(config, "hidden_size", None)
        if width is None:
            raise RuntimeError("semantic model config has no hidden_size")
        return int(width)

    @torch.no_grad()
    def encode_tokens(self, texts: Sequence[str], *, max_length: int = 512) -> FeatureBatch:
        if not texts:
            raise ValueError("cannot encode an empty text batch")
        encoded = self.tokenizer(  # type: ignore[operator]
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self.model(**encoded, return_dict=True)  # type: ignore[operator]
        return FeatureBatch(output.last_hidden_state.detach(), encoded["attention_mask"].bool())

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        pooled = self.encode_tokens(texts).pooled_last_token().float().cpu()
        return pooled.tolist()


class SemanticFeatureCache:
    """GPU-local immutable skill cache plus bounded normalized command cache."""

    _KIND_IDS = {"general": 0, "task_specific": 1, "common_mistake": 2}

    def __init__(
        self,
        encoder: FrozenSemanticEncoder,
        *,
        skill_max_length: int = 192,
        command_cache_size: int = 4096,
    ) -> None:
        self.encoder = encoder
        self.skill_max_length = skill_max_length
        self.command_cache_size = command_cache_size
        self._skills: dict[str, tuple[Tensor, Tensor]] = {}
        self._commands: OrderedDict[str, Tensor] = OrderedDict()

    def skill_batch(
        self,
        records: Sequence[SkillRecord],
        *,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not records or batch_size <= 0:
            raise ValueError("skill batch requires records and a positive batch size")
        missing = [record for record in records if record.skill_id not in self._skills]
        if missing:
            features = self.encoder.encode_tokens(
                [record.text for record in missing], max_length=self.skill_max_length
            )
            for row, record in enumerate(missing):
                length = int(features.valid[row].sum().item())
                self._skills[record.skill_id] = (
                    features.tokens[row, :length].detach(),
                    features.valid[row, :length].detach(),
                )
        max_tokens = max(self._skills[record.skill_id][0].shape[0] for record in records)
        width = self._skills[records[0].skill_id][0].shape[-1]
        tokens = torch.zeros(
            (len(records), max_tokens, width),
            device=self.encoder.device,
            dtype=self._skills[records[0].skill_id][0].dtype,
        )
        valid = torch.zeros((len(records), max_tokens), device=self.encoder.device, dtype=torch.bool)
        for index, record in enumerate(records):
            item_tokens, item_valid = self._skills[record.skill_id]
            length = item_tokens.shape[0]
            tokens[index, :length] = item_tokens
            valid[index, :length] = item_valid
        kind_ids = torch.tensor(
            [self._KIND_IDS[record.kind] for record in records],
            device=self.encoder.device,
            dtype=torch.long,
        )
        return (
            tokens.unsqueeze(0).expand(batch_size, -1, -1, -1),
            valid.unsqueeze(0).expand(batch_size, -1, -1),
            kind_ids.unsqueeze(0).expand(batch_size, -1),
        )

    def command_batch(self, command_groups: Sequence[Sequence[str]]) -> tuple[Tensor, Tensor]:
        if not command_groups or any(not group for group in command_groups):
            raise ValueError("each grounding sample requires at least one command")
        normalized_groups = [tuple(_normalize_command(command) for command in group) for group in command_groups]
        missing = list(
            dict.fromkeys(
                command for group in normalized_groups for command in group if command not in self._commands
            )
        )
        if missing:
            features = self.encoder.encode_tokens(missing, max_length=64).pooled_last_token()
            for command, feature in zip(missing, features):
                self._commands[command] = feature.detach()
                self._commands.move_to_end(command)
            while len(self._commands) > self.command_cache_size:
                self._commands.popitem(last=False)
        max_commands = max(len(group) for group in normalized_groups)
        width = next(iter(self._commands.values())).shape[-1]
        embeddings = torch.zeros(
            (len(command_groups), max_commands, width),
            device=self.encoder.device,
            dtype=next(iter(self._commands.values())).dtype,
        )
        valid = torch.zeros((len(command_groups), max_commands), device=self.encoder.device, dtype=torch.bool)
        for row, group in enumerate(normalized_groups):
            for column, command in enumerate(group):
                embeddings[row, column] = self._commands[command]
                valid[row, column] = True
                self._commands.move_to_end(command)
        return embeddings, valid


def _normalize_command(command: str) -> str:
    return " ".join(command.lower().split())
