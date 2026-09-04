from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from infoskill.episode import TrajectoryGroup
from infoskill.rollout import GenerationRequest, GenerationResult

from .hybrid_prefix import build_hybrid_vllm_inputs


@dataclass(frozen=True)
class TokenBatch:
    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    raw_prompt_ids: tuple[tuple[int, ...], ...]


class VerlBatchCodec:
    def __init__(
        self,
        tokenizer: object,
        *,
        max_prompt_tokens: int,
        max_response_tokens: int,
        max_soft_prefix_length: int = 5,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.max_response_tokens = max_response_tokens
        self.max_soft_prefix_length = max_soft_prefix_length
        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(tokenizer, "eos_token_id", None)
        if pad is None:
            raise RuntimeError("policy tokenizer has no pad or EOS token")
        self.pad_token_id = int(pad)

    def encode_prompts(self, requests: Sequence[GenerationRequest]) -> TokenBatch:
        raw = tuple(self._prompt_ids(request.user_message) for request in requests)
        return self._encode_token_ids(raw)

    def generation_dataproto(self, requests: Sequence[GenerationRequest]):
        from verl import DataProto

        text_batch = self.encode_prompts(requests)
        parameters = requests[0].parameters
        if any(request.parameters != parameters for request in requests):
            raise ValueError("one VERL generation batch requires identical sampling parameters")
        vllm_inputs = build_hybrid_vllm_inputs(
            raw_prompt_ids=text_batch.raw_prompt_ids,
            soft_prefixes=tuple(request.soft_prefix for request in requests),
            placeholder_token_id=self.pad_token_id,
        )
        prefix_lengths = [
            int(item["infoskill_prefix_embeds"].shape[0])
            if "infoskill_prefix_embeds" in item
            else 0
            for item in vllm_inputs
        ]
        if max(prefix_lengths, default=0) > self.max_soft_prefix_length:
            raise RuntimeError(
                "soft prefix exceeds configured max_soft_prefix_length; no silent truncation"
            )
        raw_prompt_ids = tuple(
            tuple(int(value) for value in item["prompt_token_ids"])
            for item in vllm_inputs
        )
        batch = self._encode_token_ids(raw_prompt_ids)
        return DataProto.from_dict(
            tensors={
                "input_ids": batch.input_ids,
                "attention_mask": batch.attention_mask,
                "position_ids": batch.position_ids,
            },
            non_tensors={
                "raw_prompt_ids": np.array([list(item) for item in batch.raw_prompt_ids], dtype=object),
                "request_ids": np.array([request.request_id for request in requests], dtype=object),
                "semantic_seeds": np.array([request.seed for request in requests], dtype=np.int64),
                "infoskill_prefix_embeds": _object_array(
                    [item.get("infoskill_prefix_embeds") for item in vllm_inputs]
                ),
                "infoskill_prefix_masks": _object_array(
                    [item.get("infoskill_prefix_mask") for item in vllm_inputs]
                ),
            },
            meta_info={
                "do_sample": parameters.do_sample,
                "validate": not parameters.do_sample,
                "temperature": parameters.temperature,
                "top_p": parameters.top_p,
                "max_tokens": parameters.max_new_tokens,
                "eos_token_id": int(self.tokenizer.eos_token_id),
                "pad_token_id": self.pad_token_id,
            },
        )

    def decode_generation(
        self,
        requests: Sequence[GenerationRequest],
        output: object,
    ) -> tuple[GenerationResult, ...]:
        responses = output.batch["responses"]  # type: ignore[attr-defined]
        attention = output.batch["attention_mask"][:, -responses.shape[-1] :]  # type: ignore[attr-defined]
        rollout_logprobs = output.batch.get("rollout_log_probs")  # type: ignore[attr-defined]
        results = []
        for row, request in enumerate(requests):
            count = int(attention[row].sum().item())
            token_ids = tuple(int(value) for value in responses[row, :count].tolist())
            if rollout_logprobs is None:
                token_logprobs = tuple(0.0 for _ in token_ids)
            else:
                token_logprobs = tuple(float(value) for value in rollout_logprobs[row, :count].tolist())
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            results.append(
                GenerationResult(
                    request_id=request.request_id,
                    text=text,
                    finish_reason=self._finish_reason(text, token_ids, request.parameters.max_new_tokens),
                    token_ids=token_ids,
                    token_logprobs=token_logprobs,
                    prompt_token_count=len(self._prompt_ids(request.user_message)),
                )
            )
        return tuple(results)

    def training_dataproto(
        self,
        groups: Sequence[TrajectoryGroup],
        advantages: Sequence[Sequence[float]],
    ):
        from verl import DataProto

        examples: list[
            tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...], float]
        ] = []
        for group, group_advantages in zip(groups, advantages):
            for trajectory, advantage in zip(group.trajectories, group_advantages):
                for step in trajectory.steps:
                    if step.conditioned_input.soft_prefix is not None:
                        raise RuntimeError(
                            "INFO-SKILL policy recomputation is not installed; refusing prefix-free training"
                        )
                    examples.append(
                        (
                            self._prompt_ids(step.conditioned_input.user_message),
                            step.generation.token_ids,
                            step.generation.token_logprobs,
                            float(advantage),
                        )
                    )
        if not examples:
            raise ValueError("policy update requires at least one generated environment step")
        prompt_width = max(len(prompt) for prompt, _, _, _ in examples)
        response_width = max(len(response) for _, response, _, _ in examples)
        response_width = max(response_width, 1)
        total_width = prompt_width + response_width
        input_ids = torch.full((len(examples), total_width), self.pad_token_id, dtype=torch.long)
        prompts = torch.full((len(examples), prompt_width), self.pad_token_id, dtype=torch.long)
        responses = torch.full((len(examples), response_width), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        advantage_tensor = torch.zeros((len(examples), response_width), dtype=torch.float32)
        rollout_logprobs = torch.zeros(
            (len(examples), response_width), dtype=torch.float32
        )
        for row, (prompt, response, response_logprobs, advantage) in enumerate(examples):
            prompt_start = prompt_width - len(prompt)
            prompt_tensor = torch.tensor(prompt, dtype=torch.long)
            prompts[row, prompt_start:] = prompt_tensor
            input_ids[row, prompt_start:prompt_width] = prompt_tensor
            attention[row, prompt_start:prompt_width] = 1
            if response:
                response_tensor = torch.tensor(response, dtype=torch.long)
                responses[row, : len(response)] = response_tensor
                input_ids[row, prompt_width : prompt_width + len(response)] = response_tensor
                attention[row, prompt_width : prompt_width + len(response)] = 1
                advantage_tensor[row, : len(response)] = advantage
                rollout_logprobs[row, : len(response)] = torch.tensor(
                    response_logprobs, dtype=torch.float32
                )
        positions = (attention.cumsum(dim=-1) - 1).clamp_min(0)
        global_token_num = attention.sum(dim=-1).tolist()
        return DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention,
                "position_ids": positions,
                "advantages": advantage_tensor,
                "rollout_log_probs": rollout_logprobs,
            },
            meta_info={
                "temperature": 1.0,
                "global_token_num": global_token_num,
            },
        )

    def _prompt_ids(self, user_message: str) -> tuple[int, ...]:
        result = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(result, "tolist"):
            result = result.tolist()
        if result and isinstance(result[0], list):
            result = result[0]
        ids = tuple(int(value) for value in result)
        if len(ids) > self.max_prompt_tokens:
            raise RuntimeError(
                f"policy prompt has {len(ids)} tokens, exceeding {self.max_prompt_tokens}; no silent truncation"
            )
        return ids

    def _encode_token_ids(self, raw: tuple[tuple[int, ...], ...]) -> TokenBatch:
        maximum = max(len(item) for item in raw)
        input_ids = torch.full((len(raw), maximum), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for row, item in enumerate(raw):
            input_ids[row, -len(item) :] = torch.tensor(item, dtype=torch.long)
            attention[row, -len(item) :] = 1
        positions = (attention.cumsum(dim=-1) - 1).clamp_min(0)
        return TokenBatch(input_ids, attention, positions, raw)

    def _finish_reason(self, text: str, token_ids: tuple[int, ...], maximum: int) -> str:
        if "</action>" in text:
            return "action_stop"
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if token_ids and eos is not None and token_ids[-1] == int(eos):
            return "eos"
        if len(token_ids) >= maximum:
            return "length"
        return "stop"


def _object_array(values: Sequence[object]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = list(values)
    return result
