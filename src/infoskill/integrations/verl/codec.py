from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from infoskill.episode import TrajectoryGroup
from infoskill.rollout import GenerationRequest, GenerationResult, PromptLengthError

from .generation_boundary import trim_vllm_padding_sentinel
from .hybrid_prefix import build_hybrid_vllm_inputs
from .object_array import object_array
from .policy_replay import PolicyReplayExample, build_policy_replay_tensors


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
        raw = tuple(
            self._prompt_ids(
                request.user_message,
                request_id=request.request_id,
            )
            for request in requests
        )
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
                "infoskill_prefix_embeds": object_array(
                    [item.get("infoskill_prefix_embeds") for item in vllm_inputs]
                ),
                "infoskill_prefix_masks": object_array(
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
                token_ids, token_logprobs = trim_vllm_padding_sentinel(
                    token_ids=token_ids,
                    token_logprobs=token_logprobs,
                    pad_token_id=self.pad_token_id,
                )
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

        examples: list[PolicyReplayExample] = []
        row_metadata: list[dict[str, object]] = []
        for group_index, (group, group_advantages) in enumerate(
            zip(groups, advantages)
        ):
            for trajectory_index, (trajectory, advantage) in enumerate(
                zip(group.trajectories, group_advantages)
            ):
                for step_index, step in enumerate(trajectory.steps):
                    trace = step.conditioned_input.conditioning_trace
                    latent = getattr(trace, "latent", None)
                    prompt = step.conditioned_input.user_message
                    examples.append(
                        PolicyReplayExample(
                            prompt_ids=self._prompt_ids(prompt),
                            response_ids=step.generation.token_ids,
                            response_logprobs=step.generation.token_logprobs,
                            advantage=float(advantage),
                            latent=latent,
                            rollout_prefix=step.conditioned_input.soft_prefix,
                        )
                    )
                    row_metadata.append(
                        {
                            "original_sample_index": len(examples) - 1,
                            "group_index": group_index,
                            "trajectory_index": trajectory_index,
                            "step_index": step_index,
                            "task_id": trajectory.task.task_id,
                            "task_type": trajectory.task.task_type,
                            "environment_path": trajectory.task.environment_path,
                            "rollout_id": trajectory.rollout_id,
                            "generation_request_id": step.generation.request_id,
                            "prompt_sha256": hashlib.sha256(
                                prompt.encode("utf-8")
                            ).hexdigest(),
                            "prompt": prompt,
                            "prompt_token_count": step.generation.prompt_token_count,
                            "generated_text": step.generation.text,
                            "response_token_count": len(step.generation.token_ids),
                            "executed_action": step.action.executed_action,
                            "resolved_action": step.action.resolved_action,
                            "candidate_skill_ids": list(
                                step.conditioned_input.candidate_skill_ids
                            ),
                        }
                    )
        tensors = build_policy_replay_tensors(
            examples,
            pad_token_id=self.pad_token_id,
        )
        global_token_num = tensors["attention_mask"].sum(dim=-1).tolist()
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors={
                "infoskill_replay_metadata": object_array(row_metadata),
            },
            meta_info={
                "temperature": 1.0,
                "global_token_num": global_token_num,
            },
        )

    def _prompt_ids(
        self,
        user_message: str,
        *,
        request_id: str = "unknown",
    ) -> tuple[int, ...]:
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
            raise PromptLengthError(
                request_id=request_id,
                token_count=len(ids),
                max_prompt_tokens=self.max_prompt_tokens,
                user_message=user_message,
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
