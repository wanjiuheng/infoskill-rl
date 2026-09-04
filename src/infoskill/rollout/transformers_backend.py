from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch
from torch import Tensor

from .contracts import GenerationRequest, GenerationResult


class TransformersBackend:
    """Slow correctness backend; loads local weights and never calls an API."""

    def __init__(
        self,
        *,
        tokenizer: object,
        model: object,
        max_prompt_tokens: int = 4096,
        action_end_text: str = "</action>",
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.max_prompt_tokens = max_prompt_tokens
        self.action_end_text = action_end_text
        self.model.eval()  # type: ignore[attr-defined]

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        adapter_path: str | None = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        max_prompt_tokens: int = 4096,
    ) -> "TransformersBackend":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Transformers backend requires transformers") from error
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(torch.device(device))
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as error:
                raise RuntimeError("loading a LoRA adapter requires peft") from error
            model = PeftModel.from_pretrained(model, adapter_path).to(torch.device(device))
        return cls(tokenizer=tokenizer, model=model, max_prompt_tokens=max_prompt_tokens)

    @torch.no_grad()
    def generate(self, requests: tuple[GenerationRequest, ...]) -> tuple[GenerationResult, ...]:
        return tuple(self._generate_one(request) for request in requests)

    def _generate_one(self, request: GenerationRequest) -> GenerationResult:
        prompt_ids = self.tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            [{"role": "user", "content": request.user_message}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_tokens = int(prompt_ids.shape[-1])
        if prompt_tokens > self.max_prompt_tokens:
            raise RuntimeError(
                f"prompt exceeds max_prompt_tokens ({prompt_tokens}>{self.max_prompt_tokens}) "
                f"for request {request.request_id}"
            )
        device = next(self.model.parameters()).device  # type: ignore[attr-defined]
        prompt_ids = prompt_ids.to(device)
        generation_input = self._generation_input(prompt_ids, request.soft_prefix)
        parameters = request.parameters
        kwargs = {
            "max_new_tokens": parameters.max_new_tokens,
            "do_sample": parameters.do_sample,
            "top_p": parameters.top_p,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self._pad_token_id(),
            "eos_token_id": self.tokenizer.eos_token_id,  # type: ignore[attr-defined]
            "stopping_criteria": self._stopping_criteria(device),
        }
        if parameters.do_sample:
            if parameters.temperature <= 0:
                raise ValueError("sampled generation requires positive temperature")
            kwargs["temperature"] = parameters.temperature

        cuda_devices = [device.index or 0] if device.type == "cuda" else []
        rng_context = torch.random.fork_rng(devices=cuda_devices)
        with rng_context:
            torch.manual_seed(request.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(request.seed)
            output = self.model.generate(**generation_input, **kwargs)  # type: ignore[attr-defined]
        generated_count = len(output.scores)
        generated = output.sequences[0, -generated_count:] if generated_count else output.sequences[0, :0]
        token_ids = tuple(int(value) for value in generated.tolist())
        logprobs = tuple(
            float(torch.log_softmax(score[0].float(), dim=-1)[token].item())
            for score, token in zip(output.scores, token_ids)
        )
        text = self.tokenizer.decode(generated, skip_special_tokens=True)  # type: ignore[attr-defined]
        finish_reason = self._finish_reason(text, token_ids, generated_count, parameters.max_new_tokens)
        return GenerationResult(
            request_id=request.request_id,
            text=text,
            finish_reason=finish_reason,
            token_ids=token_ids,
            token_logprobs=logprobs,
            prompt_token_count=prompt_tokens,
        )

    def _generation_input(self, prompt_ids: Tensor, soft_prefix: object | None) -> dict[str, Tensor]:
        attention = torch.ones_like(prompt_ids)
        if soft_prefix is None:
            return {"input_ids": prompt_ids, "attention_mask": attention}
        if not isinstance(soft_prefix, Tensor) or soft_prefix.ndim != 2:
            raise TypeError("soft_prefix must be a [prefix_length, hidden_size] torch.Tensor")
        token_embeddings = self.model.get_input_embeddings()(prompt_ids)  # type: ignore[attr-defined]
        prefix = soft_prefix.to(device=token_embeddings.device, dtype=token_embeddings.dtype).unsqueeze(0)
        if prefix.shape[-1] != token_embeddings.shape[-1]:
            raise ValueError("soft prefix width does not match policy hidden size")
        combined = torch.cat((prefix, token_embeddings), dim=1)
        prefix_attention = torch.ones(
            (1, prefix.shape[1]), device=attention.device, dtype=attention.dtype
        )
        return {"inputs_embeds": combined, "attention_mask": torch.cat((prefix_attention, attention), dim=1)}

    def _stopping_criteria(self, device: torch.device) -> object:
        try:
            from transformers import StoppingCriteriaList
        except ImportError as error:
            raise RuntimeError("Transformers backend requires transformers") from error
        stop_ids = self.tokenizer.encode(self.action_end_text, add_special_tokens=False)  # type: ignore[attr-defined]
        return StoppingCriteriaList([_TokenSuffixStop(stop_ids, device=device)])

    def _pad_token_id(self) -> int:
        value = getattr(self.tokenizer, "pad_token_id", None)
        if value is None:
            value = getattr(self.tokenizer, "eos_token_id", None)
        if value is None:
            raise RuntimeError("policy tokenizer has neither pad_token_id nor eos_token_id")
        return int(value)

    def _finish_reason(
        self,
        text: str,
        token_ids: tuple[int, ...],
        generated_count: int,
        maximum: int,
    ) -> str:
        if self.action_end_text in text:
            return "action_stop"
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if token_ids and eos is not None and token_ids[-1] == int(eos):
            return "eos"
        if generated_count >= maximum:
            return "length"
        return "stop"


class _TokenSuffixStop:
    def __init__(self, token_ids: Iterable[int], *, device: torch.device) -> None:
        from transformers import StoppingCriteria

        self._delegate_type = StoppingCriteria
        self.stop = torch.tensor(tuple(token_ids), device=device, dtype=torch.long)
        if self.stop.numel() == 0:
            raise ValueError("action stop text tokenized to an empty sequence")

    def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs: object) -> bool:
        del scores, kwargs
        return bool(
            input_ids.shape[-1] >= self.stop.numel()
            and torch.equal(input_ids[0, -self.stop.numel() :], self.stop)
        )
