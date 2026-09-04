from __future__ import annotations

from typing import Any

from .hybrid_prefix import (
    clone_sampling_params_with_seeds,
    temporary_sampling_overrides,
)


def vllm_action_stop_settings(action_end_text: str) -> dict[str, object]:
    """Return vLLM 0.8.4-compatible settings for a textual action stop."""
    if not action_end_text:
        raise ValueError("action stop text cannot be empty")
    return {
        # A Python list assigned to DictConfig becomes OmegaConf ListConfig,
        # which vLLM 0.8.4 rejects via an exact isinstance(..., list) check.
        # A scalar string remains native and SamplingParams converts it itself.
        "stop": action_end_text,
        "detokenize": True,
        "include_stop_str_in_output": True,
    }


class _HybridInferenceEngine:
    def __init__(
        self,
        engine: object,
        *,
        prefix_embeds: tuple[object | None, ...],
        prefix_masks: tuple[object | None, ...],
        semantic_seeds: tuple[int, ...],
    ) -> None:
        self._engine = engine
        self._prefix_embeds = prefix_embeds
        self._prefix_masks = prefix_masks
        self._semantic_seeds = semantic_seeds

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def generate(self, *, prompts: list[dict[str, Any]], sampling_params: object, **kwargs: Any):
        batch_size = len(prompts)
        if not (
            len(self._prefix_embeds)
            == len(self._prefix_masks)
            == len(self._semantic_seeds)
            == batch_size
        ):
            raise RuntimeError("hybrid-prefix metadata batch size drifted before vLLM generation")

        enriched = []
        for prompt, prefix, mask in zip(prompts, self._prefix_embeds, self._prefix_masks):
            item = dict(prompt)
            if (prefix is None) != (mask is None):
                raise RuntimeError("hybrid-prefix embeds and mask must be present together")
            if prefix is not None:
                item["infoskill_prefix_embeds"] = prefix
                item["infoskill_prefix_mask"] = list(mask)  # type: ignore[arg-type]
            enriched.append(item)
        per_request_params = clone_sampling_params_with_seeds(
            sampling_params, self._semantic_seeds
        )
        return self._engine.generate(
            prompts=enriched,
            sampling_params=per_request_params,
            **kwargs,
        )


def hybrid_vllm_rollout_class():
    """Create the subclass lazily so importing INFO-SKILL stays dependency-light."""
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

    class HybridPrefixVLLMRollout(vLLMRollout):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            config = kwargs.get("config")
            require_hybrid = bool(
                config is not None
                and config.get("infoskill_hybrid_prefix", False)
            )
            if require_hybrid:
                from vllm.inputs.data import INFOSKILL_HYBRID_PREFIX_API

                if INFOSKILL_HYBRID_PREFIX_API != 1:
                    raise RuntimeError("unsupported INFO-SKILL Hybrid Prefix API")

            if not require_hybrid:
                super().__init__(*args, **kwargs)
                return

            # SkillRL 8e66726 hard-codes prefix caching. Replace only its local
            # constructor binding while this pinned rollout instance is built.
            from verl.workers.rollout.vllm_rollout import vllm_rollout_spmd

            original_llm = vllm_rollout_spmd.LLM

            def no_prefix_cache_llm(*llm_args: Any, **llm_kwargs: Any):
                llm_kwargs["enable_prefix_caching"] = False
                llm_kwargs["enforce_eager"] = True
                return original_llm(*llm_args, **llm_kwargs)

            vllm_rollout_spmd.LLM = no_prefix_cache_llm
            try:
                super().__init__(*args, **kwargs)
            finally:
                vllm_rollout_spmd.LLM = original_llm

        def generate_sequences(self, prompts: object, **kwargs: Any):
            non_tensors = prompts.non_tensor_batch  # type: ignore[attr-defined]
            batch_size = int(prompts.batch["input_ids"].size(0))  # type: ignore[attr-defined]
            prefixes = tuple(
                non_tensors.pop(
                    "infoskill_prefix_embeds", [None] * batch_size
                )
            )
            masks = tuple(
                non_tensors.pop("infoskill_prefix_masks", [None] * batch_size)
            )
            seeds = tuple(
                int(value)
                for value in non_tensors.pop("semantic_seeds", [0] * batch_size)
            )
            engine = self.inference_engine
            self.inference_engine = _HybridInferenceEngine(
                engine,
                prefix_embeds=prefixes,
                prefix_masks=masks,
                semantic_seeds=seeds,
            )
            try:
                meta = prompts.meta_info  # type: ignore[attr-defined]
                with temporary_sampling_overrides(
                    self.sampling_params,
                    temperature=float(meta.get("temperature", 1.0)),
                    top_p=float(meta.get("top_p", 1.0)),
                    max_tokens=int(meta.get("max_tokens", self.config.response_length)),
                    response_cap=int(self.config.response_length),
                ):
                    return super().generate_sequences(prompts, **kwargs)
            finally:
                self.inference_engine = engine

    return HybridPrefixVLLMRollout
