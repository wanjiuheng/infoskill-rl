from __future__ import annotations

import copy
import unittest

from infoskill.integrations.verl.hybrid_prefix import (
    build_hybrid_vllm_inputs,
    clone_sampling_params_with_seeds,
    temporary_sampling_overrides,
)
from infoskill.integrations.verl.hybrid_rollout import _HybridInferenceEngine


class _FakeTensor:
    def __init__(self, rows: int, width: int) -> None:
        self.ndim = 2
        self.shape = (rows, width)
        self.detached = False
        self.cpu = False
        self.contiguous_value = False

    def detach(self) -> "_FakeTensor":
        result = copy.copy(self)
        result.detached = True
        return result

    def to(self, device: str) -> "_FakeTensor":
        result = copy.copy(self)
        result.cpu = device == "cpu"
        return result

    def contiguous(self) -> "_FakeTensor":
        result = copy.copy(self)
        result.contiguous_value = True
        return result


class _SamplingParams:
    def __init__(
        self,
        seed: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
    ) -> None:
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def clone(self) -> "_SamplingParams":
        return copy.copy(self)


class _Engine:
    def __init__(self) -> None:
        self.call = None

    def generate(self, **kwargs):
        self.call = kwargs
        return ("ok",)


class HybridPrefixTransportTests(unittest.TestCase):
    def test_prefix_rows_become_explicit_leading_placeholder_positions(self) -> None:
        prefix = _FakeTensor(rows=5, width=3584)

        inputs = build_hybrid_vllm_inputs(
            raw_prompt_ids=([10, 11], [20]),
            soft_prefixes=(prefix, None),
            placeholder_token_id=0,
        )

        self.assertEqual(inputs[0]["prompt_token_ids"], [0, 0, 0, 0, 0, 10, 11])
        self.assertEqual(inputs[0]["infoskill_prefix_mask"], [True] * 5 + [False, False])
        transported = inputs[0]["infoskill_prefix_embeds"]
        self.assertTrue(transported.detached)
        self.assertTrue(transported.cpu)
        self.assertTrue(transported.contiguous_value)
        self.assertEqual(inputs[1], {"prompt_token_ids": [20]})

    def test_prefix_batch_size_must_match_prompt_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch size"):
            build_hybrid_vllm_inputs(
                raw_prompt_ids=([10], [20]),
                soft_prefixes=(_FakeTensor(5, 8),),
                placeholder_token_id=0,
            )

    def test_sampling_params_are_cloned_and_seeded_per_request(self) -> None:
        base = _SamplingParams(seed=999)

        actual = clone_sampling_params_with_seeds(base, (3, 7))

        self.assertEqual([item.seed for item in actual], [3, 7])
        self.assertEqual(base.seed, 999)
        self.assertIsNot(actual[0], base)
        self.assertIsNot(actual[0], actual[1])

    def test_engine_proxy_injects_prefix_and_per_request_sampling(self) -> None:
        engine = _Engine()
        prefix = _FakeTensor(rows=2, width=8)
        proxy = _HybridInferenceEngine(
            engine,
            prefix_embeds=(prefix, None),
            prefix_masks=((True, True, False), None),
            semantic_seeds=(13, 17),
        )

        output = proxy.generate(
            prompts=[{"prompt_token_ids": [0, 0, 9]}, {"prompt_token_ids": [8]}],
            sampling_params=_SamplingParams(seed=999),
            use_tqdm=False,
        )

        self.assertEqual(output, ("ok",))
        self.assertIs(engine.call["prompts"][0]["infoskill_prefix_embeds"], prefix)
        self.assertEqual(
            engine.call["prompts"][0]["infoskill_prefix_mask"],
            [True, True, False],
        )
        self.assertNotIn("infoskill_prefix_embeds", engine.call["prompts"][1])
        self.assertEqual(
            [item.seed for item in engine.call["sampling_params"]],
            [13, 17],
        )

    def test_sampling_overrides_apply_requested_values_and_restore(self) -> None:
        params = _SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256)

        with temporary_sampling_overrides(
            params,
            temperature=0.4,
            top_p=0.9,
            max_tokens=128,
            response_cap=256,
        ):
            self.assertEqual(
                (params.temperature, params.top_p, params.max_tokens),
                (0.4, 0.9, 128),
            )

        self.assertEqual(
            (params.temperature, params.top_p, params.max_tokens),
            (1.0, 1.0, 256),
        )

    def test_sampling_overrides_reject_response_above_runtime_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "response cap"):
            with temporary_sampling_overrides(
                _SamplingParams(),
                temperature=1.0,
                top_p=1.0,
                max_tokens=257,
                response_cap=256,
            ):
                pass


if __name__ == "__main__":
    unittest.main()
