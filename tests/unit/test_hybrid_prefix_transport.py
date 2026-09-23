from __future__ import annotations

import copy
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from infoskill.integrations.verl.hybrid_prefix import (
    build_hybrid_vllm_inputs,
    clone_sampling_params_with_seeds,
    fingerprint_vllm_generation_inputs,
    temporary_sampling_overrides,
)
from infoskill.integrations.verl.hybrid_rollout import (
    _HybridInferenceEngine,
    _extract_teacher_forced_response_logprobs,
    _hybrid_cuda_graph_environment,
    vllm_action_stop_settings,
)


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

    def view(self, _dtype: object):
        class FakeBytes:
            def numpy(self):
                return self

            def tobytes(self):
                return b"fixed-prefix"

        return FakeBytes()


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
    def test_teacher_forced_extraction_selects_response_prompt_logprobs(self) -> None:
        token_ids = (101, 102, 201, 202)
        output = SimpleNamespace(
            prompt_logprobs=[
                None,
                {102: SimpleNamespace(logprob=-0.2)},
                {201: SimpleNamespace(logprob=-1.1)},
                {202: SimpleNamespace(logprob=-2.2)},
            ]
        )

        values = _extract_teacher_forced_response_logprobs(
            outputs=(output,),
            prompt_token_ids=(token_ids,),
            response_lengths=(2,),
            response_width=3,
        )

        self.assertEqual(values, [[-1.1, -2.2, 0.0]])

    def test_teacher_forced_extraction_rejects_missing_target_token(self) -> None:
        output = SimpleNamespace(
            prompt_logprobs=[None, {102: SimpleNamespace(logprob=-0.2)}]
        )
        with self.assertRaisesRegex(RuntimeError, "target token"):
            _extract_teacher_forced_response_logprobs(
                outputs=(output,),
                prompt_token_ids=((101, 999),),
                response_lengths=(1,),
                response_width=1,
            )

    def test_cuda_graph_restores_custom_kernels_after_vllm_defaults(self) -> None:
        class FakeVllmConfig:
            def __init__(self) -> None:
                self.compilation_config = types.SimpleNamespace()
                self.__post_init__()

            def __post_init__(self) -> None:
                self.compilation_config.use_cudagraph = True
                self.compilation_config.use_inductor = True
                self.compilation_config.custom_ops = ["none"]

        config_module = types.ModuleType("vllm.config")
        config_module.VllmConfig = FakeVllmConfig
        vllm_module = types.ModuleType("vllm")
        vllm_module.config = config_module

        with patch.dict(
            sys.modules,
            {"vllm": vllm_module, "vllm.config": config_module},
        ):
            with _hybrid_cuda_graph_environment(True):
                configured = FakeVllmConfig()
                self.assertTrue(configured.compilation_config.use_cudagraph)
                self.assertFalse(configured.compilation_config.use_inductor)
                self.assertEqual(
                    configured.compilation_config.custom_ops,
                    ["all"],
                )

            restored = FakeVllmConfig()
            self.assertTrue(restored.compilation_config.use_inductor)
            self.assertEqual(restored.compilation_config.custom_ops, ["none"])

    def test_cuda_graph_environment_is_scoped_and_restored(self) -> None:
        variable = "VLLM_INFOSKILL_HYBRID_PREFIX_CUDA_GRAPH"
        class FakeVllmConfig:
            def __post_init__(self) -> None:
                pass

        config_module = types.ModuleType("vllm.config")
        config_module.VllmConfig = FakeVllmConfig
        vllm_module = types.ModuleType("vllm")
        vllm_module.config = config_module
        with patch.dict(
            sys.modules,
            {"vllm": vllm_module, "vllm.config": config_module},
        ):
            with patch.dict(os.environ, {variable: "existing"}, clear=False):
                with _hybrid_cuda_graph_environment(True):
                    self.assertEqual(os.environ[variable], "1")
                self.assertEqual(os.environ[variable], "existing")

        with patch.dict(os.environ, {}, clear=True):
            with _hybrid_cuda_graph_environment(False):
                self.assertNotIn(variable, os.environ)
            self.assertNotIn(variable, os.environ)

    def test_action_stop_uses_native_string_and_enables_detokenization(self) -> None:
        settings = vllm_action_stop_settings("</action>")

        self.assertEqual(settings["stop"], "</action>")
        self.assertIsInstance(settings["stop"], str)
        self.assertTrue(settings["detokenize"])
        self.assertTrue(settings["include_stop_str_in_output"])

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

        with patch.dict(os.environ, {"INFOSKILL_VLLM_INPUT_AUDIT": "1"}):
            with patch(
                "infoskill.integrations.verl.hybrid_rollout.fingerprint_vllm_generation_inputs",
                return_value={"batch_sha256": "fake"},
            ):
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
        self.assertEqual(proxy.input_fingerprints[0]["batch_sha256"], "fake")

    def test_actual_vllm_input_digest_changes_with_seed_and_prefix(self) -> None:
        params = _SamplingParams(seed=7)
        plain = fingerprint_vllm_generation_inputs(
            [{"prompt_token_ids": [3, 4]}], [params]
        )
        seeded = fingerprint_vllm_generation_inputs(
            [{"prompt_token_ids": [3, 4]}], [_SamplingParams(seed=8)]
        )
        prefixed = fingerprint_vllm_generation_inputs(
            [{"prompt_token_ids": [3, 4],
              "infoskill_prefix_embeds": b"prefix",
              "infoskill_prefix_mask": [True, False]}],
            [params],
        )
        self.assertNotEqual(plain["batch_sha256"], seeded["batch_sha256"])
        self.assertNotEqual(plain["batch_sha256"], prefixed["batch_sha256"])

    def test_input_audit_is_off_on_normal_training_and_evaluation(self) -> None:
        proxy = _HybridInferenceEngine(
            _Engine(),
            prefix_embeds=(None,),
            prefix_masks=(None,),
            semantic_seeds=(7,),
        )
        with patch.dict(os.environ, {}, clear=True):
            proxy.generate(
                prompts=[{"prompt_token_ids": [3, 4]}],
                sampling_params=_SamplingParams(),
            )
        self.assertEqual(proxy.input_fingerprints, [])

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
