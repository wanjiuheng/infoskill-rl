import unittest
from unittest.mock import patch

from infoskill.m1_lora_layer_localization import (
    VllmLoraKernelIntervention,
    VllmLoraPreCaptureSplitKOneIntervention,
    VllmLayerCapture,
    _aggregate_boundary_comparisons,
    _reference_expand_delta,
    _reference_shrink,
    classify_kernel_causality,
    classify_layer_localization,
    compare_layer_rounds,
    compare_lora_rounds,
)


def trace(layer0="a", layer1="b"):
    return (
        {"rank": 0, "module": "model.layers.0", "call": 0,
         "output": [{"sha256": layer0}]},
        {"rank": 0, "module": "model.layers.1", "call": 0,
         "output": [{"sha256": layer1}]},
    )


class LayerLocalizationTests(unittest.TestCase):
    def test_boundary_aggregate_preserves_first_changed_seam(self):
        exact = {
            "comparable_rows": 3,
            "returned_logprob_comparable_rows": 3,
            "changed_row_counts": {
                "final_hidden": 0,
                "raw_logits": 0,
                "processed_logits": 0,
                "sampled_token": 0,
                "returned_logprob": 0,
            },
        }
        changed = {
            **exact,
            "changed_row_counts": {
                **exact["changed_row_counts"],
                "final_hidden": 2,
                "raw_logits": 2,
            },
        }

        report = _aggregate_boundary_comparisons((exact, changed))

        self.assertEqual(report["comparable_rows"], 6)
        self.assertEqual(report["first_changed_boundary"], "final_hidden")

    def test_finds_first_changed_decoder_layer(self):
        report = compare_layer_rounds((trace(), trace(layer1="changed")))
        self.assertEqual(report["first_changed_layer"], 1)
        self.assertEqual(report["first_changed_layer_by_rank"], {"0": 1})
        self.assertEqual(report["changed_modules"], ["model.layers.1"])

    def test_stable_rounds_have_no_changed_layer(self):
        report = compare_layer_rounds((trace(), trace()))
        self.assertIsNone(report["first_changed_layer"])
        self.assertEqual(report["comparable_records"], 2)

    def test_observer_effect_blocks_kernel_attribution(self):
        report = classify_layer_localization(
            checkpoint_control={"first_changed_boundary": "raw_logits"},
            checkpoint_instrumented={"first_changed_boundary": "exact_at_captured_boundaries"},
            base_instrumented={"first_changed_boundary": "exact_at_captured_boundaries"},
            layer_comparison={"first_changed_layer": None},
            lora_comparison={"first_changed_stage": None},
            lora_disabled={"first_changed_boundary": "exact_at_captured_boundaries"},
        )
        self.assertEqual(report, "observer_effect_inconclusive")

    def test_lora_delta_is_attributed_only_with_clean_controls(self):
        report = classify_layer_localization(
            checkpoint_control={"first_changed_boundary": "raw_logits"},
            checkpoint_instrumented={"first_changed_boundary": "raw_logits"},
            base_instrumented={"first_changed_boundary": "exact_at_captured_boundaries"},
            layer_comparison={"first_changed_layer": 3},
            lora_comparison={"first_changed_stage": "lora_expand_delta"},
            lora_disabled={"first_changed_boundary": "exact_at_captured_boundaries"},
        )
        self.assertEqual(report, "lora_expand_delta_drift")

    def test_lora_stage_comparison_finds_first_changed_stage(self):
        baseline = ({
            "rank": 0,
            "module": "model.layers.3.self_attn.qkv_proj",
            "call": 0,
            "input": [{"sha256": "input"}],
            "base_output": [{"sha256": "base"}],
            "lora_shrink": [{"sha256": "shrink"}],
            "lora_expand_delta": [{"sha256": "delta"}],
            "combined_output": [{"sha256": "combined"}],
        },)
        changed = (dict(baseline[0], lora_shrink=[{"sha256": "changed"}]),)
        report = compare_lora_rounds((baseline, changed))
        self.assertEqual(report["first_changed_stage"], "lora_shrink_active")

    def test_lora_stage_comparison_uses_execution_order_not_global_stage_order(self):
        first = (
            {
                "rank": 1,
                "module": "model.layers.0.mlp.gate_up_proj",
                "call": 0,
                "input": [{"sha256": "gate-input"}],
                "base_output": [{"sha256": "gate-base"}],
                "lora_shrink_active": [{"sha256": "gate-shrink"}],
                "lora_expand_delta": [{"sha256": "gate-delta"}],
                "combined_output": [{"sha256": "gate-combined"}],
            },
            {
                "rank": 1,
                "module": "model.layers.0.mlp.down_proj",
                "call": 0,
                "input": [{"sha256": "down-input"}],
                "base_output": [{"sha256": "down-base"}],
                "lora_shrink_active": [{"sha256": "down-shrink"}],
                "lora_expand_delta": [{"sha256": "down-delta"}],
                "combined_output": [{"sha256": "down-combined"}],
            },
        )
        second = (
            dict(
                first[0],
                lora_expand_delta=[{"sha256": "gate-delta-changed"}],
                combined_output=[{"sha256": "gate-combined-changed"}],
            ),
            dict(first[1], input=[{"sha256": "down-input-changed"}]),
        )

        report = compare_lora_rounds((first, second))

        self.assertEqual(report["first_changed_stage"], "lora_expand_delta")
        self.assertEqual(
            report["first_changed_event"],
            {
                "rank": 1,
                "module": "model.layers.0.mlp.gate_up_proj",
                "call": 0,
                "stage": "lora_expand_delta",
            },
        )

    def test_kernel_causality_attributes_shrink_only_after_intervention(self):
        exact = {"first_changed_boundary": "exact_at_captured_boundaries"}
        drift = {"first_changed_boundary": "final_hidden"}

        report = classify_kernel_causality(
            native=drift,
            native_split_k_one=exact,
            reference_shrink=exact,
            reference_expand=drift,
            reference_full=exact,
            rotation=drift,
        )

        self.assertEqual(
            report,
            "native_shrink_split_k_atomic_nondeterminism",
        )

    def test_kernel_causality_does_not_blame_split_k_when_split_one_drifts(self):
        exact = {"first_changed_boundary": "exact_at_captured_boundaries"}
        drift = {"first_changed_boundary": "final_hidden"}

        report = classify_kernel_causality(
            native=drift,
            native_split_k_one=drift,
            reference_shrink=exact,
            reference_expand=drift,
            reference_full=exact,
            rotation=drift,
        )

        self.assertEqual(
            report,
            "native_shrink_nondeterminism_not_eliminated_by_split_k_one",
        )

    def test_kernel_causality_remains_inconclusive_when_full_reference_drifts(self):
        drift = {"first_changed_boundary": "final_hidden"}

        report = classify_kernel_causality(
            native=drift,
            native_split_k_one=drift,
            reference_shrink=drift,
            reference_expand=drift,
            reference_full=drift,
            rotation=drift,
        )

        self.assertEqual(report, "drift_persists_with_full_reference_lora")

    def test_split_k_one_intervention_is_scoped_and_restores_wrapper(self):
        calls = []

        class Wrapper:
            def add_shrink(self, y, x, weights, scale, **kwargs):
                calls.append(("original", y, x, weights, scale, kwargs))

        class Lora:
            def __init__(self):
                self.punica_wrapper = Wrapper()

        class Model:
            def __init__(self):
                self.lora = Lora()

            def named_modules(self):
                return iter((("model.layers.0.self_attn.qkv_proj", self.lora),))

        class Runner:
            def __init__(self):
                self.model = Model()

        runner = Runner()
        wrapper = runner.model.lora.punica_wrapper
        original = wrapper.add_shrink
        intervention = VllmLoraKernelIntervention(
            runner,
            "native_split_k_one",
        )
        with patch(
            "infoskill.m1_lora_layer_localization._native_shrink_with_split_k",
            return_value=True,
        ) as replacement:
            intervention.install()
            wrapper.add_shrink("y", "x", "weights", 0.5, marker="value")
            replacement.assert_called_once_with(
                wrapper,
                "y",
                "x",
                "weights",
                0.5,
                split_k=1,
            )
            intervention.remove()

        wrapper.add_shrink("y2", "x2", "weights2", 1.0, marker="restored")
        self.assertEqual(calls[0][0], "original")
        self.assertEqual(calls[0][-1], {"marker": "restored"})
        self.assertEqual(wrapper.add_shrink.__func__, original.__func__)

    def test_precapture_split_k_one_patches_class_counts_and_restores(self):
        calls = []

        class Wrapper:
            def add_shrink(self, y, x, weights, scale, **kwargs):
                calls.append(("original", y, x, weights, scale, kwargs))

        original = Wrapper.add_shrink
        intervention = VllmLoraPreCaptureSplitKOneIntervention(Wrapper)
        with patch(
            "infoskill.m1_lora_layer_localization._native_shrink_with_split_k",
            return_value=True,
        ) as replacement:
            intervention.install()
            wrapper = Wrapper()
            wrapper.add_shrink("y", "x", "weights", 0.5, marker="ignored")
            replacement.assert_called_once_with(
                wrapper,
                "y",
                "x",
                "weights",
                0.5,
                split_k=1,
            )
            self.assertEqual(intervention.call_count, 1)
            self.assertEqual(intervention.kernel_launch_count, 1)
            intervention.remove()

        self.assertIs(Wrapper.add_shrink, original)
        Wrapper().add_shrink("y2", "x2", "weights2", 1.0, marker="restored")
        self.assertEqual(calls[0][-1], {"marker": "restored"})

    def test_lora_attribution_is_rejected_when_disabled_path_drifts(self):
        report = classify_layer_localization(
            checkpoint_control={"first_changed_boundary": "raw_logits"},
            checkpoint_instrumented={"first_changed_boundary": "raw_logits"},
            base_instrumented={
                "first_changed_boundary": "exact_at_captured_boundaries"
            },
            layer_comparison={"first_changed_layer": 3},
            lora_comparison={"first_changed_stage": "lora_expand_delta"},
            lora_disabled={"first_changed_boundary": "final_hidden"},
        )
        self.assertEqual(report, "drift_persists_without_lora_request")

    def test_scoped_lora_hooks_capture_stages_and_restore_methods(self):
        class FakeTensor:
            def detach(self): return self
            def clone(self): return FakeTensor()
            def __sub__(self, _other): return FakeTensor()

        class Handle:
            removed = False
            def remove(self): self.removed = True

        class Layer:
            def __init__(self): self.hook = None
            def register_forward_hook(self, hook):
                self.hook = hook
                return Handle()

        class Wrapper:
            def add_shrink(self, y, _x, *_args, **_kwargs): return y
            def add_expand(self, y, _x, *_args, **_kwargs): return y

        class Lora:
            def __init__(self): self.punica_wrapper = Wrapper()
            def apply(self, x, _bias=None):
                output = FakeTensor()
                low_rank = FakeTensor()
                self.punica_wrapper.add_shrink(low_rank, x)
                self.punica_wrapper.add_expand(output, low_rank)
                return output

        class Model:
            def __init__(self):
                self.layer = Layer()
                self.lora = Lora()
            def named_modules(self):
                return iter((
                    ("model.layers.0", self.layer),
                    ("model.layers.0.self_attn.qkv_proj", self.lora),
                ))

        class Runner:
            def __init__(self): self.model = Model()

        runner = Runner()
        original_apply = runner.model.lora.apply
        capture = VllmLayerCapture(runner, rank=0, layer=0)
        with patch(
            "infoskill.m1_lora_layer_localization._tensor_outputs",
            return_value=[{"sha256": "captured"}],
        ):
            capture.install()
            runner.model.lora.apply(FakeTensor())
            rows = capture.take()
            capture.remove()
        lora_row = next(row for row in rows if "qkv_proj" in row["module"])
        for stage in (
            "input", "base_output", "lora_shrink",
            "lora_expand_delta", "combined_output",
        ):
            self.assertIn(stage, lora_row)
        self.assertEqual(runner.model.lora.apply.__func__, original_apply.__func__)

    def test_reference_lora_respects_token_mapping(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")

        class Wrapper:
            token_lora_indices = torch.tensor([0, -1, 1])

        inputs = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        lora_a = torch.tensor([
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]],
        ])
        shrink = _reference_shrink(
            Wrapper(), torch.zeros((1, 3, 2)), inputs, (lora_a,), 2.0
        )
        self.assertTrue(torch.equal(
            shrink,
            torch.tensor([[[2.0, 4.0], [0.0, 0.0], [18.0, 14.0]]]),
        ))

        lora_b = torch.tensor([
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 1.0], [2.0, 0.0]]],
        ])
        expanded = _reference_expand_delta(
            Wrapper(),
            torch.zeros((3, 2)),
            shrink,
            (lora_b,),
            None,
            (2,),
        )
        self.assertTrue(torch.equal(
            expanded,
            torch.tensor([[2.0, 4.0], [0.0, 0.0], [32.0, 36.0]]),
        ))


if __name__ == "__main__":
    unittest.main()
