import unittest
from unittest.mock import patch

from infoskill.m1_lora_layer_localization import (
    VllmLayerCapture,
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
    def test_finds_first_changed_decoder_layer(self):
        report = compare_layer_rounds((trace(), trace(layer1="changed")))
        self.assertEqual(report["first_changed_layer"], 1)
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
        self.assertEqual(report["first_changed_stage"], "lora_shrink")

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


if __name__ == "__main__":
    unittest.main()
