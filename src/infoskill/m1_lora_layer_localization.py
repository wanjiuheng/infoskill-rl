"""Layer-by-layer localization for active vLLM LoRA inference drift.

This module is diagnostic-only.  It deliberately records hashes and bounded
statistics rather than tensor payloads, and never mutates model weights.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from infoskill.m1_lora_boundary import compare_boundary_rounds
from infoskill.rollout import GenerationRequest


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_LORA_STAGE_ORDER = (
    "input",
    "base_output",
    "lora_shrink",
    "lora_expand_delta",
    "combined_output",
)


def tensor_fingerprint(value: object) -> dict[str, object]:
    """Return an exact byte hash plus bounded numeric diagnostics."""

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("layer localization can fingerprint only torch tensors")
    detached = value.detach().to("cpu").contiguous()
    byte_view = detached.reshape(-1).view(torch.uint8)
    numeric = detached.float()
    return {
        "shape": [int(size) for size in detached.shape],
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "sha256": hashlib.sha256(byte_view.numpy().tobytes()).hexdigest(),
        "l2_norm": float(numeric.norm().item()),
        "abs_max": float(numeric.abs().max().item()) if numeric.numel() else 0.0,
    }


def _tensor_outputs(value: object) -> list[dict[str, object]]:
    import torch

    if isinstance(value, torch.Tensor):
        return [tensor_fingerprint(value)]
    if isinstance(value, (tuple, list)):
        return [tensor_fingerprint(item) for item in value if isinstance(item, torch.Tensor)]
    return []


def _record_key(row: Mapping[str, object]) -> tuple[object, object, object]:
    return row.get("rank"), row.get("module"), row.get("call")


def _hashes(row: Mapping[str, object], field: str) -> tuple[object, ...]:
    values = row.get(field, ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    return tuple(item.get("sha256") for item in values if isinstance(item, Mapping))


def compare_layer_rounds(
    rounds: Sequence[Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    """Find the earliest decoder layer whose exact output differs by round."""

    if len(rounds) < 2:
        raise ValueError("layer comparison needs at least two rounds")
    baseline = {_record_key(row): row for row in rounds[0]}
    common = set(baseline)
    indexed = []
    for rows in rounds[1:]:
        current = {_record_key(row): row for row in rows}
        indexed.append(current)
        common &= set(current)
    changed_modules: set[str] = set()
    for key in common:
        expected = _hashes(baseline[key], "output")
        if any(_hashes(current[key], "output") != expected for current in indexed):
            changed_modules.add(str(key[1]))
    changed_layers = sorted(
        int(match.group(1))
        for name in changed_modules
        if (match := _LAYER_PATTERN.search(name)) is not None
    )
    return {
        "round_count": len(rounds),
        "comparable_records": len(common),
        "changed_modules": sorted(changed_modules),
        "first_changed_layer": changed_layers[0] if changed_layers else None,
    }


def compare_lora_rounds(
    rounds: Sequence[Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    """Find the first exact LoRA sub-stage that changes across rounds."""

    if len(rounds) < 2:
        raise ValueError("LoRA comparison needs at least two rounds")
    baseline = {_record_key(row): row for row in rounds[0]}
    common = set(baseline)
    indexed = []
    for rows in rounds[1:]:
        current = {_record_key(row): row for row in rows}
        indexed.append(current)
        common &= set(current)
    changed_counts = {stage: 0 for stage in _LORA_STAGE_ORDER}
    for key in common:
        for stage in _LORA_STAGE_ORDER:
            expected = _hashes(baseline[key], stage)
            if any(_hashes(current[key], stage) != expected for current in indexed):
                changed_counts[stage] += 1
    first = next((stage for stage in _LORA_STAGE_ORDER if changed_counts[stage]), None)
    return {
        "round_count": len(rounds),
        "comparable_records": len(common),
        "changed_record_counts": changed_counts,
        "first_changed_stage": first,
    }


def classify_layer_localization(
    *,
    checkpoint_control: Mapping[str, object],
    checkpoint_instrumented: Mapping[str, object],
    base_instrumented: Mapping[str, object],
    layer_comparison: Mapping[str, object],
    lora_comparison: Mapping[str, object],
    lora_disabled: Mapping[str, object],
) -> str:
    """Attribute only after all causal and observer controls have passed."""

    exact = "exact_at_captured_boundaries"
    control = checkpoint_control.get("first_changed_boundary")
    instrumented = checkpoint_instrumented.get("first_changed_boundary")
    base = base_instrumented.get("first_changed_boundary")
    disabled = lora_disabled.get("first_changed_boundary")
    if control == "insufficient_common_history" or instrumented == "insufficient_common_history":
        return "inconclusive_no_common_history"
    if control != exact and instrumented == exact:
        return "observer_effect_inconclusive"
    if base != exact:
        return "base_control_drift"
    if disabled != exact:
        return "drift_persists_without_lora_request"
    if control == exact:
        return "no_checkpoint_drift_in_bounded_replay"
    first_layer = layer_comparison.get("first_changed_layer")
    if first_layer is None:
        return "drift_after_last_captured_decoder_layer"
    stage = lora_comparison.get("first_changed_stage")
    if stage is None:
        return "decoder_layer_drift_outside_captured_lora_stages"
    return f"{stage}_drift"


class VllmLayerCapture:
    """Scoped eager-mode hooks for decoder outputs and one layer's LoRA stages."""

    def __init__(self, model_runner: object, rank: int, *, layer: int | None = None):
        self.runner = model_runner
        self.rank = rank
        self.layer = layer
        self.rows: list[dict[str, object]] = []
        self._handles: list[object] = []
        self._patched: list[tuple[object, str, object]] = []
        self._calls: dict[str, int] = {}
        self._active_lora: dict[str, Any] | None = None

    def _next_call(self, module: str) -> int:
        value = self._calls.get(module, 0)
        self._calls[module] = value + 1
        return value

    def _decoder_hook(self, name: str):
        def hook(_module, _inputs, output):
            self.rows.append({
                "rank": self.rank,
                "module": name,
                "call": self._next_call(name),
                "output": _tensor_outputs(output),
            })
        return hook

    def _install_decoder_hooks(self, model: object) -> None:
        selected = []
        for name, module in model.named_modules():
            match = _LAYER_PATTERN.search(name)
            if match is None or name[match.end():]:
                continue
            if self.layer is None or int(match.group(1)) == self.layer:
                selected.append((name, module))
        if not selected:
            raise RuntimeError("no decoder layers matched the localization capture")
        for name, module in selected:
            self._handles.append(module.register_forward_hook(self._decoder_hook(name)))

    def _install_lora_hooks(self, model: object) -> None:
        if self.layer is None:
            return
        prefix = re.compile(rf"(?:^|\.)layers\.{self.layer}(?:\.|$)")
        modules = [
            (name, module) for name, module in model.named_modules()
            if prefix.search(name) is not None
            and hasattr(module, "punica_wrapper")
            and callable(getattr(module, "apply", None))
        ]
        if not modules:
            raise RuntimeError(f"decoder layer {self.layer} has no vLLM LoRA modules")

        wrappers: dict[int, object] = {}
        for name, module in modules:
            original = module.apply
            self._patched.append((module, "apply", original))

            def apply(*args, __name=name, __original=original, **kwargs):
                row = {
                    "rank": self.rank,
                    "module": __name,
                    "call": self._next_call(__name),
                    "input": _tensor_outputs(args[0] if args else kwargs.get("x")),
                }
                self._active_lora = row
                try:
                    result = __original(*args, **kwargs)
                    row.setdefault("combined_output", _tensor_outputs(result))
                    self.rows.append(row)
                    return result
                finally:
                    self._active_lora = None

            module.apply = apply
            wrapper = module.punica_wrapper
            wrappers[id(wrapper)] = wrapper

        for wrapper in wrappers.values():
            original_shrink = wrapper.add_shrink
            original_expand = wrapper.add_expand
            self._patched.append((wrapper, "add_shrink", original_shrink))
            self._patched.append((wrapper, "add_expand", original_expand))

            def shrink(y, x, *args, __original=original_shrink, **kwargs):
                result = __original(y, x, *args, **kwargs)
                if self._active_lora is not None:
                    self._active_lora["lora_shrink"] = _tensor_outputs(y)
                return result

            def expand(y, x, *args, __original=original_expand, **kwargs):
                base = y.detach().clone()
                if self._active_lora is not None:
                    self._active_lora["base_output"] = _tensor_outputs(base)
                result = __original(y, x, *args, **kwargs)
                if self._active_lora is not None:
                    self._active_lora["lora_expand_delta"] = _tensor_outputs(y - base)
                    self._active_lora["combined_output"] = _tensor_outputs(y)
                return result

            wrapper.add_shrink = shrink
            wrapper.add_expand = expand

    def install(self) -> None:
        if self._handles or self._patched:
            raise RuntimeError("layer localization capture is already installed")
        model = self.runner.model
        try:
            self._install_decoder_hooks(model)
            self._install_lora_hooks(model)
        except Exception:
            self.remove()
            raise

    def take(self) -> list[dict[str, object]]:
        rows, self.rows = self.rows, []
        self._calls = {}
        return rows

    def remove(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles = []
        for owner, name, original in reversed(self._patched):
            setattr(owner, name, original)
        self._patched = []
        self._active_lora = None


def _flatten_worker_rows(
    reports: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        row
        for report in reports
        for row in report.get("rows", ())
        if isinstance(row, Mapping)
    )


def _run_rounds(
    runtime: object,
    requests: tuple[GenerationRequest, ...],
    count: int,
    *,
    capture_layers: bool,
    progress,
    label: str,
) -> tuple[tuple[tuple[Mapping[str, object], ...], ...], tuple[tuple[Mapping[str, object], ...], ...]]:
    boundary_rounds = []
    layer_rounds = []
    for index in range(count):
        runtime.reset_rollout_prefix_cache()
        runtime.generate(requests)
        boundary_rounds.append(
            _flatten_worker_rows(runtime.take_vllm_boundary_rows())
        )
        if capture_layers:
            layer_rounds.append(_flatten_worker_rows(runtime.take_vllm_layer_rows()))
        if progress is not None:
            progress(label, f"round-{index + 1}-complete")
    return tuple(boundary_rounds), tuple(layer_rounds)


def collect_layer_localization_report(
    *,
    runtime_factory,
    checkpoint_runtime_directory: Path,
    probes: tuple[GenerationRequest, ...],
    detailed_rounds: int = 6,
    control_rounds: int = 3,
    progress=None,
    save_partial=None,
) -> dict[str, object]:
    """Run causal controls and layer/LoRA localization in one unattended job."""

    if len(probes) != 3:
        raise ValueError("layer localization requires exactly three probes")
    if detailed_rounds < 2 or control_rounds < 2:
        raise ValueError("layer localization requires at least two rounds per phase")
    requests = tuple(replace(probe, soft_prefix=None) for probe in probes)
    evidence: dict[str, object] = {}

    def persist() -> None:
        if save_partial is not None:
            save_partial({"schema_version": 1, "evidence": evidence})

    runtime = runtime_factory(False)
    try:
        load_reports = tuple(
            runtime.load_portable_state(checkpoint_runtime_directory) or ()
        )
        with runtime.rollout_session():
            runtime.begin_vllm_boundary_capture()
            try:
                control, _ = _run_rounds(
                    runtime, requests, detailed_rounds,
                    capture_layers=False, progress=progress,
                    label="checkpoint-eager-control",
                )
                evidence["checkpoint_eager_control"] = {
                    "comparison": compare_boundary_rounds(control),
                    "rounds": control,
                }
                persist()

                runtime.set_vllm_lora_request_enabled(False)
                try:
                    disabled, _ = _run_rounds(
                        runtime, requests, detailed_rounds,
                        capture_layers=False, progress=progress,
                        label="checkpoint-eager-lora-disabled",
                    )
                finally:
                    runtime.set_vllm_lora_request_enabled(True)
                evidence["checkpoint_eager_lora_disabled"] = {
                    "comparison": compare_boundary_rounds(disabled),
                    "rounds": disabled,
                }
                persist()

                runtime.begin_vllm_layer_capture()
                try:
                    instrumented, layer_rounds = _run_rounds(
                        runtime, requests, detailed_rounds,
                        capture_layers=True, progress=progress,
                        label="checkpoint-eager-layer-scan",
                    )
                finally:
                    runtime.end_vllm_layer_capture()
                layer_comparison = compare_layer_rounds(layer_rounds)
                evidence["checkpoint_eager_layer_scan"] = {
                    "boundary_comparison": compare_boundary_rounds(instrumented),
                    "layer_comparison": layer_comparison,
                    "boundary_rounds": instrumented,
                    "layer_rounds": layer_rounds,
                }
                persist()

                first_layer = layer_comparison["first_changed_layer"]
                lora_comparison: dict[str, object] = {
                    "round_count": 0,
                    "comparable_records": 0,
                    "changed_record_counts": {
                        stage: 0 for stage in _LORA_STAGE_ORDER
                    },
                    "first_changed_stage": None,
                }
                if first_layer is not None:
                    runtime.begin_vllm_layer_capture(int(first_layer))
                    try:
                        detailed_boundary, detailed_layers = _run_rounds(
                            runtime, requests, detailed_rounds,
                            capture_layers=True, progress=progress,
                            label=f"checkpoint-eager-layer-{first_layer}-lora",
                        )
                    finally:
                        runtime.end_vllm_layer_capture()
                    lora_comparison = compare_lora_rounds(detailed_layers)
                    evidence["checkpoint_eager_lora_detail"] = {
                        "target_layer": first_layer,
                        "boundary_comparison": compare_boundary_rounds(
                            detailed_boundary
                        ),
                        "lora_comparison": lora_comparison,
                        "boundary_rounds": detailed_boundary,
                        "layer_rounds": detailed_layers,
                    }
                    persist()
            finally:
                runtime.end_vllm_boundary_capture()
        evidence["checkpoint_load_reports"] = load_reports
    finally:
        runtime.close()

    runtime = runtime_factory(False)
    try:
        with runtime.rollout_session():
            runtime.begin_vllm_boundary_capture()
            runtime.begin_vllm_layer_capture()
            try:
                base_boundary, base_layers = _run_rounds(
                    runtime, requests, control_rounds,
                    capture_layers=True, progress=progress,
                    label="base-eager-layer-scan",
                )
            finally:
                runtime.end_vllm_layer_capture()
                runtime.end_vllm_boundary_capture()
        evidence["base_eager_layer_scan"] = {
            "boundary_comparison": compare_boundary_rounds(base_boundary),
            "layer_comparison": compare_layer_rounds(base_layers),
            "boundary_rounds": base_boundary,
            "layer_rounds": base_layers,
        }
        persist()
    finally:
        runtime.close()

    for label, load_checkpoint in (
        ("checkpoint_graph_control", True),
        ("base_graph_control", False),
    ):
        runtime = runtime_factory(True)
        try:
            if load_checkpoint:
                runtime.load_portable_state(checkpoint_runtime_directory)
            with runtime.rollout_session():
                runtime.begin_vllm_boundary_capture()
                try:
                    rounds, _ = _run_rounds(
                        runtime, requests, control_rounds,
                        capture_layers=False, progress=progress, label=label,
                    )
                finally:
                    runtime.end_vllm_boundary_capture()
            evidence[label] = {
                "comparison": compare_boundary_rounds(rounds),
                "rounds": rounds,
            }
            persist()
        finally:
            runtime.close()

    checkpoint_control = evidence["checkpoint_eager_control"]["comparison"]
    checkpoint_scan = evidence["checkpoint_eager_layer_scan"]
    base_scan = evidence["base_eager_layer_scan"]
    disabled = evidence["checkpoint_eager_lora_disabled"]["comparison"]
    classification = classify_layer_localization(
        checkpoint_control=checkpoint_control,
        checkpoint_instrumented=checkpoint_scan["boundary_comparison"],
        base_instrumented=base_scan["boundary_comparison"],
        layer_comparison=checkpoint_scan["layer_comparison"],
        lora_comparison=lora_comparison,
        lora_disabled=disabled,
    )
    return {
        "schema_version": 1,
        "classification": classification,
        "scope": "fixed one-token synthetic replay; not a valid_seen efficacy result",
        "controls": {
            "checkpoint_drift_reproduced_without_hooks": (
                checkpoint_control["first_changed_boundary"]
                != "exact_at_captured_boundaries"
            ),
            "checkpoint_drift_reproduced_with_layer_hooks": (
                checkpoint_scan["boundary_comparison"]["first_changed_boundary"]
                != "exact_at_captured_boundaries"
            ),
            "base_eager_stable_with_layer_hooks": (
                base_scan["boundary_comparison"]["first_changed_boundary"]
                == "exact_at_captured_boundaries"
            ),
            "lora_disabled_stable": (
                disabled["first_changed_boundary"]
                == "exact_at_captured_boundaries"
            ),
            "base_graph_stable": (
                evidence["base_graph_control"]["comparison"][
                    "first_changed_boundary"
                ] == "exact_at_captured_boundaries"
            ),
        },
        "first_changed_layer": checkpoint_scan["layer_comparison"][
            "first_changed_layer"
        ],
        "first_changed_lora_stage": lora_comparison["first_changed_stage"],
        "evidence": evidence,
        "interpretation_limit": (
            "A root-cause label is emitted only when no-hook, observer, base, "
            "and LoRA-disabled controls support it. Exact hashes localize the "
            "first observed seam but do not by themselves prove a kernel defect."
        ),
    }
