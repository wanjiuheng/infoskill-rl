"""Layer-by-layer localization for active vLLM LoRA inference drift.

The capture and comparison paths are diagnostic-only.  The scoped
``native_split_k_one`` intervention is also reused by the registered runtime
fix; it mutates neither model weights nor the installed vLLM package.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from infoskill.m1_lora_boundary import compare_boundary_rounds
from infoskill.rollout import GenerationRequest


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_NATIVE_LORA_STAGE_ORDER = (
    "token_lora_mapping",
    "input",
    "base_output",
    "lora_shrink_active",
    "lora_expand_delta",
    "combined_output",
)
_LORA_DIAGNOSTIC_FIELDS = _NATIVE_LORA_STAGE_ORDER + (
    "reference_shrink",
    "reference_expand_from_native_shrink",
    "reference_full_delta",
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


def _tensor_difference(actual: object, expected: object) -> dict[str, object]:
    import torch

    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError("tensor comparison requires torch tensors")
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"tensor comparison shape mismatch: {tuple(actual.shape)} != "
            f"{tuple(expected.shape)}"
        )
    delta = actual.detach().float() - expected.detach().float()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs_error": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "mean_abs_error": float(delta.abs().mean().item()) if delta.numel() else 0.0,
    }


def _token_lora_mapping(wrapper: object, token_count: int):
    """Return only mapping entries consumed by the current linear call."""

    import torch

    mapping = getattr(wrapper, "token_lora_indices", None)
    if not isinstance(mapping, torch.Tensor):
        mapping = getattr(wrapper, "_token_lora_indices", None)
    if not isinstance(mapping, torch.Tensor) or mapping.numel() < token_count:
        return None
    return mapping.reshape(-1)[:token_count]


def _active_shrink_rows(value: object, mapping: object):
    import torch

    if not isinstance(value, torch.Tensor) or not isinstance(mapping, torch.Tensor):
        return None
    if value.ndim != 3 or value.size(1) != mapping.numel():
        return None
    return value[:, mapping >= 0, :]


def _vllm_084_shrink_split_k(token_count: int) -> int:
    """Mirror the fixed split-K selection in pinned vLLM 0.8.4."""

    return 64 if token_count < 128 else 8


def _native_shrink_with_split_k(
    wrapper: object,
    output: object,
    inputs: object,
    lora_a_weights: object,
    scaling: object,
    *,
    split_k: int,
) -> None:
    """Launch pinned vLLM's native shrink kernel with one chosen split-K.

    This diagnostic-only path copies the vLLM 0.8.4 launch geometry and
    metadata exactly.  It changes only ``SPLIT_K``; it does not install a
    wheel patch or mutate model weights.
    """

    import torch
    import triton
    from vllm.lora.ops.triton_ops.lora_shrink import _lora_shrink_kernel
    from vllm.lora.ops.triton_ops.utils import _get_lora_a_ptr

    if not isinstance(output, torch.Tensor) or not isinstance(inputs, torch.Tensor):
        raise TypeError("native split-K shrink requires tensor inputs and output")
    if not isinstance(lora_a_weights, (tuple, list)) or not lora_a_weights:
        raise TypeError("native split-K shrink requires non-empty LoRA-A weights")
    if split_k < 1:
        raise ValueError("split_k must be positive")
    x = inputs.reshape(-1, inputs.shape[-1])
    metadata = wrapper.token_mapping_meta.meta_args(x.size(0))
    (
        token_lora_mapping,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
    ) = metadata
    if no_lora_flag_cpu.numel() != 1:
        raise RuntimeError("pinned vLLM no-LoRA flag has unexpected geometry")
    if no_lora_flag_cpu.item():
        return
    if x.dtype != lora_a_weights[0].dtype:
        raise RuntimeError("native split-K shrink input/weight dtype mismatch")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("native split-K shrink requires float16 or bfloat16")
    if not x.is_contiguous() or not output.is_contiguous():
        raise RuntimeError("native split-K shrink requires contiguous tensors")

    token_count = x.size(0)
    if token_lora_mapping.size(0) != token_count:
        raise RuntimeError("native split-K shrink token mapping length mismatch")
    if token_indices_sorted_by_lora_ids.size(0) != token_count:
        raise RuntimeError("native split-K shrink sorted-token length mismatch")
    if lora_ids.size(0) != num_tokens_per_lora.size(0):
        raise RuntimeError("native split-K shrink LoRA metadata length mismatch")
    if lora_token_start_loc.size(0) != lora_ids.size(0) + 1:
        raise RuntimeError("native split-K shrink start-location length mismatch")

    for weight in lora_a_weights:
        if not isinstance(weight, torch.Tensor):
            raise TypeError("native split-K shrink LoRA-A weight is not a tensor")
        if weight.dtype != x.dtype:
            raise RuntimeError("native split-K shrink LoRA-A dtype mismatch")
    if x.size(1) != lora_a_weights[0].size(-1):
        raise RuntimeError("native split-K shrink hidden size mismatch")

    (
        lora_ptr_tensor,
        lora_stride_d0,
        lora_stride_d1,
        lora_stride_d2,
    ) = _get_lora_a_ptr(lora_a_weights, x.device)
    rank, hidden_size = lora_a_weights[0].shape[-2:]
    slice_count = len(lora_a_weights)
    max_loras = lora_ids.size(0)
    block_m = 32
    block_n = 16
    block_k = 256 if token_count < 128 else 32
    even_k = hidden_size % (block_k * split_k) == 0
    grid = (
        split_k
        * triton.cdiv(token_count, block_m)
        * triton.cdiv(rank, block_n),
        slice_count,
        max_loras,
    )
    with torch.inference_mode():
        _lora_shrink_kernel[grid](
            x,
            lora_ptr_tensor,
            output,
            token_count,
            rank,
            hidden_size,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            float(scaling),
            x.stride(0),
            x.stride(1),
            lora_stride_d0,
            lora_stride_d1,
            lora_stride_d2,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            block_m,
            block_n,
            block_k,
            even_k,
            split_k,
            slice_count,
            num_warps=4,
            num_ctas=1,
            num_stages=2,
            maxnreg=None,
        )


def _reference_shrink(
    wrapper: object,
    template: object,
    inputs: object,
    lora_a_stacked: object,
    scale: object,
):
    """Compute the vLLM 0.8.4 shrink semantics with deterministic FP32 GEMMs."""

    import torch

    if not isinstance(template, torch.Tensor) or not isinstance(inputs, torch.Tensor):
        return None
    if not isinstance(lora_a_stacked, (tuple, list)):
        return None
    x = inputs.reshape(-1, inputs.shape[-1])
    mapping = _token_lora_mapping(wrapper, x.size(0))
    if mapping is None or template.ndim != 3:
        return None
    result = torch.zeros_like(template)
    active_slots = sorted(
        int(item) for item in torch.unique(mapping[mapping >= 0]).detach().cpu().tolist()
    )
    for slice_index, weights in enumerate(lora_a_stacked):
        if not isinstance(weights, torch.Tensor):
            return None
        for slot in active_slots:
            mask = mapping == slot
            positions = torch.nonzero(mask, as_tuple=False).reshape(-1)
            weight = weights[slot, 0].float()
            product = torch.matmul(x[mask].float(), weight.transpose(0, 1))
            product = product * float(scale)
            result[slice_index, positions, :product.size(-1)] = product.to(
                result.dtype
            )
    return result


def _reference_expand_delta(
    wrapper: object,
    output_template: object,
    shrink: object,
    lora_b_stacked: object,
    lora_bias_stacked: object,
    output_slices: object,
    *,
    offset_start: int = 0,
):
    """Compute the vLLM 0.8.4 expand semantics as an FP32 delta tensor."""

    import torch

    if not isinstance(output_template, torch.Tensor) or not isinstance(shrink, torch.Tensor):
        return None
    if not isinstance(lora_b_stacked, (tuple, list)):
        return None
    if not isinstance(output_slices, (tuple, list)):
        return None
    flat_output = output_template.reshape(-1, output_template.shape[-1])
    if shrink.ndim != 3:
        return None
    mapping = _token_lora_mapping(wrapper, shrink.size(1))
    if mapping is None or flat_output.size(0) != shrink.size(1):
        return None
    result = torch.zeros_like(flat_output)
    active_slots = sorted(
        int(item) for item in torch.unique(mapping[mapping >= 0]).detach().cpu().tolist()
    )
    offset = int(offset_start)
    for slice_index, (weights, width) in enumerate(
        zip(lora_b_stacked, output_slices, strict=True)
    ):
        if not isinstance(weights, torch.Tensor):
            return None
        width = int(width)
        for slot in active_slots:
            mask = mapping == slot
            positions = torch.nonzero(mask, as_tuple=False).reshape(-1)
            weight = weights[slot, 0].float()
            product = torch.matmul(
                shrink[slice_index, mask].float(), weight.transpose(0, 1)
            )
            if product.size(-1) != width:
                raise RuntimeError(
                    "reference expand width differs from vLLM output_slices"
                )
            bias = None
            if isinstance(lora_bias_stacked, (tuple, list)):
                candidate = lora_bias_stacked[slice_index]
                if isinstance(candidate, torch.Tensor):
                    bias = candidate.reshape(candidate.size(0), -1)[
                        slot, :width
                    ].float()
            if bias is not None:
                product = product + bias
            result[positions, offset:offset + width] = product.to(result.dtype)
        offset += width
    return result.view_as(output_template)


def _apply_reference_expand(
    wrapper: object,
    output: object,
    shrink: object,
    lora_b_stacked: object,
    lora_bias_stacked: object,
    output_slices: object,
    *,
    offset_start: int,
    add_inputs: bool,
) -> None:
    import torch

    if not isinstance(output, torch.Tensor):
        raise TypeError("reference expand requires a tensor output")
    delta = _reference_expand_delta(
        wrapper,
        output,
        shrink,
        lora_b_stacked,
        lora_bias_stacked,
        output_slices,
        offset_start=offset_start,
    )
    if delta is None:
        raise RuntimeError("reference expand could not resolve vLLM LoRA metadata")
    if add_inputs:
        output.add_(delta)
    else:
        output.copy_(delta)


class VllmLoraKernelIntervention:
    """Replace vLLM 0.8.4 Punica stages inside one rollout session."""

    MODES = {
        "native_split_k_one",
        "reference_shrink",
        "reference_expand",
        "reference_full",
    }

    def __init__(self, model_runner: object, mode: str):
        if mode not in self.MODES:
            raise ValueError(f"unsupported LoRA kernel intervention: {mode}")
        self.runner = model_runner
        self.mode = mode
        self._patched: list[tuple[object, str, object]] = []

    def install(self) -> None:
        if self._patched:
            raise RuntimeError("LoRA kernel intervention is already active")
        wrappers = {
            id(module.punica_wrapper): module.punica_wrapper
            for _, module in self.runner.model.named_modules()
            if hasattr(module, "punica_wrapper")
        }
        if not wrappers:
            raise RuntimeError("no vLLM Punica wrappers found for intervention")
        try:
            for wrapper in wrappers.values():
                if self.mode == "native_split_k_one":
                    original = wrapper.add_shrink
                    self._patched.append((wrapper, "add_shrink", original))

                    def split_k_one_shrink(
                        y,
                        x,
                        lora_a_stacked,
                        scale,
                        *,
                        __wrapper=wrapper,
                        **kwargs,
                    ):
                        _native_shrink_with_split_k(
                            __wrapper,
                            y,
                            x,
                            lora_a_stacked,
                            scale,
                            split_k=1,
                        )

                    wrapper.add_shrink = split_k_one_shrink
                if self.mode in {"reference_shrink", "reference_full"}:
                    original = wrapper.add_shrink
                    self._patched.append((wrapper, "add_shrink", original))

                    def shrink(
                        y,
                        x,
                        lora_a_stacked,
                        scale,
                        *,
                        __wrapper=wrapper,
                        **kwargs,
                    ):
                        reference = _reference_shrink(
                            __wrapper, y, x, lora_a_stacked, scale
                        )
                        if reference is None:
                            raise RuntimeError(
                                "reference shrink could not resolve vLLM LoRA "
                                "metadata"
                            )
                        y.copy_(reference)

                    wrapper.add_shrink = shrink
                if self.mode in {"reference_expand", "reference_full"}:
                    original = wrapper.add_expand
                    self._patched.append((wrapper, "add_expand", original))

                    def expand(
                        y,
                        x,
                        lora_b_stacked,
                        lora_bias_stacked,
                        output_slices,
                        offset_start=0,
                        add_inputs=True,
                        *,
                        __wrapper=wrapper,
                        **kwargs,
                    ):
                        _apply_reference_expand(
                            __wrapper,
                            y,
                            x,
                            lora_b_stacked,
                            lora_bias_stacked,
                            output_slices,
                            offset_start=offset_start,
                            add_inputs=add_inputs,
                        )

                    wrapper.add_expand = expand
        except Exception:
            self.remove()
            raise

    def remove(self) -> None:
        for owner, name, original in reversed(self._patched):
            setattr(owner, name, original)
        self._patched = []


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
    changed_by_rank: dict[int, set[str]] = {}
    for key in common:
        expected = _hashes(baseline[key], "output")
        if any(_hashes(current[key], "output") != expected for current in indexed):
            module = str(key[1])
            rank = int(key[0])
            changed_modules.add(module)
            changed_by_rank.setdefault(rank, set()).add(module)
    changed_layers = sorted(
        int(match.group(1))
        for name in changed_modules
        if (match := _LAYER_PATTERN.search(name)) is not None
    )
    first_by_rank = {}
    for rank, modules in sorted(changed_by_rank.items()):
        layers = sorted(
            int(match.group(1))
            for name in modules
            if (match := _LAYER_PATTERN.search(name)) is not None
        )
        first_by_rank[str(rank)] = layers[0] if layers else None
    return {
        "round_count": len(rounds),
        "comparable_records": len(common),
        "changed_modules": sorted(changed_modules),
        "first_changed_layer": changed_layers[0] if changed_layers else None,
        "first_changed_layer_by_rank": first_by_rank,
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
    changed_counts = {stage: 0 for stage in _LORA_DIAGNOSTIC_FIELDS}
    changed: set[tuple[object, object, object, str]] = set()
    for key in common:
        for stage in _LORA_DIAGNOSTIC_FIELDS:
            field = stage
            if stage == "lora_shrink_active" and not _hashes(
                baseline[key], stage
            ):
                field = "lora_shrink"
            expected = _hashes(baseline[key], field)
            if expected and any(
                _hashes(current[key], field) != expected for current in indexed
            ):
                changed_counts[stage] += 1
                changed.add((*key, stage))

    # A downstream module's input can change only because an earlier module
    # already changed.  Preserve the actual module execution order from the
    # first round instead of globally prioritising the field name "input".
    ordered_events = []
    first_by_rank: dict[str, dict[str, object]] = {}
    for row in rounds[0]:
        key = _record_key(row)
        if key not in common:
            continue
        for stage in _NATIVE_LORA_STAGE_ORDER:
            if (*key, stage) not in changed:
                continue
            event = {
                "rank": int(key[0]),
                "module": str(key[1]),
                "call": int(key[2]),
                "stage": stage,
            }
            ordered_events.append(event)
            first_by_rank.setdefault(str(key[0]), event)
            break
    first_event = ordered_events[0] if ordered_events else None
    return {
        "round_count": len(rounds),
        "comparable_records": len(common),
        "changed_record_counts": changed_counts,
        "first_changed_stage": (
            first_event["stage"] if first_event is not None else None
        ),
        "first_changed_event": first_event,
        "first_changed_event_by_rank": first_by_rank,
    }


def classify_kernel_causality(
    *,
    native: Mapping[str, object],
    native_split_k_one: Mapping[str, object],
    reference_shrink: Mapping[str, object],
    reference_expand: Mapping[str, object],
    reference_full: Mapping[str, object],
    rotation: Mapping[str, object],
) -> str:
    """Classify the native kernel stage by causal replacement controls."""

    exact = "exact_at_captured_boundaries"
    boundaries = (
        native.get("first_changed_boundary"),
        native_split_k_one.get("first_changed_boundary"),
        reference_shrink.get("first_changed_boundary"),
        reference_expand.get("first_changed_boundary"),
        reference_full.get("first_changed_boundary"),
        rotation.get("first_changed_boundary"),
    )
    if "insufficient_common_history" in boundaries:
        return "inconclusive_no_common_history"
    if native.get("first_changed_boundary") == exact:
        return "native_drift_not_reproduced"
    if rotation.get("first_changed_boundary") == exact:
        return "request_order_sensitive_inconclusive"
    if reference_full.get("first_changed_boundary") != exact:
        return "drift_persists_with_full_reference_lora"
    shrink_fixed = reference_shrink.get("first_changed_boundary") == exact
    expand_fixed = reference_expand.get("first_changed_boundary") == exact
    if shrink_fixed and not expand_fixed:
        if native_split_k_one.get("first_changed_boundary") == exact:
            return "native_shrink_split_k_atomic_nondeterminism"
        return "native_shrink_nondeterminism_not_eliminated_by_split_k_one"
    if expand_fixed and not shrink_fixed:
        return "native_expand_nondeterminism"
    if shrink_fixed and expand_fixed:
        return "either_native_stage_independently_triggers_drift"
    return "compound_native_shrink_expand_drift"


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
                    row.pop("_reference_shrink_tensor", None)
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

            def shrink(
                y,
                x,
                *args,
                __original=original_shrink,
                __wrapper=wrapper,
                **kwargs,
            ):
                result = __original(y, x, *args, **kwargs)
                if self._active_lora is not None:
                    self._active_lora["lora_shrink"] = _tensor_outputs(y)
                    try:
                        import torch
                    except ImportError:
                        torch = None
                    if (
                        torch is not None
                        and isinstance(y, torch.Tensor)
                        and isinstance(x, torch.Tensor)
                        and len(args) >= 2
                    ):
                        lora_a_stacked, scale = args[:2]
                        token_count = int(x.reshape(-1, x.shape[-1]).size(0))
                        mapping = _token_lora_mapping(__wrapper, token_count)
                        if mapping is None:
                            raise RuntimeError(
                                "layer audit could not resolve token LoRA mapping"
                            )
                        active = _active_shrink_rows(y, mapping)
                        if active is None:
                            raise RuntimeError(
                                "layer audit found incompatible shrink geometry"
                            )
                        self._active_lora["token_lora_mapping"] = (
                            _tensor_outputs(mapping)
                        )
                        self._active_lora["lora_shrink_active"] = (
                            _tensor_outputs(active)
                        )
                        reference = _reference_shrink(
                            __wrapper, y, x, lora_a_stacked, scale
                        )
                        if reference is None:
                            raise RuntimeError(
                                "layer audit could not compute reference shrink"
                            )
                        reference_active = _active_shrink_rows(reference, mapping)
                        if reference_active is None:
                            raise RuntimeError(
                                "layer audit reference shrink geometry differs"
                            )
                        self._active_lora["reference_shrink"] = (
                            _tensor_outputs(reference_active)
                        )
                        self._active_lora[
                            "native_vs_reference_shrink"
                        ] = _tensor_difference(active, reference_active)
                        self._active_lora["_reference_shrink_tensor"] = reference
                        split_k = _vllm_084_shrink_split_k(token_count)
                        self._active_lora["native_shrink_split_k"] = split_k
                        self._active_lora[
                            "native_shrink_atomic_reduction"
                        ] = split_k > 1
                return result

            def expand(
                y,
                x,
                *args,
                __original=original_expand,
                __wrapper=wrapper,
                **kwargs,
            ):
                base = y.detach().clone()
                if self._active_lora is not None:
                    self._active_lora["base_output"] = _tensor_outputs(base)
                    if len(args) >= 3:
                        lora_b_stacked = args[0]
                        lora_bias_stacked = args[1]
                        output_slices = args[2]
                        offset_start = int(
                            kwargs.get(
                                "offset_start",
                                args[3] if len(args) >= 4 else 0,
                            )
                        )
                        native_reference = _reference_expand_delta(
                            __wrapper,
                            y,
                            x,
                            lora_b_stacked,
                            lora_bias_stacked,
                            output_slices,
                            offset_start=offset_start,
                        )
                        if native_reference is None:
                            raise RuntimeError(
                                "layer audit could not compute reference expand"
                            )
                        self._active_lora[
                            "reference_expand_from_native_shrink"
                        ] = _tensor_outputs(native_reference)
                        full_reference = _reference_expand_delta(
                            __wrapper,
                            y,
                            self._active_lora.get("_reference_shrink_tensor"),
                            lora_b_stacked,
                            lora_bias_stacked,
                            output_slices,
                            offset_start=offset_start,
                        )
                        if full_reference is None:
                            raise RuntimeError(
                                "layer audit could not compute full reference LoRA"
                            )
                        self._active_lora["reference_full_delta"] = (
                            _tensor_outputs(full_reference)
                        )
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


def _annotate_layer_histories(
    boundary_rows: Sequence[Mapping[str, object]],
    layer_rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    histories: dict[int, str] = {}
    ambiguous: set[int] = set()
    for row in boundary_rows:
        rank = int(row["rank"])
        history = str(row["history_sha256"])
        if rank in histories and histories[rank] != history:
            ambiguous.add(rank)
        histories[rank] = history
    return tuple(
        dict(
            row,
            probe_history_sha256=(
                histories.get(int(row["rank"]))
                if int(row["rank"]) not in ambiguous
                else None
            ),
        )
        for row in layer_rows
    )


def _aggregate_boundary_comparisons(
    comparisons: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not comparisons:
        raise ValueError("boundary aggregate requires at least one comparison")
    names = (
        "final_hidden",
        "raw_logits",
        "processed_logits",
        "sampled_token",
        "returned_logprob",
    )
    counts = {
        name: sum(
            int(report.get("changed_row_counts", {}).get(name, 0))
            for report in comparisons
        )
        for name in names
    }
    comparable = sum(int(item.get("comparable_rows", 0)) for item in comparisons)
    first = next((name for name in names if counts[name]), None)
    if comparable == 0:
        first = "insufficient_common_history"
    return {
        "cell_count": len(comparisons),
        "comparable_rows": comparable,
        "returned_logprob_comparable_rows": sum(
            int(item.get("returned_logprob_comparable_rows", 0))
            for item in comparisons
        ),
        "changed_row_counts": counts,
        "first_changed_boundary": first or "exact_at_captured_boundaries",
    }


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
            layer_rounds.append(
                _annotate_layer_histories(
                    boundary_rounds[-1],
                    _flatten_worker_rows(runtime.take_vllm_layer_rows()),
                )
            )
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
            save_partial({"schema_version": 3, "evidence": evidence})

    runtime = runtime_factory(False)
    try:
        load_reports = tuple(
            runtime.load_portable_state(checkpoint_runtime_directory) or ()
        )
        with runtime.rollout_session():
            runtime.begin_vllm_boundary_capture()
            try:
                phase_started = time.perf_counter()
                control, _ = _run_rounds(
                    runtime, requests, detailed_rounds,
                    capture_layers=False, progress=progress,
                    label="checkpoint-eager-control",
                )
                evidence["checkpoint_eager_control"] = {
                    "comparison": compare_boundary_rounds(control),
                    "rounds": control,
                    "diagnostic_seconds": time.perf_counter() - phase_started,
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
                        stage: 0 for stage in _LORA_DIAGNOSTIC_FIELDS
                    },
                    "first_changed_stage": None,
                    "first_changed_event": None,
                    "first_changed_event_by_rank": {},
                }
                layer_details: dict[str, object] = {}
                per_rank_layers = layer_comparison.get(
                    "first_changed_layer_by_rank", {}
                )
                target_layers = sorted({
                    int(layer)
                    for layer in per_rank_layers.values()
                    if layer is not None
                })
                if first_layer is not None and int(first_layer) not in target_layers:
                    target_layers.append(int(first_layer))
                    target_layers.sort()
                for target_layer in target_layers:
                    runtime.begin_vllm_layer_capture(target_layer)
                    try:
                        detailed_boundary, detailed_layers = _run_rounds(
                            runtime, requests, detailed_rounds,
                            capture_layers=True, progress=progress,
                            label=f"checkpoint-eager-layer-{target_layer}-lora",
                        )
                    finally:
                        runtime.end_vllm_layer_capture()
                    detail_comparison = compare_lora_rounds(detailed_layers)
                    layer_details[str(target_layer)] = {
                        "target_layer": target_layer,
                        "boundary_comparison": compare_boundary_rounds(
                            detailed_boundary
                        ),
                        "lora_comparison": detail_comparison,
                        "boundary_rounds": detailed_boundary,
                        "layer_rounds": detailed_layers,
                    }
                    persist()
                evidence["checkpoint_eager_lora_details"] = layer_details
                if first_layer is not None and str(first_layer) in layer_details:
                    evidence["checkpoint_eager_lora_detail"] = layer_details[
                        str(first_layer)
                    ]
                    lora_comparison = evidence[
                        "checkpoint_eager_lora_detail"
                    ]["lora_comparison"]

                first_lora_event_by_rank = {}
                for rank, target_layer in per_rank_layers.items():
                    detail = layer_details.get(str(target_layer), {})
                    comparison = detail.get("lora_comparison", {})
                    event = comparison.get(
                        "first_changed_event_by_rank", {}
                    ).get(str(rank))
                    if event is not None:
                        first_lora_event_by_rank[str(rank)] = event
                evidence["first_changed_lora_event_by_rank"] = (
                    first_lora_event_by_rank
                )
                persist()

                rotation_cells = {}
                rotation_comparisons = []
                for offset in range(len(requests)):
                    rotated = requests[offset:] + requests[:offset]
                    rotation_rounds, _ = _run_rounds(
                        runtime,
                        rotated,
                        control_rounds,
                        capture_layers=False,
                        progress=progress,
                        label=f"checkpoint-eager-probe-rotation-{offset}",
                    )
                    comparison = compare_boundary_rounds(rotation_rounds)
                    rotation_comparisons.append(comparison)
                    rotation_cells[str(offset)] = {
                        "request_order": [
                            probe.request_id for probe in rotated
                        ],
                        "comparison": comparison,
                        "rounds": rotation_rounds,
                    }
                evidence["checkpoint_eager_probe_rotation"] = {
                    "comparison": _aggregate_boundary_comparisons(
                        rotation_comparisons
                    ),
                    "cells": rotation_cells,
                }
                persist()

                intervention_evidence = {}
                for mode in (
                    "native_split_k_one",
                    "reference_shrink",
                    "reference_expand",
                    "reference_full",
                ):
                    runtime.begin_vllm_lora_kernel_intervention(mode)
                    phase_started = time.perf_counter()
                    try:
                        intervention_rounds, _ = _run_rounds(
                            runtime,
                            requests,
                            detailed_rounds,
                            capture_layers=False,
                            progress=progress,
                            label=f"checkpoint-eager-{mode}",
                        )
                    finally:
                        runtime.end_vllm_lora_kernel_intervention()
                    intervention_evidence[mode] = {
                        "comparison": compare_boundary_rounds(
                            intervention_rounds
                        ),
                        "rounds": intervention_rounds,
                        "diagnostic_seconds": time.perf_counter() - phase_started,
                    }
                    evidence["checkpoint_eager_lora_interventions"] = (
                        intervention_evidence
                    )
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
    layer_classification = classify_layer_localization(
        checkpoint_control=checkpoint_control,
        checkpoint_instrumented=checkpoint_scan["boundary_comparison"],
        base_instrumented=base_scan["boundary_comparison"],
        layer_comparison=checkpoint_scan["layer_comparison"],
        lora_comparison=lora_comparison,
        lora_disabled=disabled,
    )
    interventions = evidence["checkpoint_eager_lora_interventions"]
    rotation = evidence["checkpoint_eager_probe_rotation"]["comparison"]
    attributable_layer_labels = {
        "lora_shrink_active_drift",
        "lora_expand_delta_drift",
        "combined_output_drift",
    }
    if layer_classification in attributable_layer_labels:
        classification = classify_kernel_causality(
            native=checkpoint_control,
            native_split_k_one=interventions[
                "native_split_k_one"
            ]["comparison"],
            reference_shrink=interventions["reference_shrink"]["comparison"],
            reference_expand=interventions["reference_expand"]["comparison"],
            reference_full=interventions["reference_full"]["comparison"],
            rotation=rotation,
        )
    else:
        classification = f"blocked_by_{layer_classification}"
    return {
        "schema_version": 3,
        "classification": classification,
        "layer_classification": layer_classification,
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
            "checkpoint_graph_has_common_history": (
                evidence["checkpoint_graph_control"]["comparison"][
                    "comparable_rows"
                ] > 0
            ),
            "probe_rotation_reproduced_drift": (
                rotation["first_changed_boundary"]
                != "exact_at_captured_boundaries"
            ),
            "full_reference_lora_stable": (
                interventions["reference_full"]["comparison"][
                    "first_changed_boundary"
                ] == "exact_at_captured_boundaries"
            ),
            "native_split_k_one_stable": (
                interventions["native_split_k_one"]["comparison"][
                    "first_changed_boundary"
                ] == "exact_at_captured_boundaries"
            ),
        },
        "first_changed_layer": checkpoint_scan["layer_comparison"][
            "first_changed_layer"
        ],
        "first_changed_layer_by_rank": checkpoint_scan["layer_comparison"][
            "first_changed_layer_by_rank"
        ],
        "first_changed_lora_stage": lora_comparison["first_changed_stage"],
        "first_changed_lora_event_by_rank": evidence[
            "first_changed_lora_event_by_rank"
        ],
        "kernel_causality": {
            "native": checkpoint_control,
            "native_split_k_one": interventions[
                "native_split_k_one"
            ]["comparison"],
            "probe_rotation": rotation,
            "reference_shrink": interventions[
                "reference_shrink"
            ]["comparison"],
            "reference_expand": interventions[
                "reference_expand"
            ]["comparison"],
            "reference_full": interventions[
                "reference_full"
            ]["comparison"],
        },
        "diagnostic_phase_seconds": {
            "native": evidence["checkpoint_eager_control"][
                "diagnostic_seconds"
            ],
            **{
                mode: value["diagnostic_seconds"]
                for mode, value in interventions.items()
            },
        },
        "evidence": evidence,
        "interpretation_limit": (
            "A kernel root-cause label is emitted only when no-hook, observer, "
            "base, LoRA-disabled, probe-rotation, and deterministic replacement "
            "controls support it. Split-K is named as causal only when the same "
            "native Triton shrink kernel becomes stable with SPLIT_K=1. Timings "
            "are diagnostic-phase measurements, not production throughput. This "
            "synthetic one-token result still does not measure ALFWorld "
            "success-rate efficacy."
        ),
    }
