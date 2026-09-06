from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from infoskill.domain.state import CanonicalAgentState, render_policy_message
from infoskill.rollout import GenerationParameters, GenerationRequest, GenerationResult


@dataclass(frozen=True, slots=True)
class CheckpointEffectSamples:
    actor_comparison: tuple[Mapping[str, object], ...]
    baseline_vllm: tuple[Mapping[str, object], ...]
    checkpoint_vllm: tuple[Mapping[str, object], ...]
    baseline_results: tuple[GenerationResult, ...]
    checkpoint_results: tuple[GenerationResult, ...]


def collect_independent_checkpoint_effect_samples(
    *,
    runtime_factory: Callable[[], object],
    checkpoint_runtime_directory: Path,
    probes: tuple[GenerationRequest, ...],
) -> CheckpointEffectSamples:
    """Use a fresh load-first runtime and a separate baseline runtime."""

    checkpoint_runtime = runtime_factory()
    try:
        checkpoint_runtime.load_portable_state(  # type: ignore[attr-defined]
            checkpoint_runtime_directory
        )
        actor_comparison = tuple(
            checkpoint_runtime.compare_portable_actor_state(  # type: ignore[attr-defined]
                checkpoint_runtime_directory
            )
        )
        with checkpoint_runtime.rollout_session():  # type: ignore[attr-defined]
            checkpoint_runtime.reset_rollout_prefix_cache()  # type: ignore[attr-defined]
            checkpoint_vllm = tuple(
                checkpoint_runtime.vllm_lora_snapshot()  # type: ignore[attr-defined]
            )
            checkpoint_results = tuple(
                checkpoint_runtime.generate(probes)  # type: ignore[attr-defined]
            )
    finally:
        checkpoint_runtime.close()  # type: ignore[attr-defined]

    baseline_runtime = runtime_factory()
    try:
        with baseline_runtime.rollout_session():  # type: ignore[attr-defined]
            baseline_runtime.reset_rollout_prefix_cache()  # type: ignore[attr-defined]
            baseline_vllm = tuple(
                baseline_runtime.vllm_lora_snapshot()  # type: ignore[attr-defined]
            )
            baseline_results = tuple(
                baseline_runtime.generate(probes)  # type: ignore[attr-defined]
            )
    finally:
        baseline_runtime.close()  # type: ignore[attr-defined]

    return CheckpointEffectSamples(
        actor_comparison=actor_comparison,
        baseline_vllm=baseline_vllm,
        checkpoint_vllm=checkpoint_vllm,
        baseline_results=baseline_results,
        checkpoint_results=checkpoint_results,
    )


def build_checkpoint_effect_probes(
    *, master_seed: int, max_new_tokens: int = 64
) -> tuple[GenerationRequest, ...]:
    """Build small, stable ALFWorld-shaped prompts for a before/after policy probe."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    states = (
        CanonicalAgentState(
            task_id="checkpoint-effect/pick-place",
            split="valid_seen",
            task_type="pick_and_place_simple",
            goal="put a mug in the cabinet",
            step_index=0,
            observation=(
                "You are in the middle of a room. Looking quickly around you, "
                "you see a cabinet 1, a countertop 1, and a sidetable 1."
            ),
            history=(),
            admissible_commands=(
                "go to cabinet 1",
                "go to countertop 1",
                "go to sidetable 1",
                "inventory",
                "look",
            ),
        ),
        CanonicalAgentState(
            task_id="checkpoint-effect/heat-place",
            split="valid_seen",
            task_type="pick_heat_then_place_in_recep",
            goal="heat a cup and put it in the cabinet",
            step_index=0,
            observation=(
                "You are in the middle of a room. Looking quickly around you, "
                "you see a cabinet 1, a countertop 1, a microwave 1, and a sinkbasin 1."
            ),
            history=(),
            admissible_commands=(
                "go to cabinet 1",
                "go to countertop 1",
                "go to microwave 1",
                "go to sinkbasin 1",
                "inventory",
                "look",
            ),
        ),
        CanonicalAgentState(
            task_id="checkpoint-effect/light",
            split="valid_seen",
            task_type="look_at_obj_in_light",
            goal="look at a book under the light of a desklamp",
            step_index=0,
            observation=(
                "You are in the middle of a room. Looking quickly around you, "
                "you see a desk 1, a desklamp 1, a drawer 1, and a shelf 1."
            ),
            history=(),
            admissible_commands=(
                "go to desk 1",
                "go to desklamp 1",
                "go to drawer 1",
                "go to shelf 1",
                "inventory",
                "look",
            ),
        ),
    )
    parameters = GenerationParameters(
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
    )
    return tuple(
        GenerationRequest(
            request_id=state.task_id,
            task_id=state.task_id,
            rollout_id=0,
            env_step=0,
            user_message=render_policy_message(state, history_limit=2),
            parameters=parameters,
            seed=master_seed + index,
        )
        for index, state in enumerate(states)
    )


def summarize_named_tensors(tensors: Mapping[str, object]) -> dict[str, object]:
    """Return aggregate LoRA statistics without retaining tensor contents."""

    partitions: dict[str, list[object]] = {
        "all": [],
        "lora_a": [],
        "lora_b": [],
    }
    for name, tensor in tensors.items():
        partitions["all"].append(tensor)
        lowered = name.lower()
        if "lora_a" in lowered:
            partitions["lora_a"].append(tensor)
        if "lora_b" in lowered:
            partitions["lora_b"].append(tensor)
    return {
        "tensor_names_sha256": _tensor_names_sha256(tensors),
        "partitions": {
            name: _summarize_tensor_sequence(values)
            for name, values in partitions.items()
        },
    }


def compare_named_tensors(
    expected: Mapping[str, object], actual: Mapping[str, object]
) -> dict[str, object]:
    """Compare a portable adapter mapping with the live FSDP LoRA mapping."""

    import torch

    expected_names = set(expected)
    actual_names = set(actual)
    shared = sorted(expected_names & actual_names)
    shape_mismatches: list[str] = []
    unequal_names: list[str] = []
    max_abs_error = 0.0
    for name in shared:
        expected_tensor = expected[name]
        actual_tensor = actual[name]
        expected_shape = tuple(expected_tensor.shape)  # type: ignore[attr-defined]
        actual_shape = tuple(actual_tensor.shape)  # type: ignore[attr-defined]
        if expected_shape != actual_shape:
            shape_mismatches.append(name)
            continue
        left = expected_tensor.detach().to(device="cpu")  # type: ignore[attr-defined]
        right = actual_tensor.detach().to(device="cpu")  # type: ignore[attr-defined]
        if not torch.equal(left, right):
            unequal_names.append(name)
            error = float(
                (left.float() - right.float()).abs().max().item()
            )
            max_abs_error = max(max_abs_error, error)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    exact = not (missing or unexpected or shape_mismatches or unequal_names)
    return {
        "exact": exact,
        "expected_tensor_count": len(expected),
        "actual_tensor_count": len(actual),
        "missing_tensor_count": len(missing),
        "unexpected_tensor_count": len(unexpected),
        "shape_mismatch_count": len(shape_mismatches),
        "unequal_tensor_count": len(unequal_names),
        "max_abs_error": max_abs_error,
        "missing_tensors": missing[:20],
        "unexpected_tensors": unexpected[:20],
        "shape_mismatches": shape_mismatches[:20],
        "unequal_tensors": unequal_names[:20],
        "expected_summary": summarize_named_tensors(expected),
        "actual_summary": summarize_named_tensors(actual),
    }


def compare_generation_results(
    baseline: Sequence[GenerationResult],
    checkpoint: Sequence[GenerationResult],
    *,
    logprob_tolerance: float = 1e-7,
) -> dict[str, object]:
    """Compare deterministic generations while retaining per-probe evidence."""

    if logprob_tolerance < 0:
        raise ValueError("logprob_tolerance must be non-negative")
    before = {result.request_id: result for result in baseline}
    after = {result.request_id: result for result in checkpoint}
    missing = sorted(set(before) - set(after))
    unexpected = sorted(set(after) - set(before))
    probes = []
    maximum = 0.0
    aligned_count = 0
    changed_logprob_count = 0
    token_exact = not (missing or unexpected)
    text_exact = not (missing or unexpected)
    for request_id in sorted(set(before) & set(after)):
        left = before[request_id]
        right = after[request_id]
        tokens_equal = left.token_ids == right.token_ids
        texts_equal = left.text == right.text
        token_exact = token_exact and tokens_equal
        text_exact = text_exact and texts_equal
        local_errors = [
            abs(left_logprob - right_logprob)
            for left_token, right_token, left_logprob, right_logprob in zip(
                left.token_ids,
                right.token_ids,
                left.token_logprobs,
                right.token_logprobs,
            )
            if left_token == right_token
        ]
        local_maximum = max(local_errors, default=0.0)
        maximum = max(maximum, local_maximum)
        aligned_count += len(local_errors)
        local_changed = sum(error > logprob_tolerance for error in local_errors)
        changed_logprob_count += local_changed
        probes.append(
            {
                "request_id": request_id,
                "baseline_text": left.text,
                "checkpoint_text": right.text,
                "baseline_token_count": len(left.token_ids),
                "checkpoint_token_count": len(right.token_ids),
                "tokens_exact": tokens_equal,
                "text_exact": texts_equal,
                "aligned_logprob_count": len(local_errors),
                "changed_logprob_count": local_changed,
                "max_logprob_abs_error": local_maximum,
            }
        )
    logprobs_changed = changed_logprob_count > 0
    return {
        "request_count": len(before),
        "missing_requests": missing,
        "unexpected_requests": unexpected,
        "tokens_exact": token_exact,
        "text_exact": text_exact,
        "aligned_logprob_count": aligned_count,
        "changed_logprob_count": changed_logprob_count,
        "max_logprob_abs_error": maximum,
        "logprob_tolerance": logprob_tolerance,
        "logprobs_changed": logprobs_changed,
        "effect_visible": (not token_exact) or logprobs_changed,
        "probes": probes,
    }


def classify_checkpoint_effect(
    *,
    actor_snapshots: Sequence[Mapping[str, object]],
    vllm_snapshots: Sequence[Mapping[str, object]],
    vllm_checkpoint_comparison: Mapping[str, object],
    generation_comparison: Mapping[str, object],
) -> dict[str, object]:
    """Locate the first failed boundary in the portable-checkpoint inference path."""

    actor_exact = bool(actor_snapshots) and all(
        snapshot.get("exact") is True for snapshot in actor_snapshots
    )
    vllm_present = bool(vllm_snapshots) and all(
        _snapshot_has_nonzero_lora(snapshot) for snapshot in vllm_snapshots
    )
    vllm_matches = vllm_checkpoint_comparison.get("all_ranks_match") is True
    effect_visible = generation_comparison.get("effect_visible") is True
    if not actor_exact:
        classification = "checkpoint_to_fsdp_mismatch"
    elif not (vllm_present and vllm_matches):
        classification = "fsdp_to_vllm_missing_or_zero"
    elif not effect_visible:
        classification = "checkpoint_effect_below_probe_precision"
    else:
        classification = "checkpoint_effect_visible"
    return {
        "passed": actor_exact and vllm_present and vllm_matches and effect_visible,
        "classification": classification,
        "actor_matches_checkpoint_on_all_ranks": actor_exact,
        "nonzero_vllm_lora_on_all_ranks": vllm_present,
        "vllm_matches_checkpoint_aggregate_on_all_ranks": vllm_matches,
        "generation_effect_visible": effect_visible,
    }


def compare_vllm_to_checkpoint_aggregate(
    actor_snapshots: Sequence[Mapping[str, object]],
    vllm_snapshots: Sequence[Mapping[str, object]],
    *,
    lora_scaling: float,
    max_abs_relative_tolerance: float = 5e-3,
    l2_relative_tolerance: float = 1e-4,
) -> dict[str, object]:
    """Check vLLM's transformed LoRA against checkpoint aggregate statistics."""

    if lora_scaling <= 0:
        raise ValueError("lora_scaling must be positive")
    if min(max_abs_relative_tolerance, l2_relative_tolerance) < 0:
        raise ValueError("aggregate tolerances must be non-negative")
    actor_by_rank = {int(snapshot["rank"]): snapshot for snapshot in actor_snapshots}
    vllm_by_rank = {int(snapshot["rank"]): snapshot for snapshot in vllm_snapshots}
    missing_ranks = sorted(set(actor_by_rank) - set(vllm_by_rank))
    unexpected_ranks = sorted(set(vllm_by_rank) - set(actor_by_rank))
    ranks = []
    for rank in sorted(set(actor_by_rank) & set(vllm_by_rank)):
        actor_partitions = _nested_mapping(
            actor_by_rank[rank], "expected_summary", "partitions"
        )
        adapters = vllm_by_rank[rank].get("adapters")
        adapter = (
            next(iter(adapters.values()))
            if isinstance(adapters, Mapping) and len(adapters) == 1
            else None
        )
        vllm_partitions = (
            _nested_mapping(adapter, "summary", "partitions")
            if isinstance(adapter, Mapping)
            else {}
        )
        checks = {}
        for name, scale in (("lora_a", 1.0), ("lora_b", lora_scaling)):
            expected = actor_partitions.get(name)
            actual = vllm_partitions.get(name)
            checks[name] = _compare_partition_summary(
                expected if isinstance(expected, Mapping) else {},
                actual if isinstance(actual, Mapping) else {},
                scale=scale,
                max_abs_relative_tolerance=max_abs_relative_tolerance,
                l2_relative_tolerance=l2_relative_tolerance,
            )
        ranks.append(
            {
                "rank": rank,
                "matches": all(check["matches"] for check in checks.values()),
                "partitions": checks,
            }
        )
    all_match = (
        bool(ranks)
        and not missing_ranks
        and not unexpected_ranks
        and all(item["matches"] for item in ranks)
    )
    return {
        "all_ranks_match": all_match,
        "lora_scaling_merged_into_vllm_b": lora_scaling,
        "max_abs_relative_tolerance": max_abs_relative_tolerance,
        "l2_relative_tolerance": l2_relative_tolerance,
        "missing_ranks": missing_ranks,
        "unexpected_ranks": unexpected_ranks,
        "ranks": ranks,
    }


def flatten_lora_model_tensors(adapter: object) -> dict[str, object]:
    """Flatten pinned vLLM 0.8.4 LoRAModel weights for aggregate inspection."""

    result: dict[str, object] = {}
    for module_name, layer in adapter.loras.items():  # type: ignore[attr-defined]
        for field in ("lora_a", "lora_b"):
            _flatten_tensor_value(
                result,
                prefix=f"{module_name}.{field}",
                value=getattr(layer, field),
            )
    return result


def _flatten_tensor_value(
    destination: dict[str, object], *, prefix: str, value: object
) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten_tensor_value(
                destination,
                prefix=f"{prefix}.{index}",
                value=item,
            )
        return
    destination[prefix] = value


def _snapshot_has_nonzero_lora(snapshot: Mapping[str, object]) -> bool:
    adapters = snapshot.get("adapters")
    if not isinstance(adapters, Mapping) or len(adapters) != 1:
        return False
    active_ids = snapshot.get("active_adapter_ids")
    if not isinstance(active_ids, Sequence) or isinstance(active_ids, (str, bytes)):
        return False
    adapter_id = next(iter(adapters))
    if str(adapter_id) not in {str(value) for value in active_ids}:
        return False
    adapter = next(iter(adapters.values()))
    if not isinstance(adapter, Mapping):
        return False
    summary = adapter.get("summary")
    if not isinstance(summary, Mapping):
        return False
    partitions = summary.get("partitions")
    if not isinstance(partitions, Mapping):
        return False
    lora_b = partitions.get("lora_b")
    return isinstance(lora_b, Mapping) and int(lora_b.get("nonzero_count", 0)) > 0


def _compare_partition_summary(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
    *,
    scale: float,
    max_abs_relative_tolerance: float,
    l2_relative_tolerance: float,
) -> dict[str, object]:
    exact_fields = ("tensor_count", "element_count", "nonzero_count")
    exact = all(expected.get(field) == actual.get(field) for field in exact_fields)
    expected_max = float(expected.get("max_abs", math.nan)) * scale
    actual_max = float(actual.get("max_abs", math.nan))
    expected_l2_squared = float(expected.get("l2_squared", math.nan)) * scale**2
    actual_l2_squared = float(actual.get("l2_squared", math.nan))
    numeric = math.isclose(
        expected_max,
        actual_max,
        rel_tol=max_abs_relative_tolerance,
        abs_tol=1e-8,
    ) and math.isclose(
        expected_l2_squared,
        actual_l2_squared,
        rel_tol=l2_relative_tolerance,
        abs_tol=1e-8,
    )
    return {
        "matches": exact and numeric,
        "expected_tensor_count": expected.get("tensor_count"),
        "actual_tensor_count": actual.get("tensor_count"),
        "expected_element_count": expected.get("element_count"),
        "actual_element_count": actual.get("element_count"),
        "expected_nonzero_count": expected.get("nonzero_count"),
        "actual_nonzero_count": actual.get("nonzero_count"),
        "expected_max_abs_after_scaling": expected_max,
        "actual_max_abs": actual_max,
        "expected_l2_squared_after_scaling": expected_l2_squared,
        "actual_l2_squared": actual_l2_squared,
    }


def _nested_mapping(value: Mapping[str, object], *keys: str) -> Mapping[str, object]:
    current: object = value
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _summarize_tensor_sequence(tensors: Sequence[object]) -> dict[str, object]:
    tensor_count = 0
    element_count = 0
    nonzero_count = 0
    maximum = 0.0
    l2_squared = 0.0
    dtype_counts: dict[str, int] = {}
    for value in tensors:
        detached = value.detach()  # type: ignore[attr-defined]
        dtype = str(detached.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        tensor = detached.float()
        tensor_count += 1
        element_count += int(tensor.numel())
        if tensor.numel() == 0:
            continue
        nonzero_count += int(tensor.count_nonzero().item())
        maximum = max(maximum, float(tensor.abs().max().item()))
        l2_squared += float(tensor.square().sum().item())
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "nonzero_count": nonzero_count,
        "max_abs": maximum,
        "l2_squared": l2_squared,
        "l2_norm": math.sqrt(l2_squared),
        "dtype_counts": dict(sorted(dtype_counts.items())),
    }


def _tensor_names_sha256(tensors: Mapping[str, object]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))  # type: ignore[attr-defined]
        digest.update(b"\n")
    return digest.hexdigest()
