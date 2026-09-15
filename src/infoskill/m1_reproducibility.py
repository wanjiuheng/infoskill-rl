from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from infoskill.checkpoint_effect import (
    build_checkpoint_effect_probes,
    compare_generation_results,
)
from infoskill.rollout import GenerationRequest, GenerationResult


@dataclass(frozen=True, slots=True)
class RuntimeReproducibilitySample:
    label: str
    checkpoint_loaded: bool
    load_reports: tuple[Mapping[str, object], ...]
    actor_snapshots: tuple[Mapping[str, object], ...]
    infoskill_snapshots: tuple[Mapping[str, object], ...]
    vllm_lora_snapshots: tuple[tuple[Mapping[str, object], ...], ...]
    vllm_base_snapshots: tuple[tuple[Mapping[str, object], ...], ...]
    generation_rounds: tuple[tuple[GenerationResult, ...], ...]


def build_hybrid_prefix_reproducibility_probes(
    *,
    hidden_size: int,
    prefix_length: int,
    master_seed: int,
    max_new_tokens: int,
    case_count: int,
) -> tuple[GenerationRequest, ...]:
    """Build fixed ALFWorld prompts with exact synthetic soft prefixes."""

    if min(hidden_size, prefix_length, max_new_tokens, case_count) <= 0:
        raise ValueError("hybrid-prefix probe dimensions must be positive")
    base = build_checkpoint_effect_probes(
        master_seed=master_seed,
        max_new_tokens=max_new_tokens,
    )
    if case_count > len(base):
        raise ValueError(f"case_count must not exceed {len(base)}")

    import torch

    positions = torch.arange(
        prefix_length * hidden_size,
        dtype=torch.float32,
    ).reshape(prefix_length, hidden_size)
    probes = []
    for index, request in enumerate(base[:case_count]):
        prefix = torch.sin(positions * 0.0009765625 + float(index)) * 0.0086
        probes.append(replace(request, soft_prefix=prefix.to(torch.bfloat16)))
    return tuple(probes)


def collect_m1_reproducibility_samples(
    *,
    checkpoint_runtime_factory: Callable[[], object],
    base_runtime_factory: Callable[[], object],
    checkpoint_runtime_directory: Path,
    probes: tuple[GenerationRequest, ...],
    progress: Callable[[str, str], None] | None = None,
) -> tuple[RuntimeReproducibilitySample, ...]:
    """Probe two fresh checkpoint runtimes and two fresh base-only controls."""

    samples = []
    for label, factory, load_checkpoint in (
        ("checkpoint-a", checkpoint_runtime_factory, True),
        ("checkpoint-b", checkpoint_runtime_factory, True),
        ("base-a", base_runtime_factory, False),
        ("base-b", base_runtime_factory, False),
    ):
        if progress is not None:
            progress(label, "initializing")
        runtime = factory()
        try:
            load_reports = ()
            actor_snapshots = ()
            infoskill_snapshots = ()
            if load_checkpoint:
                load_reports = tuple(
                    runtime.load_portable_state(  # type: ignore[attr-defined]
                        checkpoint_runtime_directory
                    )
                    or ()
                )
                actor_snapshots = tuple(
                    runtime.compare_portable_actor_state(  # type: ignore[attr-defined]
                        checkpoint_runtime_directory
                    )
                )
                infoskill_snapshots = tuple(
                    runtime.infoskill_module_snapshot()  # type: ignore[attr-defined]
                )
                if progress is not None:
                    progress(label, "checkpoint-loaded-and-fingerprinted")

            lora_snapshots = []
            base_snapshots = []
            generation_rounds = []
            with runtime.rollout_session():  # type: ignore[attr-defined]
                for round_index in range(2):
                    runtime.reset_rollout_prefix_cache()  # type: ignore[attr-defined]
                    lora_snapshots.append(
                        tuple(runtime.vllm_lora_snapshot())  # type: ignore[attr-defined]
                    )
                    base_snapshots.append(
                        tuple(runtime.vllm_base_fingerprint())  # type: ignore[attr-defined]
                    )
                    generation_rounds.append(
                        tuple(runtime.generate(probes))  # type: ignore[attr-defined]
                    )
                    if progress is not None:
                        progress(label, f"generation-round-{round_index + 1}-complete")
                lora_snapshots.append(
                    tuple(runtime.vllm_lora_snapshot())  # type: ignore[attr-defined]
                )
                base_snapshots.append(
                    tuple(runtime.vllm_base_fingerprint())  # type: ignore[attr-defined]
                )
            samples.append(
                RuntimeReproducibilitySample(
                    label=label,
                    checkpoint_loaded=load_checkpoint,
                    load_reports=load_reports,
                    actor_snapshots=actor_snapshots,
                    infoskill_snapshots=infoskill_snapshots,
                    vllm_lora_snapshots=tuple(lora_snapshots),
                    vllm_base_snapshots=tuple(base_snapshots),
                    generation_rounds=tuple(generation_rounds),
                )
            )
        finally:
            runtime.close()  # type: ignore[attr-defined]
            if progress is not None:
                progress(label, "closed")
    return tuple(samples)


def build_m1_reproducibility_report(
    samples: Sequence[RuntimeReproducibilitySample],
    *,
    logprob_tolerance: float = 1e-7,
) -> dict[str, object]:
    """Classify the first non-reproducible checkpoint inference boundary."""

    by_label = {sample.label: sample for sample in samples}
    required = {"checkpoint-a", "checkpoint-b", "base-a", "base-b"}
    if set(by_label) != required:
        raise ValueError("reproducibility samples must contain two checkpoint and two base runs")

    checkpoint_a = by_label["checkpoint-a"]
    checkpoint_b = by_label["checkpoint-b"]
    base_a = by_label["base-a"]
    base_b = by_label["base-b"]

    generation = {
        "checkpoint_a_within_runtime": compare_generation_results(
            checkpoint_a.generation_rounds[0],
            checkpoint_a.generation_rounds[1],
            logprob_tolerance=logprob_tolerance,
        ),
        "checkpoint_b_within_runtime": compare_generation_results(
            checkpoint_b.generation_rounds[0],
            checkpoint_b.generation_rounds[1],
            logprob_tolerance=logprob_tolerance,
        ),
        "checkpoint_across_runtimes": compare_generation_results(
            checkpoint_a.generation_rounds[0],
            checkpoint_b.generation_rounds[0],
            logprob_tolerance=logprob_tolerance,
        ),
        "base_a_within_runtime": compare_generation_results(
            base_a.generation_rounds[0],
            base_a.generation_rounds[1],
            logprob_tolerance=logprob_tolerance,
        ),
        "base_b_within_runtime": compare_generation_results(
            base_b.generation_rounds[0],
            base_b.generation_rounds[1],
            logprob_tolerance=logprob_tolerance,
        ),
        "base_across_runtimes": compare_generation_results(
            base_a.generation_rounds[0],
            base_b.generation_rounds[0],
            logprob_tolerance=logprob_tolerance,
        ),
    }
    exact_generation = {
        name: _generation_is_exact(comparison)
        for name, comparison in generation.items()
    }

    actor_matches_checkpoint = all(
        snapshot.get("exact") is True
        for sample in (checkpoint_a, checkpoint_b)
        for snapshot in sample.actor_snapshots
    ) and all(sample.actor_snapshots for sample in (checkpoint_a, checkpoint_b))
    actor_a_hashes = _rank_hashes(
        checkpoint_a.actor_snapshots,
        ("actual_summary", "tensor_payload_sha256"),
    )
    actor_b_hashes = _rank_hashes(
        checkpoint_b.actor_snapshots,
        ("actual_summary", "tensor_payload_sha256"),
    )
    actor_cross_runtime = (
        _fingerprints_are_complete(actor_a_hashes)
        and actor_a_hashes == actor_b_hashes
    )
    infoskill_a_hashes = _rank_hashes(
        checkpoint_a.infoskill_snapshots,
        ("summary", "tensor_payload_sha256"),
    )
    infoskill_b_hashes = _rank_hashes(
        checkpoint_b.infoskill_snapshots,
        ("summary", "tensor_payload_sha256"),
    )
    infoskill_cross_runtime = (
        _fingerprints_are_complete(infoskill_a_hashes)
        and infoskill_a_hashes == infoskill_b_hashes
    )
    checkpoint_registry_a_hashes = _vllm_registered_lora_hashes(
        checkpoint_a.vllm_lora_snapshots[0]
    )
    checkpoint_registry_b_hashes = _vllm_registered_lora_hashes(
        checkpoint_b.vllm_lora_snapshots[0]
    )
    checkpoint_registry_cross_runtime = (
        _fingerprints_are_complete(checkpoint_registry_a_hashes)
        and checkpoint_registry_a_hashes == checkpoint_registry_b_hashes
    )
    checkpoint_registry_stable = all(
        _snapshot_series_is_stable(
            sample.vllm_lora_snapshots,
            _vllm_registered_lora_hashes,
        )
        for sample in (checkpoint_a, checkpoint_b)
    )
    checkpoint_slots_a_hashes = _vllm_active_slot_hashes(
        checkpoint_a.vllm_lora_snapshots[0]
    )
    checkpoint_slots_b_hashes = _vllm_active_slot_hashes(
        checkpoint_b.vllm_lora_snapshots[0]
    )
    checkpoint_slots_cross_runtime = (
        _fingerprints_are_complete(checkpoint_slots_a_hashes)
        and checkpoint_slots_a_hashes == checkpoint_slots_b_hashes
    )
    checkpoint_slots_stable = all(
        _snapshot_series_is_stable(
            sample.vllm_lora_snapshots,
            _vllm_active_slot_hashes,
        )
        for sample in (checkpoint_a, checkpoint_b)
    )
    checkpoint_base_a_hashes = _rank_hashes(
        checkpoint_a.vllm_base_snapshots[0],
        ("summary", "tensor_payload_sha256"),
    )
    checkpoint_base_b_hashes = _rank_hashes(
        checkpoint_b.vllm_base_snapshots[0],
        ("summary", "tensor_payload_sha256"),
    )
    checkpoint_base_cross_runtime = (
        _has_selected_base_tensors(checkpoint_a.vllm_base_snapshots[0])
        and _has_selected_base_tensors(checkpoint_b.vllm_base_snapshots[0])
        and _fingerprints_are_complete(checkpoint_base_a_hashes)
        and checkpoint_base_a_hashes == checkpoint_base_b_hashes
    )
    base_a_hashes = _rank_hashes(
        base_a.vllm_base_snapshots[0],
        ("summary", "tensor_payload_sha256"),
    )
    base_b_hashes = _rank_hashes(
        base_b.vllm_base_snapshots[0],
        ("summary", "tensor_payload_sha256"),
    )
    base_control_weights_cross_runtime = (
        _has_selected_base_tensors(base_a.vllm_base_snapshots[0])
        and _has_selected_base_tensors(base_b.vllm_base_snapshots[0])
        and _fingerprints_are_complete(base_a_hashes)
        and base_a_hashes == base_b_hashes
    )
    checkpoint_load_reports_complete = all(
        sample.load_reports
        and all(
            report.get("lora_state_loaded") is True
            and report.get("infoskill_state_loaded") is True
            for report in sample.load_reports
        )
        for sample in (checkpoint_a, checkpoint_b)
    )
    base_controls_have_zero_lora_b = all(
        _vllm_lora_b_is_zero(sample.vllm_lora_snapshots[0])
        for sample in (base_a, base_b)
    )

    checks = {
        "checkpoint_load_reports_complete": checkpoint_load_reports_complete,
        "base_controls_have_zero_lora_b": base_controls_have_zero_lora_b,
        "actor_matches_checkpoint_on_all_ranks": actor_matches_checkpoint,
        "actor_lora_exact_across_runtimes": actor_cross_runtime,
        "infoskill_modules_exact_across_runtimes": infoskill_cross_runtime,
        "vllm_registered_lora_exact_across_runtimes": (
            checkpoint_registry_cross_runtime
        ),
        "vllm_registered_lora_immutable_within_runtimes": (
            checkpoint_registry_stable
        ),
        "vllm_active_gpu_slots_exact_across_runtimes": (
            checkpoint_slots_cross_runtime
        ),
        "vllm_active_gpu_slots_immutable_within_runtimes": (
            checkpoint_slots_stable
        ),
        "checkpoint_vllm_base_fingerprint_exact_across_runtimes": (
            checkpoint_base_cross_runtime
        ),
        "base_control_vllm_base_fingerprint_exact_across_runtimes": (
            base_control_weights_cross_runtime
        ),
        **exact_generation,
    }
    classification = _classify(checks)
    return {
        "schema_version": 1,
        "classification": classification,
        "reproducible": classification == "reproducible_under_probe",
        "checks": checks,
        "generation_comparisons": generation,
        "runtime_samples": [_sample_payload(sample) for sample in samples],
    }


def _classify(checks: Mapping[str, bool]) -> str:
    if not checks["checkpoint_load_reports_complete"]:
        return "checkpoint_load_incomplete"
    if not checks["base_controls_have_zero_lora_b"]:
        return "base_control_not_lora_zero"
    if not checks["actor_matches_checkpoint_on_all_ranks"]:
        return "checkpoint_to_fsdp_mismatch"
    if not checks["actor_lora_exact_across_runtimes"]:
        return "checkpoint_to_fsdp_nondeterminism"
    if not checks["infoskill_modules_exact_across_runtimes"]:
        return "checkpoint_to_infoskill_nondeterminism"
    if not (
        checks["vllm_registered_lora_exact_across_runtimes"]
        and checks["vllm_registered_lora_immutable_within_runtimes"]
    ):
        return "fsdp_to_vllm_registry_sync_nondeterminism"
    if not (
        checks["vllm_active_gpu_slots_exact_across_runtimes"]
        and checks["vllm_active_gpu_slots_immutable_within_runtimes"]
    ):
        return "vllm_registry_to_gpu_slot_sync_nondeterminism"
    if not (
        checks["checkpoint_vllm_base_fingerprint_exact_across_runtimes"]
        and checks["base_control_vllm_base_fingerprint_exact_across_runtimes"]
    ):
        return "vllm_base_weight_nondeterminism"
    if not checks["base_across_runtimes"]:
        return "base_or_hybrid_runtime_generation_nondeterminism"
    if not checks["checkpoint_across_runtimes"]:
        return "vllm_lora_execution_nondeterminism"
    within = (
        "checkpoint_a_within_runtime",
        "checkpoint_b_within_runtime",
        "base_a_within_runtime",
        "base_b_within_runtime",
    )
    if not all(checks[name] for name in within):
        return "within_runtime_generation_nondeterminism"
    return "reproducible_under_probe"


def _generation_is_exact(comparison: Mapping[str, object]) -> bool:
    return (
        comparison.get("tokens_exact") is True
        and int(comparison.get("changed_logprob_count", -1)) == 0
    )


def _rank_hashes(
    snapshots: Sequence[Mapping[str, object]],
    path: tuple[str, ...],
) -> dict[int, object]:
    return {
        int(snapshot["rank"]): _nested(snapshot, path)
        for snapshot in snapshots
    }


def _vllm_registered_lora_hashes(
    snapshots: Sequence[Mapping[str, object]],
) -> dict[int, object]:
    return {
        int(snapshot["rank"]): _single_registered_adapter_hash(snapshot)
        for snapshot in snapshots
    }


def _vllm_active_slot_hashes(
    snapshots: Sequence[Mapping[str, object]],
) -> dict[int, object]:
    result: dict[int, object] = {}
    for snapshot in snapshots:
        active_slot = _nested(
            snapshot,
            ("active_gpu_slot_summary", "tensor_payload_sha256"),
        )
        slots = snapshot.get("active_gpu_slots")
        if active_slot is None or not isinstance(slots, Mapping) or not slots:
            result[int(snapshot["rank"])] = None
        else:
            result[int(snapshot["rank"])] = {
                "payload": active_slot,
                "slot_indices": sorted(int(value) for value in slots.values()),
            }
    return result


def _fingerprints_are_complete(fingerprints: Mapping[int, object]) -> bool:
    return bool(fingerprints) and all(value is not None for value in fingerprints.values())


def _single_registered_adapter_hash(snapshot: Mapping[str, object]) -> object:
    adapters = snapshot.get("adapters")
    if not isinstance(adapters, Mapping) or len(adapters) != 1:
        return None
    adapter = next(iter(adapters.values()))
    return _nested(
        adapter if isinstance(adapter, Mapping) else {},
        ("summary", "tensor_payload_sha256"),
    )


def _vllm_lora_b_is_zero(
    snapshots: Sequence[Mapping[str, object]],
) -> bool:
    if not snapshots:
        return False
    for snapshot in snapshots:
        adapters = snapshot.get("adapters")
        if not isinstance(adapters, Mapping) or len(adapters) > 1:
            return False
        active_nonzero = _nested(
            snapshot,
            (
                "active_gpu_slot_summary",
                "partitions",
                "lora_b",
                "nonzero_count",
            ),
        )
        if active_nonzero != 0:
            return False
        if not adapters:
            slots = snapshot.get("active_gpu_slots")
            if (
                snapshot.get("registered_adapter_ids") != []
                or snapshot.get("active_adapter_ids") != []
                or not isinstance(slots, Mapping)
                or slots
            ):
                return False
            continue
        adapter = next(iter(adapters.values()))
        nonzero = _nested(
            adapter if isinstance(adapter, Mapping) else {},
            ("summary", "partitions", "lora_b", "nonzero_count"),
        )
        if nonzero != 0:
            return False
    return True


def _has_selected_base_tensors(
    snapshots: Sequence[Mapping[str, object]],
) -> bool:
    return bool(snapshots) and all(
        isinstance(snapshot.get("selected_tensor_names"), list)
        and bool(snapshot["selected_tensor_names"])
        for snapshot in snapshots
    )


def _snapshot_series_is_stable(
    series: Sequence[Sequence[Mapping[str, object]]],
    fingerprint: Callable[[Sequence[Mapping[str, object]]], object],
) -> bool:
    values = [fingerprint(snapshot) for snapshot in series]
    return bool(values) and all(value == values[0] for value in values[1:])


def _nested(value: Mapping[str, object], path: tuple[str, ...]) -> object:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _sample_payload(sample: RuntimeReproducibilitySample) -> dict[str, object]:
    return {
        "label": sample.label,
        "checkpoint_loaded": sample.checkpoint_loaded,
        "load_reports": list(sample.load_reports),
        "actor_snapshots": list(sample.actor_snapshots),
        "infoskill_snapshots": list(sample.infoskill_snapshots),
        "vllm_lora_snapshots": [list(value) for value in sample.vllm_lora_snapshots],
        "vllm_base_snapshots": [list(value) for value in sample.vllm_base_snapshots],
        "generation_rounds": [
            [asdict(result) for result in results]
            for results in sample.generation_rounds
        ],
    }
