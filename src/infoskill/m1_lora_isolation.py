"""Bounded, non-reportable M1 inference-isolation matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from infoskill.checkpoint_effect import compare_generation_results
from infoskill.m1_reproducibility import (
    _rank_hashes,
    _single_registered_adapter_hash,
    _vllm_active_slot_hashes,
    _vllm_lora_b_is_zero,
)
from infoskill.rollout import GenerationRequest, GenerationResult


@dataclass(frozen=True, slots=True)
class IsolationCell:
    name: str
    requests: tuple[GenerationRequest, ...]
    prefix_enabled: bool
    request_count: int


@dataclass(frozen=True, slots=True)
class IsolationCellSample:
    name: str
    input_digests_before: tuple[str, ...]
    input_digests_after: tuple[str, ...]
    worker_input_snapshots: tuple[tuple[Mapping[str, object], ...], ...]
    generation_rounds: tuple[tuple[GenerationResult, ...], ...]
    boundary_rounds: tuple[tuple[Mapping[str, object], ...], ...] = ()


@dataclass(frozen=True, slots=True)
class IsolationRuntimeSample:
    label: str
    graph_enabled: bool
    checkpoint_loaded: bool
    load_reports: tuple[Mapping[str, object], ...]
    actor_snapshots: tuple[Mapping[str, object], ...]
    infoskill_snapshots_before: tuple[Mapping[str, object], ...]
    infoskill_snapshots_after: tuple[Mapping[str, object], ...]
    vllm_lora_snapshots: tuple[tuple[Mapping[str, object], ...], ...]
    vllm_base_snapshots: tuple[tuple[Mapping[str, object], ...], ...]
    cells: tuple[IsolationCellSample, ...]


def build_m1_isolation_cells(
    probes: tuple[GenerationRequest, ...],
) -> tuple[IsolationCell, ...]:
    """Keep the same first two requests in padded and full three-rank cells."""

    if len(probes) != 3:
        raise ValueError("M1 isolation requires exactly three fixed probes")
    if any(probe.soft_prefix is None for probe in probes):
        raise ValueError("M1 isolation source probes must carry soft prefixes")
    token_probes = tuple(replace(probe, soft_prefix=None) for probe in probes)
    return (
        IsolationCell("hybrid-full3", probes, True, 3),
        IsolationCell("hybrid-padded2", probes[:2], True, 2),
        IsolationCell("token-full3", token_probes, False, 3),
        IsolationCell("token-padded2", token_probes[:2], False, 2),
    )


def summarize_cell_repetitions(
    rounds: Sequence[Sequence[GenerationResult]],
    *,
    logprob_tolerance: float = 1e-7,
) -> dict[str, object]:
    """Describe deterministic replay failures without comparing post-branch logits."""

    if len(rounds) != 3:
        raise ValueError("one isolation cell requires three generation rounds")
    pairwise: dict[str, object] = {}
    first_divergence: int | None = None
    changed_before_divergence = 0
    max_before_divergence = 0.0
    for left, right in ((0, 1), (0, 2), (1, 2)):
        comparison = compare_generation_results(
            rounds[left],
            rounds[right],
            logprob_tolerance=logprob_tolerance,
        )
        pairwise[f"{left}-{right}"] = comparison
        right_by_id = {result.request_id: result for result in rounds[right]}
        for baseline in rounds[left]:
            candidate = right_by_id.get(baseline.request_id)
            if candidate is None:
                continue
            common = min(len(baseline.token_ids), len(candidate.token_ids))
            divergence = next(
                (
                    position
                    for position in range(common)
                    if baseline.token_ids[position] != candidate.token_ids[position]
                ),
                None,
            )
            if divergence is None and len(baseline.token_ids) != len(candidate.token_ids):
                divergence = common
            if divergence is None:
                continue
            if first_divergence is None or divergence < first_divergence:
                first_divergence = divergence
                changed_before_divergence = 0
                max_before_divergence = 0.0
            if divergence == first_divergence:
                errors = [
                    abs(baseline.token_logprobs[position] - candidate.token_logprobs[position])
                    for position in range(divergence)
                ]
                changed_before_divergence = max(
                    changed_before_divergence,
                    sum(error > logprob_tolerance for error in errors),
                )
                max_before_divergence = max(
                    max_before_divergence,
                    max(errors, default=0.0),
                )
    return {
        "reproducible": all(
            comparison["tokens_exact"] is True
            and comparison["changed_logprob_count"] == 0
            for comparison in pairwise.values()
        ),
        "first_token_divergence": first_divergence,
        "changed_logprobs_before_divergence": changed_before_divergence,
        "max_logprob_abs_error_before_divergence": max_before_divergence,
        "pairwise": pairwise,
    }


def _prefix_digest(prefix: object | None) -> str | None:
    if prefix is None:
        return None
    if isinstance(prefix, (bytes, bytearray)):
        payload = bytes(prefix)
        kind = "bytes"
        shape: tuple[int, ...] = (len(payload),)
    else:
        import torch

        if not isinstance(prefix, torch.Tensor):
            raise TypeError("isolation soft prefix must be a tensor or test bytes")
        detached = prefix.detach().to("cpu").contiguous()
        payload = detached.view(torch.uint8).numpy().tobytes()
        kind = str(detached.dtype)
        shape = tuple(int(value) for value in detached.shape)
    digest = hashlib.sha256()
    digest.update(json.dumps({"kind": kind, "shape": shape}, sort_keys=True).encode())
    digest.update(payload)
    return digest.hexdigest()


def request_batch_digest(requests: Sequence[GenerationRequest]) -> str:
    """Fingerprint stable driver-side inputs before and after every generation."""

    payload = [
        {
            "request_id": request.request_id,
            "user_message": request.user_message,
            "parameters": asdict(request.parameters),
            "seed": request.seed,
            "soft_prefix_sha256": _prefix_digest(request.soft_prefix),
        }
        for request in requests
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def collect_m1_isolation_samples(
    *,
    runtime_factory: Callable[[bool], object],
    checkpoint_runtime_directory: Path,
    probes: tuple[GenerationRequest, ...],
    progress: Callable[[str, str], None] | None = None,
    on_sample: Callable[[IsolationRuntimeSample], None] | None = None,
    capture_boundaries: bool = False,
) -> tuple[IsolationRuntimeSample, ...]:
    """Run the complete four-runtime matrix in one bounded GPU session."""

    cells = build_m1_isolation_cells(probes)
    samples = []
    for label, graph_enabled, load_checkpoint in (
        ("checkpoint-eager", False, True),
        ("base-eager", False, False),
        ("checkpoint-graph", True, True),
        ("base-graph", True, False),
    ):
        if progress is not None:
            progress(label, "initializing")
        runtime = runtime_factory(graph_enabled)
        try:
            load_reports: tuple[Mapping[str, object], ...] = ()
            actor_snapshots: tuple[Mapping[str, object], ...] = ()
            infoskill_before: tuple[Mapping[str, object], ...] = ()
            infoskill_after: tuple[Mapping[str, object], ...] = ()
            if load_checkpoint:
                load_reports = tuple(
                    runtime.load_portable_state(checkpoint_runtime_directory) or ()
                )
                actor_snapshots = tuple(
                    runtime.compare_portable_actor_state(checkpoint_runtime_directory)
                )
                infoskill_before = tuple(runtime.infoskill_module_snapshot())
                if progress is not None:
                    progress(label, "checkpoint-loaded-and-fingerprinted")

            cell_rows: dict[str, dict[str, list[object]]] = {
                cell.name: {"before": [], "after": [], "worker": [], "results": [], "boundary": []}
                for cell in cells
            }
            with runtime.rollout_session():
                if capture_boundaries:
                    runtime.begin_vllm_boundary_capture()
                lora_before = tuple(runtime.vllm_lora_snapshot())
                base_before = tuple(runtime.vllm_base_fingerprint())
                try:
                    for round_index in range(3):
                        order = cells if round_index % 2 == 0 else tuple(reversed(cells))
                        for cell in order:
                            runtime.reset_rollout_prefix_cache()
                            before = request_batch_digest(cell.requests)
                            results = tuple(runtime.generate(cell.requests))
                            worker = tuple(runtime.vllm_last_input_fingerprints())
                            boundary = (
                                tuple(runtime.take_vllm_boundary_rows())
                                if capture_boundaries else ()
                            )
                            after = request_batch_digest(cell.requests)
                            row = cell_rows[cell.name]
                            row["before"].append(before)
                            row["results"].append(results)
                            row["worker"].append(worker)
                            row["boundary"].append(boundary)
                            row["after"].append(after)
                            if progress is not None:
                                progress(label, f"{cell.name}-round-{round_index + 1}-complete")
                    lora_after = tuple(runtime.vllm_lora_snapshot())
                    base_after = tuple(runtime.vllm_base_fingerprint())
                finally:
                    if capture_boundaries:
                        runtime.end_vllm_boundary_capture()
            if load_checkpoint:
                infoskill_after = tuple(runtime.infoskill_module_snapshot())

            sample = IsolationRuntimeSample(
                label=label,
                graph_enabled=graph_enabled,
                checkpoint_loaded=load_checkpoint,
                load_reports=load_reports,
                actor_snapshots=actor_snapshots,
                infoskill_snapshots_before=infoskill_before,
                infoskill_snapshots_after=infoskill_after,
                vllm_lora_snapshots=(lora_before, lora_after),
                vllm_base_snapshots=(base_before, base_after),
                cells=tuple(
                    IsolationCellSample(
                        name=cell.name,
                        input_digests_before=tuple(cell_rows[cell.name]["before"]),
                        input_digests_after=tuple(cell_rows[cell.name]["after"]),
                        worker_input_snapshots=tuple(cell_rows[cell.name]["worker"]),
                        generation_rounds=tuple(cell_rows[cell.name]["results"]),
                        boundary_rounds=tuple(
                            tuple(
                                row
                                for rank_report in reports
                                for row in rank_report["rows"]
                            )
                            for reports in cell_rows[cell.name]["boundary"]
                        ) if capture_boundaries else (),
                    )
                    for cell in cells
                ),
            )
            samples.append(sample)
            if on_sample is not None:
                on_sample(sample)
        finally:
            runtime.close()
            if progress is not None:
                progress(label, "closed")
    return tuple(samples)


def build_m1_lora_isolation_report(
    samples: Sequence[IsolationRuntimeSample],
) -> dict[str, object]:
    """Separate measured input/weight drift from inference execution drift."""
    if len(samples) != 4:
        raise ValueError("complete isolation report requires four runtime samples")
    by_label = {sample.label: sample for sample in samples}
    expected = {"checkpoint-eager", "base-eager", "checkpoint-graph", "base-graph"}
    if set(by_label) != expected:
        raise ValueError("isolation runtime labels are incomplete or duplicated")
    cells: dict[str, dict[str, object]] = {}
    for sample in samples:
        cell_reports = {}
        for cell in sample.cells:
            repetitions = summarize_cell_repetitions(cell.generation_rounds)
            before = cell.input_digests_before
            after = cell.input_digests_after
            worker = cell.worker_input_snapshots
            def worker_digest(snapshot: tuple[Mapping[str, object], ...]) -> dict[int, object]:
                return {
                    int(item["rank"]): item.get("calls")
                    for item in snapshot
                }
            driver_stable = len(set(before + after)) == 1
            worker_stable = bool(worker) and all(
                worker_digest(item) == worker_digest(worker[0]) for item in worker[1:]
            )
            cell_reports[cell.name] = {
                **repetitions,
                "driver_input_stable": driver_stable,
                "vllm_input_stable": worker_stable,
                "input_digests_before": list(before),
                "input_digests_after": list(after),
                "worker_input_snapshots": [list(item) for item in worker],
            }
        cells[sample.label] = cell_reports
    checkpoint_samples = (by_label["checkpoint-eager"], by_label["checkpoint-graph"])
    base_samples = (by_label["base-eager"], by_label["base-graph"])
    actor_exact = all(
        sample.load_reports
        and all(
            item.get("lora_state_loaded") is True
            and item.get("infoskill_state_loaded") is True
            for item in sample.load_reports
        )
        and sample.actor_snapshots
        and all(item.get("exact") is True for item in sample.actor_snapshots)
        for sample in checkpoint_samples
    )
    def snapshot_series_stable(sample: IsolationRuntimeSample, selector) -> bool:
        return selector(sample.vllm_lora_snapshots[0]) == selector(sample.vllm_lora_snapshots[1])
    registered = lambda rows: {
        int(row["rank"]): _single_registered_adapter_hash(row) for row in rows
    }
    registry_stable = all(snapshot_series_stable(item, registered) for item in checkpoint_samples)
    slots_stable = all(
        snapshot_series_stable(item, _vllm_active_slot_hashes)
        for item in checkpoint_samples
    )
    infoskill_stable = all(
        _rank_hashes(item.infoskill_snapshots_before, ("summary", "tensor_payload_sha256"))
        == _rank_hashes(item.infoskill_snapshots_after, ("summary", "tensor_payload_sha256"))
        for item in checkpoint_samples
    )
    base_zero = all(_vllm_lora_b_is_zero(item.vllm_lora_snapshots[0]) for item in base_samples)
    base_weights_stable = all(
        _rank_hashes(item.vllm_base_snapshots[0], ("summary", "tensor_payload_sha256"))
        == _rank_hashes(item.vllm_base_snapshots[1], ("summary", "tensor_payload_sha256"))
        for item in samples
    )
    checkpoint_registry_cross_mode = (
        registered(checkpoint_samples[0].vllm_lora_snapshots[0])
        == registered(checkpoint_samples[1].vllm_lora_snapshots[0])
    )
    checkpoint_slots_cross_mode = (
        _vllm_active_slot_hashes(checkpoint_samples[0].vllm_lora_snapshots[0])
        == _vllm_active_slot_hashes(checkpoint_samples[1].vllm_lora_snapshots[0])
    )
    infoskill_cross_mode = (
        _rank_hashes(checkpoint_samples[0].infoskill_snapshots_before, ("summary", "tensor_payload_sha256"))
        == _rank_hashes(checkpoint_samples[1].infoskill_snapshots_before, ("summary", "tensor_payload_sha256"))
    )
    base_weights_cross_mode = all(
        _rank_hashes(group[0].vllm_base_snapshots[0], ("summary", "tensor_payload_sha256"))
        == _rank_hashes(group[1].vllm_base_snapshots[0], ("summary", "tensor_payload_sha256"))
        for group in (checkpoint_samples, base_samples)
    )
    rank_count_exact = all(
        len(item.vllm_lora_snapshots[0]) == 3
        and len(item.vllm_base_snapshots[0]) == 3
        for item in samples
    )
    controls = {
        "checkpoint_actor_exact": bool(actor_exact),
        "infoskill_modules_stable": infoskill_stable,
        "registered_lora_stable": registry_stable,
        "active_gpu_slots_stable": slots_stable,
        "base_controls_zero_lora": base_zero,
        "base_weights_stable": base_weights_stable,
        "checkpoint_registry_cross_mode": checkpoint_registry_cross_mode,
        "checkpoint_slots_cross_mode": checkpoint_slots_cross_mode,
        "infoskill_modules_cross_mode": infoskill_cross_mode,
        "base_weights_cross_mode": base_weights_cross_mode,
        "three_ranks_fingerprinted": rank_count_exact,
        "all_driver_inputs_stable": all(
            cell["driver_input_stable"] for group in cells.values() for cell in group.values()
        ),
        "all_vllm_inputs_stable": all(
            cell["vllm_input_stable"] for group in cells.values() for cell in group.values()
        ),
    }
    drift = {
        label: sorted(name for name, cell in group.items() if not cell["reproducible"])
        for label, group in cells.items()
    }
    checkpoint_drift = set(drift["checkpoint-eager"] + drift["checkpoint-graph"])
    base_drift = set(drift["base-eager"] + drift["base-graph"])
    cross_mode = {}
    for kind in ("checkpoint", "base"):
        eager = by_label[f"{kind}-eager"]
        graph = by_label[f"{kind}-graph"]
        graph_by_name = {cell.name: cell for cell in graph.cells}
        cross_mode[kind] = {
            cell.name: {
                "generation_rounds": [
                    compare_generation_results(
                        left, right, logprob_tolerance=1e-7
                    )
                    for left, right in zip(
                        cell.generation_rounds,
                        graph_by_name[cell.name].generation_rounds,
                    )
                ],
                "driver_input_exact": (
                    cell.input_digests_before
                    == graph_by_name[cell.name].input_digests_before
                ),
                "worker_input_exact": (
                    cell.worker_input_snapshots
                    == graph_by_name[cell.name].worker_input_snapshots
                ),
            }
            for cell in eager.cells
        }
    if not all(controls.values()):
        classification = "control_failure_investigate_inputs_or_weights"
    elif base_drift:
        classification = "base_generation_drift"
    elif not checkpoint_drift:
        classification = "no_drift_under_fixed_probe"
    elif checkpoint_drift <= {"hybrid-padded2"}:
        classification = "hybrid_prefix_padding_sensitive_lora_drift"
    elif checkpoint_drift <= {"hybrid-full3", "hybrid-padded2"}:
        classification = "hybrid_prefix_lora_drift"
    elif drift["checkpoint-graph"] and not drift["checkpoint-eager"]:
        classification = "graph_sensitive_lora_drift"
    else:
        classification = "active_lora_inference_drift_across_cells"
    return {
        "schema_version": 1,
        "classification": classification,
        "controls": controls,
        "drift_cells": drift,
        "cross_mode": cross_mode,
        "cells": cells,
        "runtime_samples": [asdict(item) for item in samples],
        "interpretation_limit": (
            "Fixed synthetic probes localize observed execution drift; "
            "they do not estimate valid_seen success or prove a kernel root cause."
        ),
    }
