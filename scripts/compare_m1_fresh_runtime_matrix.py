#!/usr/bin/env python3
"""Aggregate the three-cell M1 fresh-runtime reproducibility matrix."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _check(report: Mapping[str, object], name: str) -> bool:
    checks = report.get("checks")
    return bool(isinstance(checks, Mapping) and checks.get(name) is True)


def _execution(report: Mapping[str, object], name: str) -> object:
    execution = report.get("execution")
    return execution.get(name) if isinstance(execution, Mapping) else None


def _checkpoint(report: Mapping[str, object], name: str) -> object:
    checkpoint = report.get("checkpoint")
    return checkpoint.get(name) if isinstance(checkpoint, Mapping) else None


def compare_matrix(
    graph_split_k_one: Mapping[str, object],
    eager_split_k_one: Mapping[str, object],
    eager_reference_full: Mapping[str, object],
) -> dict[str, object]:
    reports = {
        "graph_split_k_one": graph_split_k_one,
        "eager_split_k_one": eager_split_k_one,
        "eager_reference_full": eager_reference_full,
    }
    common_controls = {
        "same_checkpoint_directory": len({
            _checkpoint(report, "directory") for report in reports.values()
        }) == 1,
        "same_checkpoint_provenance": len({
            _checkpoint(report, "provenance_sha256")
            for report in reports.values()
        }) == 1,
        "same_probe": len({
            json.dumps(report.get("probe"), sort_keys=True)
            for report in reports.values()
        }) == 1,
        "input_fingerprints_exact_in_all_cells": all(
            _check(report, "input_fingerprints_exact")
            for report in reports.values()
        ),
        "checkpoint_load_complete_in_all_cells": all(
            _check(report, "checkpoint_load_reports_complete")
            for report in reports.values()
        ),
        "base_controls_have_zero_lora_b_in_all_cells": all(
            _check(report, "base_controls_have_zero_lora_b")
            for report in reports.values()
        ),
        "actor_matches_checkpoint_in_all_cells": all(
            _check(report, "actor_matches_checkpoint_on_all_ranks")
            for report in reports.values()
        ),
        "actor_lora_exact_in_all_cells": all(
            _check(report, "actor_lora_exact_across_runtimes")
            for report in reports.values()
        ),
        "infoskill_modules_exact_in_all_cells": all(
            _check(report, "infoskill_modules_exact_across_runtimes")
            for report in reports.values()
        ),
        "vllm_lora_registry_exact_in_all_cells": all(
            _check(report, "vllm_registered_lora_exact_across_runtimes")
            and _check(report, "vllm_active_gpu_slots_exact_across_runtimes")
            and _check(report, "vllm_registered_lora_immutable_within_runtimes")
            and _check(report, "vllm_active_gpu_slots_immutable_within_runtimes")
            for report in reports.values()
        ),
        "vllm_base_weights_exact_in_all_cells": all(
            _check(report, "checkpoint_vllm_base_fingerprint_exact_across_runtimes")
            and _check(report, "base_control_vllm_base_fingerprint_exact_across_runtimes")
            for report in reports.values()
        ),
        "graph_cell_uses_graph": (
            _execution(graph_split_k_one, "hybrid_prefix_cuda_graph") is True
        ),
        "eager_cells_disable_graph": (
            _execution(eager_split_k_one, "hybrid_prefix_cuda_graph") is False
            and _execution(eager_reference_full, "hybrid_prefix_cuda_graph") is False
        ),
        "split_k_one_verified_in_native_cells": (
            _check(graph_split_k_one, "split_k_one_verified_for_all_runtimes")
            and _check(eager_split_k_one, "split_k_one_verified_for_all_runtimes")
        ),
        "reference_cell_is_not_split_k_one": (
            _execution(eager_reference_full, "lora_shrink_split_k_one") is False
            and _execution(eager_reference_full, "lora_kernel_intervention")
            == "reference_full"
        ),
    }
    generation = {
        name: {
            "classification": report.get("classification"),
            "checkpoint_within_runtime": (
                _check(report, "checkpoint_a_within_runtime")
                and _check(report, "checkpoint_b_within_runtime")
            ),
            "checkpoint_across_runtimes": _check(
                report,
                "checkpoint_across_runtimes",
            ),
            "base_within_runtime": (
                _check(report, "base_a_within_runtime")
                and _check(report, "base_b_within_runtime")
            ),
            "base_across_runtimes": _check(report, "base_across_runtimes"),
        }
        for name, report in reports.items()
    }
    controls_valid = all(common_controls.values())
    graph = generation["graph_split_k_one"]
    eager = generation["eager_split_k_one"]
    reference = generation["eager_reference_full"]
    graph_checkpoint_exact = (
        graph["checkpoint_within_runtime"]
        and graph["checkpoint_across_runtimes"]
    )
    eager_checkpoint_exact = (
        eager["checkpoint_within_runtime"]
        and eager["checkpoint_across_runtimes"]
    )
    reference_checkpoint_exact = (
        reference["checkpoint_within_runtime"]
        and reference["checkpoint_across_runtimes"]
    )
    base_exact = all(
        cell["base_within_runtime"] and cell["base_across_runtimes"]
        for cell in generation.values()
    )
    if not controls_valid:
        classification = "invalid_fresh_runtime_matrix_controls"
    elif not base_exact:
        if (
            eager["base_within_runtime"]
            and eager["base_across_runtimes"]
            and not (
                graph["base_within_runtime"]
                and graph["base_across_runtimes"]
            )
        ):
            classification = "cuda_graph_base_or_hybrid_runtime_nondeterminism"
        else:
            classification = "base_or_hybrid_runtime_nondeterminism"
    elif graph_checkpoint_exact and eager_checkpoint_exact:
        classification = "split_k_one_reproducible_across_fresh_runtimes"
    elif not graph_checkpoint_exact and eager_checkpoint_exact:
        classification = "cuda_graph_specific_checkpoint_nondeterminism"
    elif graph_checkpoint_exact and not eager_checkpoint_exact:
        classification = "eager_specific_checkpoint_nondeterminism"
    elif not eager_checkpoint_exact:
        if reference_checkpoint_exact:
            classification = "native_lora_kernel_nondeterminism_after_split_k_one"
        else:
            classification = "lora_request_or_non_kernel_execution_nondeterminism"
    else:
        classification = "inconsistent_fresh_runtime_matrix"
    passed = (
        controls_valid
        and base_exact
        and graph_checkpoint_exact
        and eager_checkpoint_exact
    )
    return {
        "schema_version": 1,
        "classification": classification,
        "controls_valid": controls_valid,
        "passed": passed,
        "controls": common_controls,
        "generation": generation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_split_k_one", type=Path)
    parser.add_argument("eager_split_k_one", type=Path)
    parser.add_argument("eager_reference_full", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_matrix(
        _read(args.graph_split_k_one),
        _read(args.eager_split_k_one),
        _read(args.eager_reference_full),
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    # A diagnostic classification is a successful run when its controls are valid.
    return 0 if report["controls_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
