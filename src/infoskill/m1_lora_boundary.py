"""Diagnostic-only capture of pinned-vLLM logits and sampling boundaries.

The capture is deliberately bounded: no vocabulary-sized tensor is persisted.
All comparisons use the same request history, so post-branch logits are excluded.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence


def _history_digest(prompt: Sequence[int], output: Sequence[int]) -> str:
    payload = json.dumps([list(prompt), list(output)], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _logit_summary(logits: object) -> list[dict[str, object]]:
    import torch

    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise RuntimeError("boundary capture expected a 2-D logits tensor")
    if logits.shape[1] < 4:
        raise RuntimeError("boundary capture expected at least four vocabulary tokens")
    values, indices = torch.topk(logits.float(), k=4, dim=-1)
    norms = torch.logsumexp(logits.float(), dim=-1)
    cpu_values = values.detach().cpu().tolist()
    cpu_indices = indices.detach().cpu().tolist()
    cpu_norms = norms.detach().cpu().tolist()
    def encode(value: float) -> float | str:
        number = float(value)
        if math.isnan(number):
            raise RuntimeError("boundary capture found NaN logits")
        if math.isinf(number):
            return "+inf" if number > 0 else "-inf"
        return number

    return [
        {
            "top_token_ids": [int(value) for value in token_ids],
            "top_values": [encode(value) for value in top_values],
            "logsumexp": encode(norm),
        }
        for token_ids, top_values, norm in zip(cpu_indices, cpu_values, cpu_norms)
    ]


class VllmBoundaryCapture:
    """Scoped instance-method hooks; installed only inside a diagnostic session."""

    def __init__(self, model_runner: object, rank: int, *, max_rows: int = 4096):
        self.runner = model_runner
        self.rank = rank
        self.max_rows = max_rows
        self.rows: list[dict[str, object]] = []
        self.pending: list[dict[str, object]] = []
        self._original_compute = None
        self._original_sample = None
        self._original_sampler_sample = None

    def _histories(self, count: int) -> list[dict[str, object]]:
        req_ids = self.runner.input_batch.req_ids
        if len(req_ids) != count:
            raise RuntimeError(
                f"boundary logits/request rows differ: {count} vs {len(req_ids)}"
            )
        return [
            {
                "rank": self.rank,
                "history_sha256": _history_digest(
                    self.runner.requests[req_id].prompt_token_ids,
                    self.runner.requests[req_id].output_token_ids,
                ),
                "generated_position": len(
                    self.runner.requests[req_id].output_token_ids
                ),
                "batch_row": index,
                "batch_size": count,
            }
            for index, req_id in enumerate(req_ids)
        ]

    def install(self) -> None:
        if self._original_compute is not None:
            raise RuntimeError("boundary capture is already installed")
        model = self.runner.model
        sampler = model.sampler
        original_compute = model.compute_logits
        original_sample = model.sample
        original_sampler_sample = sampler.sample

        def compute(hidden_states, sampling_metadata):
            logits = original_compute(hidden_states, sampling_metadata)
            summaries = _logit_summary(logits)
            self.pending = self._histories(len(summaries))
            for row, summary in zip(self.pending, summaries):
                row["raw"] = summary
            return logits

        def sample_processed(logits, sampling_metadata):
            summaries = _logit_summary(logits)
            if len(summaries) != len(self.pending):
                raise RuntimeError("processed logits do not match pending raw logits")
            for row, summary in zip(self.pending, summaries):
                row["processed"] = summary
            return original_sampler_sample(logits, sampling_metadata)

        def sample(*, logits, sampling_metadata):
            output = original_sample(logits=logits, sampling_metadata=sampling_metadata)
            tokens = output.sampled_token_ids.detach().cpu().reshape(-1).tolist()
            logprobs = output.logprobs_tensors
            returned = (
                logprobs.logprobs.detach().cpu()[:, 0].tolist()
                if logprobs is not None else [None] * len(tokens)
            )
            if len(tokens) != len(self.pending):
                raise RuntimeError("sampled tokens do not match pending logits rows")
            if len(returned) != len(tokens):
                raise RuntimeError("returned logprobs do not match sampled tokens")
            if len(self.rows) + len(tokens) > self.max_rows:
                raise RuntimeError("boundary capture exceeded its fixed row limit")
            for row, token, value in zip(self.pending, tokens, returned):
                row["sampled_token_id"] = int(token)
                row["returned_logprob"] = None if value is None else float(value)
                self.rows.append(row)
            self.pending = []
            return output

        self._original_compute = original_compute
        self._original_sample = original_sample
        self._original_sampler_sample = original_sampler_sample
        try:
            model.compute_logits = compute
            sampler.sample = sample_processed
            model.sample = sample
        except Exception:
            self.remove()
            raise

    def take(self) -> list[dict[str, object]]:
        if self.pending:
            raise RuntimeError("boundary capture has unfinished sampling rows")
        rows, self.rows = self.rows, []
        return rows

    def remove(self) -> None:
        if self._original_compute is None:
            return
        model = self.runner.model
        sampler = model.sampler
        model.compute_logits = self._original_compute
        model.sample = self._original_sample
        sampler.sample = self._original_sampler_sample
        self._original_compute = None
        self._original_sample = None
        self._original_sampler_sample = None
        self.pending = []


def _summary_changed(left: Mapping[str, object], right: Mapping[str, object], tolerance: float) -> bool:
    if left.get("top_token_ids") != right.get("top_token_ids"):
        return True
    for key in ("top_values", "logsumexp"):
        a, b = left.get(key), right.get(key)
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b) or any(abs(float(x) - float(y)) > tolerance for x, y in zip(a, b)):
                return True
        elif a is not None and b is not None and abs(float(a) - float(b)) > tolerance:
            return True
    return False


def compare_boundary_rounds(
    rounds: Sequence[Sequence[Mapping[str, object]]], *, tolerance: float = 1e-6
) -> dict[str, object]:
    """Compare only common pre-branch histories, ranked by first changed seam."""

    if len(rounds) < 2:
        raise ValueError("boundary comparison needs at least two rounds")
    counts = {name: 0 for name in (
        "raw_logits", "processed_logits", "sampled_token", "returned_logprob"
    )}
    comparable = 0
    logprob_comparable = 0
    for left_index in range(len(rounds)):
        for right_index in range(left_index + 1, len(rounds)):
            right_rows = {
                (row.get("rank"), row["history_sha256"]): row
                for row in rounds[right_index]
            }
            for left in rounds[left_index]:
                right = right_rows.get((left.get("rank"), left["history_sha256"]))
                if right is None:
                    continue
                comparable += 1
                if _summary_changed(left["raw"], right["raw"], tolerance):
                    counts["raw_logits"] += 1
                if _summary_changed(left["processed"], right["processed"], tolerance):
                    counts["processed_logits"] += 1
                if left.get("sampled_token_id") != right.get("sampled_token_id"):
                    counts["sampled_token"] += 1
                a, b = left.get("returned_logprob"), right.get("returned_logprob")
                if a is not None and b is not None:
                    logprob_comparable += 1
                    if abs(float(a) - float(b)) > tolerance:
                        counts["returned_logprob"] += 1
    first = next((name for name, count in counts.items() if count), None)
    if comparable == 0:
        first = "insufficient_common_history"
    return {
        "comparable_rows": comparable,
        "returned_logprob_comparable_rows": logprob_comparable,
        "changed_row_counts": counts,
        "first_changed_boundary": first or "exact_at_captured_boundaries",
    }


def build_boundary_isolation_report(samples: Sequence[object]) -> dict[str, object]:
    """Report checkpoint/base × eager/Graph without attributing unsupported causes."""

    labels = {sample.label for sample in samples}
    required = {"checkpoint-eager", "base-eager", "checkpoint-graph", "base-graph"}
    if labels != required or len(samples) != 4:
        raise ValueError("boundary isolation needs four distinct runtime controls")
    cells = {
        sample.label: {
            cell.name: compare_boundary_rounds(cell.boundary_rounds)
            for cell in sample.cells
        }
        for sample in samples
    }
    if any(
        report["comparable_rows"] == 0
        for by_cell in cells.values() for report in by_cell.values()
    ):
        classification = "inconclusive_no_common_history"
    elif any(
        report["first_changed_boundary"] != "exact_at_captured_boundaries"
        for label, by_cell in cells.items() if label.startswith("base-")
        for report in by_cell.values()
    ):
        classification = "base_control_drift_blocks_lora_attribution"
    else:
        changed = {
            report["first_changed_boundary"]
            for label, by_cell in cells.items() if label.startswith("checkpoint-")
            for report in by_cell.values()
            if report["first_changed_boundary"] != "exact_at_captured_boundaries"
        }
        if "raw_logits" in changed:
            classification = "active_lora_model_logits_drift"
        elif "processed_logits" in changed:
            classification = "logits_processing_drift"
        elif "sampled_token" in changed:
            classification = "sampler_selection_drift"
        elif not all(
            report["returned_logprob_comparable_rows"] > 0
            for label, by_cell in cells.items() if label.startswith("checkpoint-")
            for report in by_cell.values()
        ):
            classification = "inconclusive_missing_returned_logprobs"
        elif not changed:
            classification = "no_drift_in_bounded_replay"
        else:
            classification = "logprob_gather_or_transport_drift"
    return {
        "schema_version": 1,
        "classification": classification,
        "scope": "fixed synthetic replay; not a 140-task success-rate result",
        "boundary_cells": cells,
    }
