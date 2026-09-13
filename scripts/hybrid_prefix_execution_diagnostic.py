#!/usr/bin/env python3
"""Isolate hybrid-prefix eager, Inductor, and CUDA Graph execution drift."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Sequence


REQUIRED_EXECUTION_MODES = (
    "eager",
    "persistent-eager",
    "compile-only",
    "cuda-graph",
)
EXECUTION_MODES = REQUIRED_EXECUTION_MODES + (
    "cuda-graph-eager-kernels",
    "dynamo-eager-native-kernels",
    "dynamo-eager-custom-kernels",
    "cuda-graph-custom-kernels",
)

FALLBACK_PROMPTS = (
    "You are standing in a room. Return one concise ALFWorld action.",
    "Goal: put the apple in the fridge. Return one concise action.",
    "Available actions: look, inventory, go to kitchen. Return one action.",
    "Goal: examine the mug under the lamp. Return one concise action.",
)


def _hybrid_prompt(
    prompt_token_ids: Sequence[int],
    prefix: object,
    placeholder_id: int,
) -> dict[str, object]:
    prefix_length = len(prefix)  # type: ignore[arg-type]
    return {
        "prompt_token_ids": [placeholder_id] * prefix_length
        + list(prompt_token_ids),
        "infoskill_prefix_embeds": prefix,
        "infoskill_prefix_mask": [True] * prefix_length
        + [False] * len(prompt_token_ids),
    }


def build_execution_rounds(
    cases: Sequence[dict[str, object]],
    *,
    placeholder_id: int,
) -> dict[str, list[dict[str, object]]]:
    """Build repeated and reordered rounds that expose stale graph inputs."""

    def requests(
        ordered_cases: Sequence[dict[str, object]], prefix_name: str
    ) -> list[dict[str, object]]:
        return [
            {
                "case_id": str(case["case_id"]),
                "prompt": _hybrid_prompt(
                    case["prompt_token_ids"],  # type: ignore[arg-type]
                    case[prefix_name],
                    placeholder_id,
                ),
            }
            for case in ordered_cases
        ]

    return {
        "prefix-a": requests(cases, "prefix_a"),
        "prefix-b": requests(cases, "prefix_b"),
        "prefix-a-repeat": requests(cases, "prefix_a"),
        "prefix-a-reverse": requests(list(reversed(cases)), "prefix_a"),
    }


def _flatten_rounds(report: dict[str, object]) -> dict[tuple[str, str], dict]:
    flattened = {}
    for round_name, cases in report["rounds"].items():  # type: ignore[union-attr]
        for case_id, result in cases.items():
            flattened[(str(round_name), str(case_id))] = result
    return flattened


def _compare_results(
    left: dict[str, object],
    right: dict[str, object],
    *,
    logprob_atol: float,
) -> dict[str, object]:
    left_flat = _flatten_rounds(left)
    right_flat = _flatten_rounds(right)
    keys = sorted(set(left_flat) | set(right_flat))
    missing = [key for key in keys if key not in left_flat or key not in right_flat]
    token_exact_count = 0
    first_token_match_count = 0
    aligned_logprob_errors: list[float] = []
    first_differences: list[dict[str, object]] = []
    for key in keys:
        if key in missing:
            continue
        left_result = left_flat[key]
        right_result = right_flat[key]
        left_tokens = list(left_result["token_ids"])
        right_tokens = list(right_result["token_ids"])
        if left_tokens == right_tokens:
            token_exact_count += 1
        elif len(first_differences) < 12:
            first_differences.append(
                {
                    "round": key[0],
                    "case_id": key[1],
                    "left_token_ids": left_tokens,
                    "right_token_ids": right_tokens,
                }
            )
        if left_tokens and right_tokens and left_tokens[0] == right_tokens[0]:
            first_token_match_count += 1
        for left_logprob, right_logprob in zip(
            left_result["token_logprobs"], right_result["token_logprobs"]
        ):
            error = abs(float(left_logprob) - float(right_logprob))
            if math.isfinite(error):
                aligned_logprob_errors.append(error)

    sequence_count = len(keys)
    max_error = max(aligned_logprob_errors, default=None)
    tokens_exact = not missing and token_exact_count == sequence_count
    logprobs_close = max_error is not None and max_error <= logprob_atol
    return {
        "sequence_count": sequence_count,
        "missing_keys": [list(key) for key in missing],
        "token_exact_count": token_exact_count,
        "tokens_exact": tokens_exact,
        "first_token_match_count": first_token_match_count,
        "first_token_match_rate": (
            first_token_match_count / sequence_count if sequence_count else 0.0
        ),
        "aligned_logprob_count": len(aligned_logprob_errors),
        "median_logprob_abs_error": (
            statistics.median(aligned_logprob_errors)
            if aligned_logprob_errors
            else None
        ),
        "max_logprob_abs_error": max_error,
        "logprobs_close": logprobs_close,
        "first_differences": first_differences,
        "passed": bool(tokens_exact and logprobs_close),
    }


def _within_mode_checks(
    report: dict[str, object], *, logprob_atol: float
) -> dict[str, object]:
    rounds = report["rounds"]
    prefix_a = {"rounds": {"same": rounds["prefix-a"]}}
    repeated = {"rounds": {"same": rounds["prefix-a-repeat"]}}
    reversed_a = {"rounds": {"same": rounds["prefix-a-reverse"]}}
    repeat = _compare_results(prefix_a, repeated, logprob_atol=logprob_atol)
    reorder = _compare_results(prefix_a, reversed_a, logprob_atol=logprob_atol)
    prefix_b = {"rounds": {"same": rounds["prefix-b"]}}
    sensitivity = _compare_results(prefix_a, prefix_b, logprob_atol=logprob_atol)
    return {
        "repeat_exact": repeat["passed"],
        "reorder_exact": reorder["passed"],
        "prefix_mutation_visible": not sensitivity["passed"],
        "repeat": repeat,
        "reorder": reorder,
        "prefix_mutation": sensitivity,
    }


def compare_execution_reports(
    reports: Sequence[dict[str, object]],
    *,
    logprob_atol: float,
) -> dict[str, object]:
    by_mode = {str(report["execution_mode"]): report for report in reports}
    missing_modes = [
        mode for mode in REQUIRED_EXECUTION_MODES if mode not in by_mode
    ]
    if missing_modes:
        raise ValueError(f"missing execution modes: {', '.join(missing_modes)}")

    eager = by_mode["eager"]
    against_eager = {
        mode: _compare_results(eager, by_mode[mode], logprob_atol=logprob_atol)
        for mode in EXECUTION_MODES[1:]
        if mode in by_mode
    }
    within = {
        mode: _within_mode_checks(by_mode[mode], logprob_atol=logprob_atol)
        for mode in EXECUTION_MODES
        if mode in by_mode
    }
    pairwise_modes = (
        ("compile-only", "cuda-graph"),
        ("dynamo-eager-native-kernels", "cuda-graph-eager-kernels"),
        ("dynamo-eager-native-kernels", "dynamo-eager-custom-kernels"),
        ("dynamo-eager-custom-kernels", "cuda-graph-custom-kernels"),
    )
    pairwise = {
        f"{left}__vs__{right}": _compare_results(
            by_mode[left], by_mode[right], logprob_atol=logprob_atol
        )
        for left, right in pairwise_modes
        if left in by_mode and right in by_mode
    }
    layer_findings = {
        "persistent_input_path_matches_eager": against_eager[
            "persistent-eager"
        ]["passed"],
        "inductor_graph_matches_compile_only": pairwise.get(
            "compile-only__vs__cuda-graph", {}
        ).get("passed"),
        "eager_adapter_graph_matches_no_graph": pairwise.get(
            "dynamo-eager-native-kernels__vs__cuda-graph-eager-kernels", {}
        ).get("passed"),
        "custom_kernels_restore_eager_without_graph": against_eager.get(
            "dynamo-eager-custom-kernels", {}
        ).get("passed"),
        "cuda_graph_custom_kernels_match_eager": against_eager.get(
            "cuda-graph-custom-kernels", {}
        ).get("passed"),
    }

    if not against_eager["persistent-eager"]["passed"]:
        classification = "persistent_input_path_divergence"
    elif layer_findings["cuda_graph_custom_kernels_match_eager"]:
        classification = "cuda_graph_custom_kernels_match_eager"
    elif layer_findings["custom_kernels_restore_eager_without_graph"]:
        classification = "custom_kernels_restore_eager_without_graph"
    elif (
        "cuda-graph-eager-kernels" in against_eager
        and against_eager["cuda-graph-eager-kernels"]["passed"]
        and not against_eager["compile-only"]["passed"]
    ):
        classification = "inductor_divergence_graph_with_eager_kernels_matches"
    elif not against_eager["compile-only"]["passed"]:
        classification = "inductor_compile_divergence"
    elif not against_eager["cuda-graph"]["passed"]:
        classification = "cuda_graph_replay_divergence"
    elif not all(
        check["repeat_exact"] and check["reorder_exact"]
        for check in within.values()
    ):
        classification = "within_mode_nondeterminism"
    else:
        classification = "all_execution_modes_match"

    passed = classification == "all_execution_modes_match"
    return {
        "schema_version": 1,
        "logprob_atol": logprob_atol,
        "classification": classification,
        "against_eager": against_eager,
        "pairwise": pairwise,
        "layer_findings": layer_findings,
        "within_mode_checks": within,
        "runtime_seconds": {
            mode: by_mode[mode].get("runtime_seconds")
            for mode in EXECUTION_MODES
            if mode in by_mode
        },
        "passed": passed,
    }


def _read_trace_prompts(path: Path, case_count: int) -> list[str]:
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("reading a trace requires zstandard") from error

    prompts: list[str] = []
    task_types: set[str] = set()
    with path.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                record = json.loads(line)
                steps = record.get("steps", [])
                if not steps:
                    continue
                task_type = str(record.get("task_type"))
                if task_type in task_types and len(task_types) < case_count:
                    continue
                message = steps[0].get("policy_user_message")
                if not isinstance(message, str) or not message:
                    continue
                prompts.append(message)
                task_types.add(task_type)
                if len(prompts) == case_count:
                    break
    if len(prompts) < case_count:
        raise RuntimeError(
            f"trace contains only {len(prompts)} usable first-step prompts; "
            f"requested {case_count}"
        )
    return prompts


def _build_runtime_cases(
    tokenizer,
    prompts: Sequence[str],
    *,
    prefix_length: int,
    hidden_size: int,
    prefix_scale: float,
    base_seed: int,
) -> list[dict[str, object]]:
    import torch

    cases = []
    for index, prompt in enumerate(prompts):
        prompt_token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(prompt_token_ids, "tolist"):
            prompt_token_ids = prompt_token_ids.tolist()
        generator_a = torch.Generator(device="cpu")
        generator_b = torch.Generator(device="cpu")
        generator_a.manual_seed(base_seed + index)
        generator_b.manual_seed(base_seed + 100_000 + index)
        cases.append(
            {
                "case_id": f"case-{index:02d}",
                "prompt_token_ids": [int(value) for value in prompt_token_ids],
                "prefix_a": torch.randn(
                    (prefix_length, hidden_size),
                    generator=generator_a,
                    dtype=torch.float32,
                ).mul_(prefix_scale),
                "prefix_b": torch.randn(
                    (prefix_length, hidden_size),
                    generator=generator_b,
                    dtype=torch.float32,
                ).mul_(prefix_scale),
            }
        )
    return cases


def _override_vllm_compilation(
    *,
    use_cudagraph: bool | None = None,
    use_inductor: bool | None = None,
    custom_ops: Sequence[str] | None = None,
) -> None:
    """Override one V1 compilation layer after vLLM applies its defaults."""
    from vllm.config import VllmConfig

    original = VllmConfig.__post_init__

    def with_overrides(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if use_cudagraph is not None:
            self.compilation_config.use_cudagraph = use_cudagraph
        if use_inductor is not None:
            self.compilation_config.use_inductor = use_inductor
        if custom_ops is not None:
            self.compilation_config.custom_ops = list(custom_ops)

    VllmConfig.__post_init__ = with_overrides


def _sample_result(sample) -> dict[str, object]:
    token_ids = [int(value) for value in sample.token_ids]
    token_logprobs = []
    for index, token_id in enumerate(token_ids):
        item = sample.logprobs[index].get(token_id)
        if item is None:
            raise RuntimeError(f"missing sampled-token logprob at position {index}")
        token_logprobs.append(float(item.logprob))
    return {
        "token_ids": token_ids,
        "token_logprobs": token_logprobs,
        "finish_reason": sample.finish_reason,
    }


def run_execution_mode(args: argparse.Namespace) -> dict[str, object]:
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    persistent_input = args.execution_mode != "eager"
    if persistent_input:
        os.environ["VLLM_INFOSKILL_HYBRID_PREFIX_CUDA_GRAPH"] = "1"
    else:
        os.environ.pop("VLLM_INFOSKILL_HYBRID_PREFIX_CUDA_GRAPH", None)

    if args.execution_mode == "compile-only":
        _override_vllm_compilation(use_cudagraph=False)
    elif args.execution_mode == "cuda-graph-eager-kernels":
        _override_vllm_compilation(use_cudagraph=True, use_inductor=False)
    elif args.execution_mode == "dynamo-eager-native-kernels":
        _override_vllm_compilation(
            use_cudagraph=False,
            use_inductor=False,
            custom_ops=("none",),
        )
    elif args.execution_mode == "dynamo-eager-custom-kernels":
        _override_vllm_compilation(
            use_cudagraph=False,
            use_inductor=False,
            custom_ops=("all",),
        )
    elif args.execution_mode == "cuda-graph-custom-kernels":
        _override_vllm_compilation(
            use_cudagraph=True,
            use_inductor=False,
            custom_ops=("all",),
        )

    import torch
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs.data import (
        INFOSKILL_HYBRID_PREFIX_API,
        INFOSKILL_HYBRID_PREFIX_CUDA_GRAPH_API,
    )

    if INFOSKILL_HYBRID_PREFIX_API != 1:
        raise RuntimeError("installed vLLM lacks INFO-SKILL Hybrid Prefix API 1")
    if persistent_input and INFOSKILL_HYBRID_PREFIX_CUDA_GRAPH_API != 1:
        raise RuntimeError(
            "installed vLLM lacks INFO-SKILL Hybrid Prefix CUDA Graph API 1"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    prompts = (
        _read_trace_prompts(args.trace, args.case_count)
        if args.trace is not None
        else list(FALLBACK_PROMPTS[: args.case_count])
    )
    if len(prompts) < args.case_count:
        raise ValueError("case_count exceeds the built-in prompt count")
    cases = _build_runtime_cases(
        tokenizer,
        prompts,
        prefix_length=args.prefix_length,
        hidden_size=int(config.hidden_size),
        prefix_scale=args.prefix_scale,
        base_seed=args.base_seed,
    )
    placeholder_id = tokenizer.pad_token_id
    if placeholder_id is None:
        placeholder_id = tokenizer.eos_token_id
    if placeholder_id is None:
        raise RuntimeError("tokenizer has no pad or eos token")

    enforce_eager = args.execution_mode in {"eager", "persistent-eager"}
    started = time.perf_counter()
    engine = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        logprobs=1,
        seed=0,
    )
    round_results = {}
    round_seconds = {}
    for round_name, requests in build_execution_rounds(
        cases, placeholder_id=int(placeholder_id)
    ).items():
        round_started = time.perf_counter()
        outputs = engine.generate(
            prompts=[request["prompt"] for request in requests],
            sampling_params=sampling,
            use_tqdm=False,
        )
        round_seconds[round_name] = time.perf_counter() - round_started
        round_results[round_name] = {
            request["case_id"]: _sample_result(output.outputs[0])
            for request, output in zip(requests, outputs, strict=True)
        }

    return {
        "schema_version": 1,
        "execution_mode": args.execution_mode,
        "model": str(Path(args.model).resolve()),
        "trace": str(args.trace.resolve()) if args.trace else None,
        "case_count": args.case_count,
        "prefix_length": args.prefix_length,
        "prefix_scale": args.prefix_scale,
        "max_tokens": args.max_tokens,
        "runtime_seconds": time.perf_counter() - started,
        "round_seconds": round_seconds,
        "rounds": round_results,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "enforce_eager": enforce_eager,
            "persistent_input_path": persistent_input,
            "cuda_graph_requested": args.execution_mode
            in {
                "cuda-graph",
                "cuda-graph-eager-kernels",
                "cuda-graph-custom-kernels",
            },
            "inductor_requested": args.execution_mode
            in {"compile-only", "cuda-graph"},
            "custom_kernel_policy": (
                "all"
                if args.execution_mode
                in {"dynamo-eager-custom-kernels", "cuda-graph-custom-kernels"}
                else "none"
                if args.execution_mode
                in {
                    "compile-only",
                    "cuda-graph",
                    "cuda-graph-eager-kernels",
                    "dynamo-eager-native-kernels",
                }
                else "default"
            ),
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Differentiate hybrid-prefix input, Inductor, and CUDA Graph drift"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--execution-mode", choices=EXECUTION_MODES, required=True)
    run.add_argument("--trace", type=Path)
    run.add_argument("--case-count", type=int, default=4)
    run.add_argument("--prefix-length", type=int, default=5)
    run.add_argument("--prefix-scale", type=float, default=0.0086)
    run.add_argument("--base-seed", type=int, default=20260914)
    run.add_argument("--max-tokens", type=int, default=12)
    run.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    run.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    run.add_argument("--max-model-len", type=int, default=4096)
    run.add_argument("--max-num-batched-tokens", type=int, default=16384)
    run.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("reports", nargs="+", type=Path)
    compare.add_argument("--logprob-atol", type=float, default=1e-3)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "run":
        if args.case_count < 1 or args.prefix_length < 1 or args.max_tokens < 1:
            raise ValueError("case_count, prefix_length, and max_tokens must be positive")
        report = run_execution_mode(args)
    else:
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
        report = compare_execution_reports(reports, logprob_atol=args.logprob_atol)

    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if args.command == "run" or report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
