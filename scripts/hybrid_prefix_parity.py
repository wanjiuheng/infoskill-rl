from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import statistics
import tempfile
from pathlib import Path


PARITY_PROMPTS = (
    "Reply with one short word.",
    "You are standing in a room. State one concise next action.",
    "Goal: put the apple in the fridge. Return one concise action.",
    "Available actions: look, inventory, go to kitchen. Return one action.",
)


def build_case_specs(*, case_count: int, base_seed: int) -> list[dict[str, object]]:
    if case_count < 1:
        raise ValueError("case_count must be positive")
    cases: list[dict[str, object]] = []
    seed_offset = 0
    while len(cases) < case_count:
        for prompt in PARITY_PROMPTS:
            if len(cases) == case_count:
                break
            cases.append(
                {
                    "case_id": f"case-{len(cases):02d}",
                    "prompt": prompt,
                    "prefix_seed": base_seed + seed_offset,
                }
            )
        seed_offset += 1
    return cases


def _hybrid_prompt(text_ids, prefix, placeholder_id: int) -> dict[str, object]:
    prefix_length = len(prefix)
    return {
        "prompt_token_ids": [placeholder_id] * prefix_length + list(text_ids),
        "infoskill_prefix_embeds": prefix,
        "infoskill_prefix_mask": (
            [True] * prefix_length + [False] * len(text_ids)
        ),
    }


def build_vllm_request_plan(
    cases: list[dict[str, object]], placeholder_id: int
) -> dict[str, list[dict[str, object]]]:
    transport_plain: list[dict[str, object]] = []
    transport_hybrid: list[dict[str, object]] = []
    cross_backend: list[dict[str, object]] = []
    for case in cases:
        case_id = case["case_id"]
        text_ids = case["text_ids"]
        transport_plain.append(
            {
                "case_id": case_id,
                "side": "plain_vllm",
                "prompt": {
                    "prompt_token_ids": (
                        list(case["prefix_token_ids"]) + list(text_ids)
                    )
                },
            }
        )
        transport_hybrid.append(
            {
                "case_id": case_id,
                "side": "hybrid_vllm",
                "prompt": _hybrid_prompt(
                    text_ids, case["token_prefix"], placeholder_id
                ),
            }
        )
        cross_backend.append(
            {
                "case_id": case_id,
                "side": "vllm",
                "prompt": _hybrid_prompt(
                    text_ids, case["random_prefix"], placeholder_id
                ),
            }
        )
    return {
        "transport_plain": transport_plain,
        "transport_hybrid": transport_hybrid,
        "cross_backend": cross_backend,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_comparisons(comparisons: list[dict[str, object]]) -> dict[str, object]:
    if not comparisons:
        raise ValueError("parity gate requires at least one comparison")
    evaluated: list[dict[str, object]] = []
    errors: list[float] = []
    token_match_count = 0
    for comparison in comparisons:
        item = dict(comparison)
        left = item.pop("left")
        right = item.pop("right")
        token_match = left["token_id"] == right["token_id"]
        if token_match:
            error = abs(float(left["logprob"]) - float(right["logprob"]))
            if not math.isfinite(error):
                error = None
            else:
                errors.append(error)
            token_match_count += 1
        else:
            error = None
        item.update(
            {
                "left": left,
                "right": right,
                "token_match": token_match,
                "logprob_abs_error": error,
            }
        )
        evaluated.append(item)

    case_count = len(comparisons)
    return {
        "case_count": case_count,
        "cases": evaluated,
        "summary": {
            "finite_error_count": len(errors),
            "token_match_count": token_match_count,
            "token_match_rate": token_match_count / case_count,
            "median_logprob_abs_error": statistics.median(errors) if errors else None,
            "p95_logprob_abs_error": _percentile(errors, 0.95),
            "max_logprob_abs_error": max(errors) if errors else None,
        },
    }


def summarize_parity(
    transport_comparisons: list[dict[str, object]],
    cross_backend_comparisons: list[dict[str, object]],
    *,
    transport_logprob_atol: float,
    cross_p95_logprob_atol: float,
    cross_max_logprob_atol: float,
    required_token_match_rate: float,
) -> dict[str, object]:
    transport = _summarize_comparisons(transport_comparisons)
    cross_backend = _summarize_comparisons(cross_backend_comparisons)
    transport_summary = transport["summary"]
    cross_summary = cross_backend["summary"]

    transport["passed"] = (
        transport_summary["token_match_rate"] >= required_token_match_rate
        and transport_summary["finite_error_count"]
        == transport_summary["token_match_count"]
        and transport_summary["max_logprob_abs_error"] <= transport_logprob_atol
    )
    cross_backend["passed"] = (
        cross_summary["token_match_rate"] >= required_token_match_rate
        and cross_summary["finite_error_count"]
        == cross_summary["token_match_count"]
        and cross_summary["p95_logprob_abs_error"] <= cross_p95_logprob_atol
        and cross_summary["max_logprob_abs_error"] <= cross_max_logprob_atol
    )
    return {
        "schema_version": 2,
        "thresholds": {
            "transport_logprob_atol": transport_logprob_atol,
            "cross_p95_logprob_atol": cross_p95_logprob_atol,
            "cross_max_logprob_atol": cross_max_logprob_atol,
            "required_token_match_rate": required_token_match_rate,
        },
        "transport_gate": transport,
        "cross_backend_gate": cross_backend,
        "passed": bool(transport["passed"] and cross_backend["passed"]),
    }


def _prepare_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise IsADirectoryError(f"output path is a directory: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{resolved.name}.", suffix=".probe", dir=resolved.parent
    ):
        pass
    return resolved


def _write_json_report(path: Path, report: dict[str, object]) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_metadata(torch) -> dict[str, object]:
    packages = {}
    for name in ("torch", "transformers", "vllm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "infoskill_hybrid_prefix_api": 1,
    }


def _dtype(torch, name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _sample_result(torch, logits) -> dict[str, object]:
    logits = logits.float()
    token_id = int(logits.argmax().item())
    logprob = float(torch.log_softmax(logits, dim=-1)[token_id].item())
    return {"token_id": token_id, "logprob": logprob}


def _prepare_transformers_cases(
    model_path: str,
    tokenizer,
    case_specs: list[dict[str, object]],
    *,
    prefix_length: int,
    hidden_size: int,
    prefix_scale: float,
    dtype,
) -> list[dict[str, object]]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to("cuda:0")
    model.eval()
    embedding_layer = model.get_input_embeddings()
    runtime_cases: list[dict[str, object]] = []
    with torch.inference_mode():
        for index, spec in enumerate(case_specs):
            text_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": spec["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            text_ids = [int(value) for value in text_ids]
            if not text_ids:
                raise RuntimeError(f"tokenizer produced no tokens for {spec['case_id']}")
            prefix_token_ids = [
                text_ids[(index + offset) % len(text_ids)]
                for offset in range(prefix_length)
            ]
            prefix_token_tensor = torch.tensor(
                prefix_token_ids, dtype=torch.long, device="cuda:0"
            )
            token_prefix = (
                embedding_layer(prefix_token_tensor)
                .detach()
                .float()
                .cpu()
                .contiguous()
            )

            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(spec["prefix_seed"]))
            random_prefix = (
                torch.randn(
                    (prefix_length, hidden_size),
                    generator=generator,
                    dtype=torch.float32,
                )
                * prefix_scale
            ).contiguous()

            ids = torch.tensor([text_ids], dtype=torch.long, device="cuda:0")
            token_embeddings = embedding_layer(ids)
            prefix_gpu = random_prefix.to(
                device="cuda:0", dtype=token_embeddings.dtype
            ).unsqueeze(0)
            inputs_embeds = torch.cat((prefix_gpu, token_embeddings), dim=1)
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long, device="cuda:0"
            )
            position_ids = torch.arange(
                inputs_embeds.shape[1], device="cuda:0"
            ).unsqueeze(0)
            output = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            runtime_cases.append(
                {
                    **spec,
                    "text_ids": text_ids,
                    "text_token_count": len(text_ids),
                    "prefix_token_ids": prefix_token_ids,
                    "token_prefix": token_prefix,
                    "random_prefix": random_prefix,
                    "transformers": _sample_result(torch, output.logits[0, -1]),
                }
            )
            del output, inputs_embeds, prefix_gpu, token_embeddings, ids

    del embedding_layer, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return runtime_cases


def _vllm_sample_result(sample) -> dict[str, object]:
    token_id = int(sample.token_ids[0])
    return {
        "token_id": token_id,
        "logprob": float(sample.logprobs[0][token_id].logprob),
    }


def _vllm_comparisons(
    model_path: str,
    cases: list[dict[str, object]],
    *,
    placeholder_id: int,
    dtype: str,
    gpu_memory_utilization: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams
    from vllm.inputs.data import INFOSKILL_HYBRID_PREFIX_API

    if INFOSKILL_HYBRID_PREFIX_API != 1:
        raise RuntimeError("installed vLLM does not expose INFO-SKILL Hybrid Prefix API 1")
    engine = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=dtype,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=512,
        max_num_batched_tokens=512,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=1,
        seed=0,
    )
    plan = build_vllm_request_plan(cases, placeholder_id)

    def run(requests: list[dict[str, object]]) -> list[dict[str, object]]:
        outputs = engine.generate(
            prompts=[request["prompt"] for request in requests],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != len(requests):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(requests)} requests"
            )
        return [
            {
                "case_id": request["case_id"],
                "side": request["side"],
                "result": _vllm_sample_result(output.outputs[0]),
            }
            for request, output in zip(requests, outputs, strict=True)
        ]

    transport_outputs = run(plan["transport_plain"])
    transport_outputs.extend(run(plan["transport_hybrid"]))
    cross_outputs = run(plan["cross_backend"])
    transport_by_case: dict[str, dict[str, dict[str, object]]] = {}
    for output in transport_outputs:
        transport_by_case.setdefault(output["case_id"], {})[output["side"]] = output[
            "result"
        ]
    cross_by_case = {
        output["case_id"]: output["result"] for output in cross_outputs
    }

    transport_comparisons: list[dict[str, object]] = []
    cross_comparisons: list[dict[str, object]] = []
    for case in cases:
        metadata = {
            "case_id": case["case_id"],
            "prompt": case["prompt"],
            "prefix_seed": case["prefix_seed"],
            "prefix_token_ids": case["prefix_token_ids"],
            "text_token_count": case["text_token_count"],
        }
        transport_results = transport_by_case[case["case_id"]]
        transport_comparisons.append(
            {
                **metadata,
                "left_backend": "plain_vllm_tokens",
                "right_backend": "hybrid_vllm_token_embeddings",
                "left": transport_results["plain_vllm"],
                "right": transport_results["hybrid_vllm"],
            }
        )
        cross_comparisons.append(
            {
                **metadata,
                "left_backend": "transformers_hybrid",
                "right_backend": "vllm_hybrid",
                "left": case["transformers"],
                "right": cross_by_case[case["case_id"]],
            }
        )
    return transport_comparisons, cross_comparisons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="INFO-SKILL Hybrid Prefix transport and cross-backend parity gate"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--prefix-length", type=int, default=5)
    parser.add_argument("--prefix-scale", type=float, default=0.02)
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=20260904)
    parser.add_argument("--transport-logprob-atol", type=float, default=0.0001)
    parser.add_argument("--cross-p95-logprob-atol", type=float, default=0.05)
    parser.add_argument("--cross-max-logprob-atol", type=float, default=0.10)
    parser.add_argument("--required-token-match-rate", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--output", type=Path, default=Path("hybrid-prefix-parity.json"))
    args = parser.parse_args()
    output_path = _prepare_output_path(args.output)

    import torch
    from transformers import AutoConfig, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the parity smoke")
    if args.prefix_length < 1:
        raise ValueError("prefix_length must be positive for the dual parity gate")
    if args.prefix_scale <= 0:
        raise ValueError("prefix_scale must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if not 0 < args.required_token_match_rate <= 1:
        raise ValueError("required_token_match_rate must be in (0, 1]")
    for name in (
        "transport_logprob_atol",
        "cross_p95_logprob_atol",
        "cross_max_logprob_atol",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hidden_size = int(getattr(config, "hidden_size"))
    placeholder_id = tokenizer.pad_token_id
    if placeholder_id is None:
        placeholder_id = tokenizer.eos_token_id
    if placeholder_id is None:
        raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")

    case_specs = build_case_specs(
        case_count=args.case_count,
        base_seed=args.base_seed,
    )
    runtime_cases = _prepare_transformers_cases(
        args.model,
        tokenizer,
        case_specs,
        prefix_length=args.prefix_length,
        hidden_size=hidden_size,
        prefix_scale=args.prefix_scale,
        dtype=_dtype(torch, args.dtype),
    )
    transport_comparisons, cross_comparisons = _vllm_comparisons(
        args.model,
        runtime_cases,
        placeholder_id=int(placeholder_id),
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    report = summarize_parity(
        transport_comparisons,
        cross_comparisons,
        transport_logprob_atol=args.transport_logprob_atol,
        cross_p95_logprob_atol=args.cross_p95_logprob_atol,
        cross_max_logprob_atol=args.cross_max_logprob_atol,
        required_token_match_rate=args.required_token_match_rate,
    )
    report.update(
        {
            "model": str(Path(args.model).resolve()),
            "dtype": args.dtype,
            "prefix_length": args.prefix_length,
            "prefix_scale": args.prefix_scale,
            "case_count": args.case_count,
            "base_seed": args.base_seed,
            "prompt_count": len({case["prompt"] for case in case_specs}),
            "prefix_seed_count": len(
                {case["prefix_seed"] for case in case_specs}
            ),
            "hidden_size": hidden_size,
            "runtime": _runtime_metadata(torch),
        }
    )
    _write_json_report(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
