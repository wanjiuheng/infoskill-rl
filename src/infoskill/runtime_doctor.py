from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
import platform
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


PACKAGES = (
    "torch",
    "transformers",
    "cachetools",
    "flash-attn",
    "peft",
    "vllm",
    "ray",
    "sentence-transformers",
    "tensordict",
)

MODULES_TO_INSPECT = (
    "infoskill",
    "vllm",
    "vllm.inputs.data",
    "vllm.entrypoints.llm",
    "vllm.engine.llm_engine",
    "vllm.worker.model_runner",
    "vllm.v1.worker.gpu_model_runner",
    "verl",
    "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd",
    "verl.workers.fsdp_workers",
)


def collect_report() -> dict[str, object]:
    report: dict[str, object] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "working_directory": str(Path.cwd().resolve()),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "platform": platform.platform(),
        "packages": {},
        "modules": {},
        "cuda": {},
    }
    packages = report["packages"]
    assert isinstance(packages, dict)
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    modules = report["modules"]
    assert isinstance(modules, dict)
    for module_name in MODULES_TO_INSPECT:
        modules[module_name] = _module_report(module_name)

    try:
        import torch

        cuda = report["cuda"]
        assert isinstance(cuda, dict)
        cuda.update(
            {
                "available": torch.cuda.is_available(),
                "torch_cuda": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            }
        )
    except Exception as error:
        report["cuda"] = {"error": f"{type(error).__name__}: {error}"}

    report["vllm_api"] = _vllm_api_report()
    return report


def _module_report(module_name: str) -> dict[str, object]:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    if spec is None or spec.origin is None:
        return {"available": False}
    path = Path(spec.origin).resolve()
    payload: dict[str, object] = {"available": True, "path": str(path)}
    if path.is_file():
        try:
            raw = path.read_bytes()
            payload.update(
                {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "contains_prompt_embeds": b"prompt_embeds" in raw,
                    "contains_sampling_seed": b"semantic_seeds" in raw or b"sampling_params.seed" in raw,
                }
            )
        except OSError as error:
            payload["read_error"] = str(error)
    return payload


def _vllm_api_report() -> dict[str, object]:
    try:
        from vllm import LLM, SamplingParams
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    result: dict[str, object] = {"available": True}
    for name, value in (("LLM.generate", LLM.generate), ("SamplingParams", SamplingParams)):
        try:
            result[name] = str(inspect.signature(value))
        except (TypeError, ValueError) as error:
            result[name] = f"unavailable: {error}"
    try:
        fields = _sampling_param_fields(SamplingParams)
        result["sampling_fields"] = list(fields)
        result["has_per_request_seed_field"] = "seed" in fields
        result["has_stop_string_fields"] = all(
            field in fields for field in ("stop", "include_stop_str_in_output")
        )
        if "seed" in fields:
            seed_probe = SamplingParams(seed=31_415_926)
            result["seed_roundtrip"] = getattr(seed_probe, "seed", None) == 31_415_926
        if result["has_stop_string_fields"]:
            result["action_stop_roundtrip"] = _action_stop_roundtrip(SamplingParams)
    except Exception as error:
        result["field_error"] = f"{type(error).__name__}: {error}"
    try:
        sampling_annotation = inspect.signature(LLM.generate).parameters["sampling_params"].annotation
        result["generate_accepts_sampling_param_sequence"] = "Sequence" in str(sampling_annotation)
    except Exception as error:
        result["sampling_sequence_error"] = f"{type(error).__name__}: {error}"
    try:
        from vllm.inputs.data import TextPrompt, TokensPrompt

        hybrid_api = getattr(
            importlib.import_module("vllm.inputs.data"),
            "INFOSKILL_HYBRID_PREFIX_API",
            None,
        )
        result["prompt_input_fields"] = {
            "TextPrompt": sorted(getattr(TextPrompt, "__annotations__", {})),
            "TokensPrompt": sorted(getattr(TokensPrompt, "__annotations__", {})),
        }
        token_fields = sorted(getattr(TokensPrompt, "__annotations__", {}))
        result["infoskill_hybrid_prefix_api"] = hybrid_api
        result["has_infoskill_hybrid_prefix"] = _has_hybrid_prefix_api(
            hybrid_api, token_fields
        )
    except Exception as error:
        result["prompt_input_error"] = f"{type(error).__name__}: {error}"
    return result


def _sampling_param_fields(sampling_params_type: object) -> tuple[str, ...]:
    """Return fields for dataclasses, msgspec.Struct classes, and regular callables."""

    fields: set[str] = set()
    dataclass_fields = getattr(sampling_params_type, "__dataclass_fields__", {})
    fields.update(str(name) for name in dataclass_fields)
    struct_fields = getattr(sampling_params_type, "__struct_fields__", ())
    fields.update(str(name) for name in struct_fields)
    try:
        fields.update(inspect.signature(sampling_params_type).parameters)
    except (TypeError, ValueError):
        pass
    return tuple(sorted(fields))


def _action_stop_roundtrip(sampling_params_type: object) -> bool:
    from infoskill.integrations.verl.hybrid_rollout import vllm_action_stop_settings

    probe = sampling_params_type(**vllm_action_stop_settings("</action>"))  # type: ignore[operator]
    stop = getattr(probe, "stop", None)
    return bool(
        (stop == "</action>" or stop == ["</action>"])
        and getattr(probe, "detokenize", None) is True
        and getattr(probe, "include_stop_str_in_output", None) is True
    )


def _has_hybrid_prefix_api(api_version: object, token_fields: object) -> bool:
    fields = set(token_fields) if isinstance(token_fields, (list, tuple, set)) else set()
    return api_version == 1 and {
        "infoskill_prefix_embeds",
        "infoskill_prefix_mask",
    }.issubset(fields)


def main() -> int:
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
        report = collect_report()
    captured_logs = [
        line
        for stream in (captured_stdout.getvalue(), captured_stderr.getvalue())
        for line in stream.splitlines()
        if line.strip()
    ]
    if captured_logs:
        report["captured_import_logs"] = captured_logs
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
