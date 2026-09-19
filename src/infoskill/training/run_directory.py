from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from infoskill.persistence.model_identity import provenance_matches_pinned_model


_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_POLICY_WARMUP_RATIO = 0.03


def resolve_training_run_directory(
    *,
    output_root: str | Path,
    profile_name: str,
    run_name: str | None,
    resume: str | None,
    default_run_name: str | None = None,
) -> tuple[Path, Path | None, bool]:
    """Resolve an in-place run or a named fork from an immutable checkpoint."""
    if resume is not None:
        checkpoint = Path(resume).expanduser().resolve()
        if checkpoint.parent.name != "checkpoints":
            raise ValueError("resume path must be a run checkpoints/step-* directory")
        if run_name is None:
            return checkpoint.parent.parent, checkpoint, False
        return _create_run_directory(output_root, run_name), checkpoint, True

    name = run_name or default_run_name or f"m0-{profile_name}"
    return _create_run_directory(output_root, name), None, False


def validate_resume_config(
    checkpoint: str | Path,
    current: Mapping[str, object],
    *,
    allow_gpu_change: bool,
    allow_performance_candidate_change: bool = False,
) -> int:
    checkpoint_path = Path(checkpoint)
    path = checkpoint_path / "resolved_config.json"
    if not path.is_file():
        raise RuntimeError(f"resume checkpoint has no resolved config: {path}")
    previous = json.loads(path.read_text(encoding="utf-8"))
    previous_gpus = int(previous.get("num_gpus", 0))
    current_gpus = int(current.get("num_gpus", 0))

    previous_without_gpus = _with_runtime_defaults(previous)
    current_without_gpus = _with_runtime_defaults(current)
    previous_without_gpus, current_without_gpus = (
        _normalize_extendable_training_target(
            previous_without_gpus,
            current_without_gpus,
        )
    )
    if allow_performance_candidate_change:
        previous_without_gpus = _without_performance_candidates(
            previous_without_gpus
        )
        current_without_gpus = _without_performance_candidates(
            current_without_gpus
        )
    matching_model_id = _matching_policy_model_id(previous, current)
    if matching_model_id is not None:
        _validate_checkpoint_policy_provenance(checkpoint_path, matching_model_id)
        previous_without_gpus = _without_policy_model_path(previous_without_gpus)
        current_without_gpus = _without_policy_model_path(current_without_gpus)
    previous_without_gpus.pop("num_gpus", None)
    current_without_gpus.pop("num_gpus", None)
    if previous_without_gpus != current_without_gpus:
        raise RuntimeError("resume configuration differs from the original run")
    if previous_gpus != current_gpus and not allow_gpu_change:
        raise RuntimeError(
            "changing GPU count during resume requires a new run_name"
        )
    return previous_gpus


def _normalize_extendable_training_target(
    previous: Mapping[str, object],
    current: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Ignore a target-only extension when it preserves scheduler semantics."""
    normalized_previous = dict(previous)
    normalized_current = dict(current)
    previous_plan = previous.get("training_plan")
    current_plan = current.get("training_plan")
    if not isinstance(previous_plan, Mapping) or not isinstance(
        current_plan, Mapping
    ):
        return normalized_previous, normalized_current

    previous_max = previous_plan.get("max_updates")
    current_max = current_plan.get("max_updates")
    if (
        not isinstance(previous_max, int)
        or isinstance(previous_max, bool)
        or not isinstance(current_max, int)
        or isinstance(current_max, bool)
        or current_max < previous_max
        or _warmup_steps(current_max) != _warmup_steps(previous_max)
    ):
        return normalized_previous, normalized_current

    normalized_previous_plan = dict(previous_plan)
    normalized_previous_plan["max_updates"] = current_max
    normalized_previous["training_plan"] = normalized_previous_plan
    return normalized_previous, normalized_current


def _warmup_steps(max_updates: int) -> int:
    return int(max_updates * _POLICY_WARMUP_RATIO)


def _with_runtime_defaults(config: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(config)
    app_config = normalized.get("app_config")
    if isinstance(app_config, Mapping):
        normalized_app = dict(app_config)
        paths = normalized_app.get("paths")
        if isinstance(paths, Mapping):
            normalized_paths = dict(paths)
            if normalized_paths.get("grounding_data") is None:
                normalized_paths.pop("grounding_data", None)
            normalized_app["paths"] = normalized_paths
        normalized["app_config"] = normalized_app
    runtime_options = normalized.get("runtime_options")
    if isinstance(runtime_options, Mapping):
        normalized_options = dict(runtime_options)
        normalized_options.setdefault("environment_workers", 1)
        normalized_options.setdefault("environment_backend", "individual")
        normalized_options.setdefault("cuda_memory_poll_interval_ms", 0)
        # Missing means the historical pre-D015 default, not today's default.
        normalized_options.setdefault("policy_max_tokens_per_gpu", 16_384)
        normalized_options.setdefault("balance_policy_tokens_across_ranks", False)
        normalized_options.setdefault("skip_unused_old_logprob_entropy", False)
        normalized_options.setdefault("rollout_max_batched_tokens", 16_384)
        normalized_options.setdefault("hybrid_prefix_cuda_graph", False)
        normalized_options.setdefault("lora_shrink_split_k_one", False)
        # Historical graph runs used vLLM V1's forced Inductor/custom_ops=none
        # policy.  Keep that identity distinct from the corrected candidate.
        historical_graph = bool(
            normalized_options.get("hybrid_prefix_cuda_graph", False)
        )
        normalized_options.setdefault(
            "hybrid_prefix_cuda_graph_custom_kernels",
            False,
        )
        normalized_options.setdefault(
            "hybrid_prefix_cuda_graph_use_inductor",
            True if historical_graph else None,
        )
        normalized_options.setdefault("fuse_kl_ppo_forward", False)
        normalized_options.setdefault("policy_gradient_clip_mode", "joint")
        normalized_options.setdefault("checkpoint_keep_recent", 2)
        normalized_options.setdefault("checkpoint_keep_best_valid", False)
        normalized_options.setdefault("actor_learning_rate", 1e-6)
        normalized["runtime_options"] = normalized_options
    return normalized


def _without_performance_candidates(
    config: Mapping[str, object],
) -> dict[str, object]:
    normalized = dict(config)
    app_config = normalized.get("app_config")
    if isinstance(app_config, Mapping):
        normalized_app = dict(app_config)
        normalized_app.pop("eval_batch_size", None)
        normalized["app_config"] = normalized_app
    evaluation_manifest = normalized.get("evaluation_manifest")
    if isinstance(evaluation_manifest, Mapping):
        normalized_manifest = dict(evaluation_manifest)
        normalized_manifest.pop("eval_batch_size", None)
        normalized_manifest.pop("comparison_role", None)
        normalized_manifest.pop("execution_mode", None)
        normalized["evaluation_manifest"] = normalized_manifest
    runtime_options = normalized.get("runtime_options")
    if not isinstance(runtime_options, Mapping):
        return normalized
    normalized_options = dict(runtime_options)
    normalized_options.pop("skip_unused_old_logprob_entropy", None)
    normalized_options.pop("rollout_max_batched_tokens", None)
    normalized_options.pop("hybrid_prefix_cuda_graph", None)
    normalized_options.pop("hybrid_prefix_cuda_graph_custom_kernels", None)
    normalized_options.pop("hybrid_prefix_cuda_graph_use_inductor", None)
    normalized_options.pop("lora_shrink_split_k_one", None)
    normalized_options.pop("fuse_kl_ppo_forward", None)
    normalized_options.pop("policy_gradient_clip_mode", None)
    # A named fork owns a new checkpoint directory, so its local retention
    # policy may change without mutating the source run. In-place resume may not.
    normalized_options.pop("checkpoint_keep_recent", None)
    normalized_options.pop("checkpoint_keep_best_valid", None)
    # A named experimental fork may intentionally change the LoRA optimizer
    # learning rate.  The checkpoint loader reapplies it after restoring the
    # optimizer/scheduler state; in-place resumes remain strict.
    normalized_options.pop("actor_learning_rate", None)
    normalized["runtime_options"] = normalized_options
    return normalized


def _matching_policy_model_id(
    previous: Mapping[str, object],
    current: Mapping[str, object],
) -> str | None:
    previous_app = previous.get("app_config")
    current_app = current.get("app_config")
    if not isinstance(previous_app, Mapping) or not isinstance(current_app, Mapping):
        return None
    previous_id = previous_app.get("policy_model_id")
    current_id = current_app.get("policy_model_id")
    if (
        isinstance(previous_id, str)
        and bool(previous_id.strip())
        and previous_id == current_id
    ):
        return previous_id
    return None


def _validate_checkpoint_policy_provenance(
    checkpoint: Path,
    model_id: str,
) -> None:
    path = checkpoint / "provenance.json"
    if not path.is_file():
        raise RuntimeError(f"resume checkpoint has no policy provenance: {path}")
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if not provenance_matches_pinned_model(provenance, model_id=model_id):
        raise RuntimeError(
            "resume checkpoint policy provenance differs from the registered model"
        )


def _without_policy_model_path(config: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(config)
    app_config = normalized.get("app_config")
    if not isinstance(app_config, Mapping):
        return normalized
    normalized_app = dict(app_config)
    paths = normalized_app.get("paths")
    if isinstance(paths, Mapping):
        normalized_paths = dict(paths)
        normalized_paths.pop("policy_model", None)
        normalized_app["paths"] = normalized_paths
    normalized["app_config"] = normalized_app
    return normalized


def _create_run_directory(output_root: str | Path, name: str) -> Path:
    if not _RUN_NAME.fullmatch(name):
        raise ValueError("run_name must contain only letters, digits, '.', '_' or '-'")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output_root).expanduser().resolve() / f"{stamp}-{name}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory
