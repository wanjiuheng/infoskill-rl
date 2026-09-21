from __future__ import annotations

import json
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Literal, Mapping, Sequence

from infoskill.app_config import AppConfig
from infoskill.conditioning import NoSkillConditioner, SkillConditioner
from infoskill.config import (
    DEFAULT_POLICY_MAX_TOKENS_PER_GPU,
    EvaluationConfig,
    SkillMode,
)
from infoskill.episode import TaskSpec, TrajectoryCollector
from infoskill.evaluation import (
    EvaluationCheckpointScore,
    EvaluationRunner,
    inherit_forked_checkpoint_selection,
    load_checkpoint_scores,
    select_best_valid,
    write_checkpoint_selection,
    write_valid_seen_learning_curve,
)
from infoskill.integrations.alfworld import (
    AlfworldEnvironmentFactory,
    discover_tasks,
    task_manifest_sha256,
)
from infoskill.learning import LogprobAlignmentError, group_relative_advantages
from infoskill.persistence import (
    CheckpointManager,
    MetricLogger,
    TrainerCheckpointState,
    TrainingTaskOutcomeWriter,
    ZstdJsonlTraceWriter,
)
from infoskill.persistence.model_identity import verify_policy_model_identity
from infoskill.rollout import GenerationParameters, PromptLengthError

from .drift_guard import TrainingDriftGuard
from .plan import TrainingPlan, TrainingProfile
from .rollout_curve import (
    load_training_rollout_step_scores,
    write_training_rollout_steps_curve,
)
from .run_directory import resolve_training_run_directory, validate_resume_config
from .schedule import TaskSchedule
from .trainer import InfoSkillTrainer, UpdateMetrics


EXPECTED_TRAIN_TASKS = 3_553


def _refresh_training_rollout_steps_curve(
    path: Path,
    *,
    metric_paths: Sequence[Path],
    max_step: int,
    logger: logging.Logger,
) -> bool:
    """Refresh the optional monitor without making training depend on it."""

    try:
        scores = load_training_rollout_step_scores(
            metric_paths,
            max_step=max_step,
        )
        if not scores:
            return False
        write_training_rollout_steps_curve(path, scores=scores)
        return True
    except Exception:
        logger.warning(
            "Unable to refresh training rollout step curve: %s",
            path,
            exc_info=True,
        )
        return False


def run_m0_training(
    *,
    config: AppConfig,
    plan: TrainingPlan,
    num_gpus: int,
    run_name: str | None,
    resume: str | None,
    persistent_rollout_session: bool = True,
    environment_workers: int = 1,
    environment_backend: str = "native_batch",
    verbose_runtime_logs: bool = False,
    cuda_memory_poll_interval_ms: int = 0,
    policy_max_tokens_per_gpu: int = DEFAULT_POLICY_MAX_TOKENS_PER_GPU,
    balance_policy_tokens_across_ranks: bool = True,
    skip_unused_old_logprob_entropy: bool = False,
    rollout_max_batched_tokens: int = 16_384,
    hybrid_prefix_cuda_graph: bool = False,
    lora_shrink_split_k_one: bool = False,
    fuse_kl_ppo_forward: bool = False,
    policy_gradient_clip_mode: Literal["joint", "separate"] = "joint",
    checkpoint_keep_recent: int = 2,
    checkpoint_keep_best_valid: bool = False,
    actor_learning_rate: float = 1e-6,
    segment_end_update: int | None = None,
) -> int:
    """Backward-compatible entry point for the token-only M0 baseline."""

    return run_policy_training(
        config=config,
        mode=SkillMode.NO_SKILL,
        plan=plan,
        num_gpus=num_gpus,
        run_name=run_name,
        resume=resume,
        segment_end_update=segment_end_update,
        persistent_rollout_session=persistent_rollout_session,
        environment_workers=environment_workers,
        environment_backend=environment_backend,
        verbose_runtime_logs=verbose_runtime_logs,
        cuda_memory_poll_interval_ms=cuda_memory_poll_interval_ms,
        policy_max_tokens_per_gpu=policy_max_tokens_per_gpu,
        balance_policy_tokens_across_ranks=balance_policy_tokens_across_ranks,
        skip_unused_old_logprob_entropy=skip_unused_old_logprob_entropy,
        rollout_max_batched_tokens=rollout_max_batched_tokens,
        hybrid_prefix_cuda_graph=hybrid_prefix_cuda_graph,
        lora_shrink_split_k_one=lora_shrink_split_k_one,
        fuse_kl_ppo_forward=fuse_kl_ppo_forward,
        policy_gradient_clip_mode=policy_gradient_clip_mode,
        checkpoint_keep_recent=checkpoint_keep_recent,
        checkpoint_keep_best_valid=checkpoint_keep_best_valid,
        actor_learning_rate=actor_learning_rate,
    )


def run_policy_training(
    *,
    config: AppConfig,
    mode: SkillMode,
    plan: TrainingPlan,
    num_gpus: int,
    run_name: str | None,
    resume: str | None,
    persistent_rollout_session: bool = True,
    environment_workers: int = 1,
    environment_backend: str = "native_batch",
    verbose_runtime_logs: bool = False,
    cuda_memory_poll_interval_ms: int = 0,
    policy_max_tokens_per_gpu: int = DEFAULT_POLICY_MAX_TOKENS_PER_GPU,
    balance_policy_tokens_across_ranks: bool = True,
    skip_unused_old_logprob_entropy: bool = False,
    rollout_max_batched_tokens: int = 16_384,
    hybrid_prefix_cuda_graph: bool = False,
    lora_shrink_split_k_one: bool = False,
    fuse_kl_ppo_forward: bool = False,
    policy_gradient_clip_mode: Literal["joint", "separate"] = "joint",
    raw_skill_prompt_format: Literal["compact", "full"] = "full",
    checkpoint_keep_recent: int = 2,
    checkpoint_keep_best_valid: bool = False,
    actor_learning_rate: float = 1e-6,
    drift_guard_ppo_kl_threshold: float | None = None,
    drift_guard_invalid_action_rate_threshold: float | None = None,
    drift_guard_consecutive_updates: int = 2,
    segment_end_update: int | None = None,
    warmstart_handoff: str | None = None,
) -> int:
    """Run one registered policy mode through the pinned VERL runtime."""

    if mode not in {
        SkillMode.NO_SKILL,
        SkillMode.RAW_SKILL_PROMPT,
        SkillMode.INFO_SKILL,
    }:
        raise ValueError(f"unsupported policy training mode: {mode.value}")
    if mode is SkillMode.INFO_SKILL and not config.paths.grounding_data:
        raise ValueError("infoskill training requires paths.grounding_data")
    if segment_end_update is not None and segment_end_update <= 0:
        raise ValueError("segment_end_update must be positive")
    if checkpoint_keep_recent <= 0:
        raise ValueError("checkpoint_keep_recent must be positive")
    if not math.isfinite(actor_learning_rate) or actor_learning_rate <= 0:
        raise ValueError("actor_learning_rate must be finite and positive")
    drift_guard_thresholds = (
        drift_guard_ppo_kl_threshold,
        drift_guard_invalid_action_rate_threshold,
    )
    if sum(value is not None for value in drift_guard_thresholds) == 1:
        raise ValueError("drift guard thresholds must be configured together")
    if drift_guard_consecutive_updates <= 0:
        raise ValueError("drift_guard_consecutive_updates must be positive")
    drift_guard = (
        TrainingDriftGuard(
            ppo_kl_threshold=float(drift_guard_ppo_kl_threshold),
            invalid_action_rate_threshold=float(
                drift_guard_invalid_action_rate_threshold
            ),
            consecutive_updates=drift_guard_consecutive_updates,
        )
        if drift_guard_ppo_kl_threshold is not None
        and drift_guard_invalid_action_rate_threshold is not None
        else None
    )
    drift_guard_config = (
        {
            "ppo_kl_threshold": drift_guard_ppo_kl_threshold,
            "invalid_action_rate_threshold": (
                drift_guard_invalid_action_rate_threshold
            ),
            "consecutive_updates": drift_guard_consecutive_updates,
        }
        if drift_guard is not None
        else None
    )
    if warmstart_handoff is not None and resume is not None:
        raise ValueError("warmstart_handoff and resume are mutually exclusive")
    if warmstart_handoff is not None and mode not in {
        SkillMode.NO_SKILL,
        SkillMode.INFO_SKILL,
    }:
        raise ValueError("actor imitation handoff is registered only for M0/M1")

    handoff_manifest: dict[str, object] | None = None
    handoff_directory: str | None = None

    if config.paths.policy_adapter is not None:
        raise ValueError(
            "formal policy training must start from the shared full-model "
            "initialization; "
            "policy_adapter must be null"
        )
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if environment_workers <= 0:
        raise ValueError("environment_workers must be positive")
    if environment_backend not in {"individual", "native_batch"}:
        raise ValueError("environment_backend must be individual or native_batch")
    if environment_backend == "native_batch" and environment_workers != 1:
        raise ValueError(
            "native_batch owns its process count; environment_workers must remain 1"
        )
    if cuda_memory_poll_interval_ms < 0:
        raise ValueError("cuda_memory_poll_interval_ms must be non-negative")
    minimum_token_budget = config.max_prompt_tokens + config.max_response_tokens
    if policy_max_tokens_per_gpu < minimum_token_budget:
        raise ValueError(
            "policy_max_tokens_per_gpu must be at least max_prompt_tokens + "
            f"max_response_tokens ({minimum_token_budget})"
        )
    if rollout_max_batched_tokens < minimum_token_budget:
        raise ValueError(
            "rollout_max_batched_tokens must be at least max_prompt_tokens + "
            f"max_response_tokens ({minimum_token_budget})"
        )
    if skip_unused_old_logprob_entropy and mode is not SkillMode.INFO_SKILL:
        raise ValueError(
            "skip_unused_old_logprob_entropy is registered only for infoskill"
        )
    if hybrid_prefix_cuda_graph and mode is not SkillMode.INFO_SKILL:
        raise ValueError(
            "hybrid_prefix_cuda_graph is registered only for infoskill"
        )
    if lora_shrink_split_k_one and mode is not SkillMode.INFO_SKILL:
        raise ValueError(
            "lora_shrink_split_k_one is registered only for infoskill"
        )
    if fuse_kl_ppo_forward and mode is not SkillMode.INFO_SKILL:
        raise ValueError(
            "fuse_kl_ppo_forward is registered only for infoskill"
        )
    if (
        policy_gradient_clip_mode != "joint"
        and mode is not SkillMode.INFO_SKILL
    ):
        raise ValueError(
            "separate policy gradient clipping is registered only for infoskill"
        )
    if (
        rollout_max_batched_tokens != 16_384
        and mode is not SkillMode.INFO_SKILL
    ):
        raise ValueError(
            "rollout_max_batched_tokens overrides are registered only for infoskill"
        )

    policy_model_identity = verify_policy_model_identity(
        config.paths.policy_model,
        model_id=config.policy_model_id,
    )
    if warmstart_handoff is not None:
        from infoskill.imitation.handoff import validate_handoff_for_runtime

        handoff_manifest, handoff_directory = validate_handoff_for_runtime(
            warmstart_handoff,
            policy_model_identity=policy_model_identity.as_dict(),
            skill_bank=(
                config.paths.skill_bank
                if mode is SkillMode.INFO_SKILL
                else None
            ),
            skill_bank_manifest=(
                config.paths.skill_bank_manifest
                if mode is SkillMode.INFO_SKILL
                else None
            ),
        )

    all_train_tasks = discover_tasks(config.paths.alfworld_data, split="train")
    if len(all_train_tasks) != EXPECTED_TRAIN_TASKS:
        raise RuntimeError(
            f"train discovery returned {len(all_train_tasks)} tasks instead of "
            f"{EXPECTED_TRAIN_TASKS}"
        )
    scheduled_tasks = all_train_tasks
    valid_seen_tasks: tuple[TaskSpec, ...] = ()
    valid_seen_manifest_sha256: str | None = None
    evaluation_config = EvaluationConfig()
    # Raw-skill checkpoints must carry an immutable retrieval plan for both the
    # train corpus and the fixed evaluation corpus.  Discover valid_seen even
    # for smoke/integration raw runs so their checkpoints can be evaluated
    # independently without introducing previously unseen retrieval queries.
    if (
        plan.evaluation_kind == "valid_seen"
        or mode in {SkillMode.RAW_SKILL_PROMPT, SkillMode.INFO_SKILL}
    ):
        valid_seen_tasks = discover_tasks(
            config.paths.alfworld_data,
            split=evaluation_config.split,
        )
        if len(valid_seen_tasks) != evaluation_config.total_tasks:
            raise RuntimeError(
                f"valid_seen discovery returned {len(valid_seen_tasks)} tasks instead "
                f"of {evaluation_config.total_tasks}"
            )
        valid_seen_manifest_sha256 = task_manifest_sha256(valid_seen_tasks)
        if valid_seen_manifest_sha256 != evaluation_config.manifest_sha256:
            raise RuntimeError(
                "valid_seen task manifest SHA256 does not match the registered "
                f"manifest: {valid_seen_manifest_sha256}"
            )
    available_updates = math.ceil(
        len(scheduled_tasks) / plan.task_groups_per_update
    )
    if plan.max_updates > available_updates:
        raise ValueError(
            f"training plan requests {plan.max_updates} updates but only "
            f"{available_updates} are available"
        )
    if plan.profile is TrainingProfile.FORMAL and available_updates != 445:
        raise RuntimeError(
            f"formal task schedule resolves to {available_updates} updates instead of 445"
        )

    skill_setup = None
    skill_provenance: dict[str, object] | None = None
    grounding_provenance: dict[str, object] | None = None
    retrieval_queries = {
        task.task_id: task.goal
        for task in (*all_train_tasks, *valid_seen_tasks)
    }
    if mode in {SkillMode.RAW_SKILL_PROMPT, SkillMode.INFO_SKILL}:
        from infoskill.builders import (
            audit_raw_skill_prompt_budget_for_model,
            build_raw_skill_setup,
        )

        skill_setup = build_raw_skill_setup(
            config,
            retrieval_queries=retrieval_queries,
            prompt_format=raw_skill_prompt_format,
        )
        if mode is SkillMode.RAW_SKILL_PROMPT:
            skill_provenance = {
                **skill_setup.provenance,
                **audit_raw_skill_prompt_budget_for_model(
                    skill_setup,
                    model_path=config.paths.policy_model,
                    max_prompt_tokens=config.max_prompt_tokens,
                ),
            }
            training_conditioner: SkillConditioner | None = skill_setup.conditioner
        else:
            from infoskill.integrations.alfworld import GroundingDataset
            from infoskill.integrations.alfworld.grounding_io import sha256_file

            grounding = GroundingDataset.load(config.paths.grounding_data)
            source_checksums = grounding.manifest.get("source_checksums", {})
            if not isinstance(source_checksums, dict):
                raise ValueError("grounding source_checksums must be an object")
            expected_skill_bank = source_checksums.get("skill_bank")
            actual_skill_bank = sha256_file(config.paths.skill_bank)
            if (
                grounding.manifest.get("derived_candidate_skill_ids") is True
                and expected_skill_bank != actual_skill_bank
            ):
                raise ValueError(
                    "grounding candidate skill IDs are not bound to the configured "
                    "skill bank"
                )
            grounding_provenance = {
                "root": str(grounding.root),
                "manifest_sha256": grounding.manifest_sha256,
                "game_count": grounding.game_count,
                "sample_count": grounding.sample_count,
                "source_split": grounding.manifest.get("source_split"),
                "formal_gate_passed": grounding.manifest.get(
                    "formal_gate_passed"
                ),
                "skill_bank_sha256": actual_skill_bank,
            }
            skill_provenance = {
                **skill_setup.provenance,
                "prompt_format": "continuous_soft_prefix_v1",
                "soft_prefix_length": 5,
                "latent_dim": 32,
                "latent_train_mode": "sample",
                "latent_eval_mode": "mean",
                "dynamic_skill_updates": False,
            }
            training_conditioner = None
    else:
        training_conditioner = NoSkillConditioner()
    run_directory, checkpoint_to_load, forked_resume = resolve_training_run_directory(
        output_root=config.paths.output_root,
        profile_name=plan.profile.value,
        run_name=run_name,
        resume=resume,
        default_run_name=(
            f"m0-{plan.profile.value}"
            if mode is SkillMode.NO_SKILL
            else (
                f"raw-skill-prompt-{plan.profile.value}"
                if mode is SkillMode.RAW_SKILL_PROMPT
                else f"infoskill-{plan.profile.value}"
            )
        ),
    )
    logger = _configure_training_logging(run_directory)
    resolved = {
        "schema_version": 1,
        "mode": mode.value,
        "num_gpus": num_gpus,
        "app_config": config.as_dict(),
        "training_plan": _plan_payload(plan),
        "runtime_options": {
            "persistent_rollout_session": persistent_rollout_session,
            "cross_step_prefix_cache": False,
            "environment_workers": environment_workers,
            "environment_backend": environment_backend,
            "cuda_memory_poll_interval_ms": cuda_memory_poll_interval_ms,
            "policy_max_tokens_per_gpu": policy_max_tokens_per_gpu,
            "balance_policy_tokens_across_ranks": (
                balance_policy_tokens_across_ranks
            ),
            "skip_unused_old_logprob_entropy": (
                skip_unused_old_logprob_entropy
            ),
            "rollout_max_batched_tokens": rollout_max_batched_tokens,
            "hybrid_prefix_cuda_graph": hybrid_prefix_cuda_graph,
            "hybrid_prefix_cuda_graph_custom_kernels": (
                hybrid_prefix_cuda_graph
            ),
            "hybrid_prefix_cuda_graph_use_inductor": (
                False if hybrid_prefix_cuda_graph else None
            ),
            "lora_shrink_split_k_one": lora_shrink_split_k_one,
            "fuse_kl_ppo_forward": fuse_kl_ppo_forward,
            "policy_gradient_clip_mode": policy_gradient_clip_mode,
            "checkpoint_keep_recent": checkpoint_keep_recent,
            "checkpoint_keep_best_valid": checkpoint_keep_best_valid,
            "actor_learning_rate": actor_learning_rate,
            "training_drift_guard": drift_guard_config,
            "warmstart_handoff": handoff_directory,
            "infoskill_auxiliary_enabled": mode is SkillMode.INFO_SKILL,
            "infoskill_auxiliary_micro_batch_size": (
                8 if mode is SkillMode.INFO_SKILL else None
            ),
            "infoskill_offline_games_per_update": (
                256 if mode is SkillMode.INFO_SKILL else None
            ),
        },
        "evaluation_manifest": (
            {
                "split": evaluation_config.split,
                "task_count": evaluation_config.total_tasks,
                "sha256": valid_seen_manifest_sha256,
                "eval_batch_size": config.eval_batch_size,
                "comparison_role": (
                    "registered_batch8"
                    if config.eval_batch_size == 8
                    else "nonregistered_monitoring_curve"
                ),
                "execution_mode": (
                    _rollout_execution_mode(
                        hybrid_prefix_cuda_graph=hybrid_prefix_cuda_graph,
                        lora_shrink_split_k_one=lora_shrink_split_k_one,
                    )
                ),
            }
            if plan.evaluation_kind == "valid_seen"
            else None
        ),
    }
    if skill_provenance is not None:
        resolved["skill_conditioning"] = skill_provenance
    if grounding_provenance is not None:
        resolved["grounding_data"] = grounding_provenance
    resume_source_num_gpus: int | None = None
    if checkpoint_to_load is not None:
        resume_source_num_gpus = validate_resume_config(
            checkpoint_to_load,
            resolved,
            allow_gpu_change=forked_resume,
            allow_performance_candidate_change=forked_resume,
        )
    if checkpoint_to_load is None or forked_resume:
        _write_json(run_directory / "resolved_config.json", resolved)

    schedule = TaskSchedule(
        scheduled_tasks,
        master_seed=config.master_seed,
        passes=1,
    )
    checkpoints = CheckpointManager(
        run_directory / "checkpoints",
        keep_recent=checkpoint_keep_recent,
    )
    restored = (
        checkpoints.load_trainer_state(checkpoint_to_load)
        if checkpoint_to_load is not None
        else None
    )
    initial_global_update = restored.global_update if restored is not None else 0
    if segment_end_update is not None:
        if segment_end_update <= initial_global_update:
            raise ValueError(
                "segment_end_update must be greater than the resumed global update"
            )
        if segment_end_update > plan.max_updates:
            raise ValueError(
                "segment_end_update cannot exceed the registered training target"
            )
    if restored is not None:
        schedule.restore(restored.schedule)

    provenance = {
        "schema_version": 1,
        "mode": mode.value,
        "train_task_count": len(all_train_tasks),
        "scheduled_task_count": len(scheduled_tasks),
        "train_task_manifest_sha256": task_manifest_sha256(all_train_tasks),
        "valid_seen_task_manifest_sha256": valid_seen_manifest_sha256,
        "skillrl_expected_commit": "8e66726ed866a4e0a7f053586a41022798192e6c",
        "policy_model": policy_model_identity.as_dict(),
        "verbose_runtime_logs": verbose_runtime_logs,
        "invocation": {
            "segment_start_update": initial_global_update,
            "segment_end_update": segment_end_update,
            "actor_learning_rate": actor_learning_rate,
            "training_drift_guard": drift_guard_config,
        },
        "checkpoint_retention": {
            "schema_version": 1,
            "keep_recent": checkpoint_keep_recent,
            "keep_best_valid": checkpoint_keep_best_valid,
            "preserve_every_evaluation_milestone": (
                not checkpoint_keep_best_valid
            ),
            "always_preserve_final": True,
        },
        "evaluation_protocol": (
            {
                "eval_batch_size": config.eval_batch_size,
                "comparison_role": (
                    "registered_batch8"
                    if config.eval_batch_size == 8
                    else "nonregistered_monitoring_curve"
                ),
                "execution_mode": (
                    _rollout_execution_mode(
                        hybrid_prefix_cuda_graph=hybrid_prefix_cuda_graph,
                        lora_shrink_split_k_one=lora_shrink_split_k_one,
                    )
                ),
            }
            if plan.evaluation_kind == "valid_seen"
            else None
        ),
    }
    if skill_provenance is not None:
        provenance["skill_conditioning"] = skill_provenance
    if grounding_provenance is not None:
        provenance["grounding_data"] = grounding_provenance
        provenance["infoskill_initialization"] = {
            "schema_version": 1,
            "initialization_seed": config.master_seed,
            "latent_dim": 32,
            "soft_prefix_length": 5,
            "projector_initial_gate": 0.01,
            "fidelity_weight": 1.0,
            "rate_weight": 0.001,
            "grounding_weight": 0.1,
        }
    if handoff_manifest is not None:
        provenance["actor_imitation_warmstart"] = {
            "directory": handoff_directory,
            "handoff_sha256": handoff_manifest["handoff_sha256"],
            "adapter_model_sha256": handoff_manifest["adapter_model_sha256"],
            "optimizer_state_loaded": False,
            "infoskill_state_loaded": False,
        }
    if checkpoint_to_load is not None:
        provenance.update(
            {
                "resume_source_checkpoint": str(checkpoint_to_load),
                "resume_source_num_gpus": resume_source_num_gpus,
                "resume_forked": forked_resume,
            }
        )
    _write_json(run_directory / "provenance.json", provenance)

    if (
        forked_resume
        and restored is not None
        and checkpoint_to_load is not None
        and plan.evaluation_kind == "valid_seen"
    ):
        if valid_seen_manifest_sha256 is None:
            raise RuntimeError(
                "forked valid_seen resume requires a registered manifest identity"
            )
        inherit_forked_checkpoint_selection(
            source_run=checkpoint_to_load.parent.parent,
            destination_run=run_directory,
            max_source_step=restored.global_update,
            task_manifest_sha256=valid_seen_manifest_sha256,
            eval_batch_size=config.eval_batch_size,
            comparison_role=(
                "registered_batch8"
                if config.eval_batch_size == 8
                else "nonregistered_monitoring_curve"
            ),
            execution_mode=(
                _rollout_execution_mode(
                    hybrid_prefix_cuda_graph=hybrid_prefix_cuda_graph,
                    lora_shrink_split_k_one=lora_shrink_split_k_one,
                )
            ),
        )

    from infoskill.integrations.verl import VerlRuntime, VerlRuntimeConfig

    if VerlRuntime is None or VerlRuntimeConfig is None:
        raise RuntimeError("the pinned VERL runtime is unavailable")
    logger.info(
        "Initializing Ray/FSDP/vLLM runtime on %d GPU(s); worker logs=%s",
        num_gpus,
        "verbose" if verbose_runtime_logs else "quiet",
    )
    runtime_started = time.perf_counter()
    runtime = VerlRuntime.start(
        VerlRuntimeConfig(
            skillrl_source=config.paths.skillrl_source,
            model_path=config.paths.policy_model,
            num_gpus=num_gpus,
            num_cpus=96,
            max_prompt_tokens=config.max_prompt_tokens,
            max_response_tokens=config.max_response_tokens,
            total_training_steps=plan.max_updates,
            actor_learning_rate=actor_learning_rate,
            action_minibatch_size=plan.action_minibatch_size,
            policy_max_tokens_per_gpu=policy_max_tokens_per_gpu,
            rollout_max_batched_tokens=rollout_max_batched_tokens,
            hybrid_prefix_cuda_graph=hybrid_prefix_cuda_graph,
            lora_shrink_split_k_one=lora_shrink_split_k_one,
            fuse_kl_ppo_forward=fuse_kl_ppo_forward,
            infoskill_policy_gradient_clip_mode=policy_gradient_clip_mode,
            balance_policy_tokens_across_ranks=(
                balance_policy_tokens_across_ranks
            ),
            skip_unused_old_logprob_entropy=(
                skip_unused_old_logprob_entropy
            ),
            gpu_memory_utilization=0.45,
            require_hybrid_prefix=mode is SkillMode.INFO_SKILL,
            soft_prefix_length=5,
            master_seed=config.master_seed,
            persistent_rollout_session=persistent_rollout_session,
            verbose_runtime_logs=verbose_runtime_logs,
            cuda_memory_poll_interval_ms=cuda_memory_poll_interval_ms,
            enable_infoskill_modules=mode is SkillMode.INFO_SKILL,
            semantic_model_path=(
                config.paths.semantic_model
                if mode is SkillMode.INFO_SKILL
                else None
            ),
            skill_bank_path=(
                config.paths.skill_bank
                if mode is SkillMode.INFO_SKILL
                else None
            ),
            enable_infoskill_auxiliary=mode is SkillMode.INFO_SKILL,
            actor_warmstart_directory=handoff_directory,
            grounding_data_path=(
                config.paths.grounding_data
                if mode is SkillMode.INFO_SKILL
                else None
            ),
            infoskill_history_length=config.history_length,
        )
    )
    if handoff_directory is not None:
        _write_json(
            run_directory / "warmstart-load.json",
            {
                "schema_version": 1,
                "status": "loaded",
                "handoff": handoff_directory,
                "handoff_sha256": handoff_manifest["handoff_sha256"],
                "worker_reports": list(runtime.warmstart_load_reports),
            },
        )
    logger.info("Runtime ready in %.1f seconds", time.perf_counter() - runtime_started)
    trainer: InfoSkillTrainer | None = None
    pause_requested = False
    final_control_status: str | None = None
    control_path = run_directory / "training-control.json"
    drift_guard_path = run_directory / "training-drift-guard.json"
    previous_signal_handlers: dict[int, object] = {}

    if drift_guard is not None:
        _write_json(drift_guard_path, drift_guard.as_dict())

    def request_pause(signal_number: int, _frame: object) -> None:
        nonlocal pause_requested
        pause_requested = True

    def pause_reason() -> str | None:
        if pause_requested:
            return "signal"
        if drift_guard is not None and drift_guard.triggered:
            return "training_drift_guard"
        if (
            segment_end_update is not None
            and trainer is not None
            and trainer.global_update >= segment_end_update
        ):
            return "segment_end_update"
        return None

    def write_training_control(status: str) -> None:
        _write_json(
            control_path,
            {
                "schema_version": 1,
                "pid": os.getpid(),
                "status": status,
                "global_update": (
                    trainer.global_update if trainer is not None else 0
                ),
                "checkpoint_every": plan.checkpoint_every,
                "pause_signal": "SIGINT or SIGTERM",
                "pause_reason": pause_reason(),
                "training_drift_guard": (
                    drift_guard.as_dict() if drift_guard is not None else None
                ),
                "resume_from": (
                    str(checkpoint_to_load)
                    if checkpoint_to_load is not None
                    else None
                ),
            },
        )

    try:
        for pause_signal in (signal.SIGINT, signal.SIGTERM):
            previous_signal_handlers[pause_signal] = signal.getsignal(pause_signal)
            signal.signal(pause_signal, request_pause)
        write_training_control("running")
    except BaseException:
        for pause_signal, previous_handler in previous_signal_handlers.items():
            signal.signal(pause_signal, previous_handler)
        runtime.close()
        raise
    logger.info(
        "Training process pid=%d; SIGINT/SIGTERM requests a checkpoint-boundary pause",
        os.getpid(),
    )
    try:
        evaluation_conditioner = training_conditioner
        if mode is SkillMode.INFO_SKILL:
            from infoskill.conditioning import RuntimeInfoSkillConditioner

            if skill_setup is None or skill_setup.retriever is None:
                raise RuntimeError("infoskill retrieval setup is incomplete")
            training_conditioner = RuntimeInfoSkillConditioner(
                retriever=skill_setup.retriever,
                runtime=runtime,
                latent_mode="sample",
                query_by_task_id=retrieval_queries,
            )
            evaluation_conditioner = RuntimeInfoSkillConditioner(
                retriever=skill_setup.retriever,
                runtime=runtime,
                latent_mode="mean",
                query_by_task_id=retrieval_queries,
            )
        if training_conditioner is None or evaluation_conditioner is None:
            raise RuntimeError("training conditioner was not initialized")
        factory = AlfworldEnvironmentFactory.from_paths(
            alfworld_source=config.paths.alfworld_source,
            config_path=config.paths.alfworld_config,
            data_root=config.paths.alfworld_data,
            max_steps=config.max_steps,
        )
        training_collector = _collector(
            config,
            factory=factory,
            runtime=runtime,
            conditioner=training_conditioner,
            training=True,
            environment_workers=environment_workers,
            environment_backend=environment_backend,
        )
        evaluation_collector = _collector(
            config,
            factory=factory,
            runtime=runtime,
            conditioner=evaluation_conditioner,
            training=False,
            environment_workers=environment_workers,
            environment_backend=environment_backend,
        )
        if checkpoint_to_load is not None:
            runtime.load_portable_state(checkpoint_to_load / "runtime")

        traces = ZstdJsonlTraceWriter(run_directory)
        metrics = MetricLogger(run_directory)
        initial_update = restored.global_update if restored is not None else 0
        training_metric_paths: tuple[Path, ...] = (metrics.jsonl_path,)
        if forked_resume and checkpoint_to_load is not None:
            source_metrics = checkpoint_to_load.parent.parent / "metrics.jsonl"
            training_metric_paths = (source_metrics, metrics.jsonl_path)
        training_curve_path = (
            run_directory / "training_rollout_steps_curve.svg"
        )
        _refresh_training_rollout_steps_curve(
            training_curve_path,
            metric_paths=training_metric_paths,
            max_step=initial_update,
            logger=logger,
        )
        task_outcomes = TrainingTaskOutcomeWriter(
            run_directory,
            expected_rollouts_per_task=plan.rollouts_per_task,
            committed_through_update=initial_update,
        )
        from tqdm.auto import tqdm

        progress = tqdm(
            total=plan.max_updates,
            initial=initial_update,
            desc=f"{mode.value}/{plan.profile.value}",
            unit="update",
            dynamic_ncols=True,
        )

        def checkpoint(update: int, current_schedule: TaskSchedule) -> None:
            path = checkpoints.save(
                state=TrainerCheckpointState(
                    global_update=update,
                    schedule=current_schedule.state(),
                    semantic_counters={},
                ),
                runtime=runtime,
                resolved_config=resolved,
                provenance=provenance,
                permanent=(
                    _checkpoint_is_permanent(
                        update=update,
                        max_updates=plan.max_updates,
                        keep_best_valid=checkpoint_keep_best_valid,
                    )
                ),
            )
            logger.info("Committed checkpoint: %s", path)

        def update_callback(
            update: UpdateMetrics,
            groups: tuple,
        ) -> None:
            _require_finite_metrics(update.values)
            guard_was_triggered = (
                drift_guard.triggered if drift_guard is not None else False
            )
            guard_observation = (
                drift_guard.observe(update)
                if drift_guard is not None
                else None
            )
            if drift_guard is not None:
                _write_json(drift_guard_path, drift_guard.as_dict())
            if (
                guard_observation is not None
                and guard_observation.triggered
                and not guard_was_triggered
            ):
                logger.warning(
                    "Training drift guard triggered at update=%d: "
                    "ppo_kl=%.6f invalid=%.6f consecutive=%d; "
                    "committing a recovery checkpoint before pause",
                    update.global_update,
                    guard_observation.ppo_kl,
                    guard_observation.invalid_action_rate,
                    guard_observation.consecutive_breaches,
                )
            advantages = tuple(
                group_relative_advantages(
                    [trajectory.reward for trajectory in group.trajectories]
                )
                for group in groups
            )
            trace_path = traces.write_training_update(
                global_update=update.global_update,
                groups=groups,
                advantages=advantages,
            )
            outcome_counts = task_outcomes.write_training_update(
                global_update=update.global_update,
                groups=groups,
                trace_path=trace_path,
            )
            values: dict[str, float | int | str | bool | None] = dict(
                update.values
            )
            values["schedule/cursor"] = schedule.cursor
            values["schedule/total"] = schedule.total
            values["trace"] = str(trace_path)
            if guard_observation is not None:
                values.update(
                    {
                        "drift_guard/breached": guard_observation.breached,
                        "drift_guard/consecutive_breaches": (
                            guard_observation.consecutive_breaches
                        ),
                        "drift_guard/triggered": guard_observation.triggered,
                        "drift_guard/trigger_update": (
                            guard_observation.trigger_update
                        ),
                    }
                )
            values.update(
                {
                    f"task_outcomes/{key}": value
                    for key, value in outcome_counts.items()
                }
            )
            metrics.log(step=update.global_update, phase="train", values=values)
            _refresh_training_rollout_steps_curve(
                training_curve_path,
                metric_paths=training_metric_paths,
                max_step=update.global_update,
                logger=logger,
            )
            logger.info(
                "update=%d/%d success=%.4f reward=%.4f invalid=%.4f",
                update.global_update,
                plan.max_updates,
                update.values.get("rollout/success_rate", float("nan")),
                update.values.get("rollout/mean_reward", float("nan")),
                update.values.get("rollout/invalid_action_rate", float("nan")),
            )
            progress.update(1)
            write_training_control(
                "pause_requested" if pause_reason() is not None else "running"
            )

        evaluate = _evaluation_callback(
            config=config,
            plan=plan,
            tasks=valid_seen_tasks,
            task_manifest_sha256_value=valid_seen_manifest_sha256,
            collector=evaluation_collector,
            run_directory=run_directory,
            traces=traces,
            metrics=metrics,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_manager=checkpoints,
            keep_best_valid=checkpoint_keep_best_valid,
            schedule=schedule,
            hybrid_prefix_cuda_graph=hybrid_prefix_cuda_graph,
            lora_shrink_split_k_one=lora_shrink_split_k_one,
        )

        def should_pause() -> bool:
            return pause_reason() is not None

        trainer = InfoSkillTrainer(
            collector=training_collector,
            runtime=runtime,
            schedule=schedule,
            task_groups_per_update=plan.task_groups_per_update,
            rollouts_per_task=plan.rollouts_per_task,
            master_seed=config.master_seed,
            auxiliary_enabled=mode is SkillMode.INFO_SKILL,
            on_update=update_callback,
            on_evaluate=evaluate,
            on_checkpoint=checkpoint,
            should_pause=should_pause,
            evaluate_every=plan.evaluation_every,
            checkpoint_every=plan.checkpoint_every,
        )
        if restored is not None:
            trainer.restore(
                global_update=restored.global_update,
                schedule_state=restored.schedule,
            )
            logger.info(
                "Resumed checkpoint=%s next_update=%d task_cursor=%d",
                checkpoint_to_load,
                trainer.global_update,
                schedule.cursor,
            )
        try:
            trainer.fit(
                max_updates=plan.max_updates,
                evaluate_at_start=(restored is None and evaluate is not None),
            )
        finally:
            progress.close()
        summary = {
            "schema_version": 1,
            "status": "paused" if trainer.paused else "complete",
            "mode": mode.value,
            "profile": plan.profile.value,
            "global_update": trainer.global_update,
            "task_cursor": schedule.cursor,
            "max_updates": plan.max_updates,
            "pause_reason": pause_reason() if trainer.paused else None,
            "training_drift_guard": (
                drift_guard.as_dict() if drift_guard is not None else None
            ),
        }
        if drift_guard is not None:
            _write_json(drift_guard_path, drift_guard.as_dict())
        _write_json(run_directory / "training_summary.json", summary)
        final_control_status = str(summary["status"])
        if trainer.paused:
            logger.info(
                "Pause completed at update=%d reason=%s; "
                "resume from checkpoints/step-%06d",
                trainer.global_update,
                pause_reason(),
                trainer.global_update,
            )
        logger.info("Policy training segment complete: %s", json.dumps(summary))
        return 0
    except Exception as error:
        if isinstance(error, PromptLengthError):
            _write_json(
                run_directory / "prompt-overflow.json",
                {
                    "schema_version": 1,
                    "mode": mode.value,
                    **error.as_dict(),
                },
            )
        if isinstance(error, LogprobAlignmentError):
            diagnostic_path = _persist_logprob_alignment_failure(
                run_directory=run_directory,
                error=error,
                mode=mode,
                attempted_global_update=(
                    trainer.global_update if trainer is not None else 0
                ),
            )
            logger.error(
                "Persisted rollout/recompute alignment diagnostics: %s",
                diagnostic_path,
            )
        # The uncaught exception below prints the complete traceback once.
        logger.error("Policy training failed: %s", error)
        final_control_status = "failed"
        raise
    finally:
        try:
            runtime.close()
        except BaseException:
            final_control_status = "failed"
            raise
        finally:
            for pause_signal, previous_handler in previous_signal_handlers.items():
                signal.signal(pause_signal, previous_handler)
            try:
                write_training_control(final_control_status or "failed")
            except OSError:
                pass


def _collector(
    config: AppConfig,
    *,
    factory: AlfworldEnvironmentFactory,
    runtime: object,
    conditioner: SkillConditioner,
    training: bool,
    environment_workers: int,
    environment_backend: str,
) -> TrajectoryCollector:
    parameters = GenerationParameters(
        do_sample=training,
        temperature=1.0 if training else 0.0,
        top_p=1.0,
        max_new_tokens=config.max_response_tokens,
    )
    return TrajectoryCollector(
        environment_factory=factory,
        conditioner=conditioner,
        rollout_backend=runtime,  # type: ignore[arg-type]
        max_steps=config.max_steps,
        history_limit=config.history_length,
        invalid_action_penalty=0.01,
        generation_parameters=parameters,
        environment_workers=environment_workers,
        environment_backend=environment_backend,  # type: ignore[arg-type]
    )


def _evaluation_callback(
    *,
    config: AppConfig,
    plan: TrainingPlan,
    tasks: Sequence[TaskSpec],
    task_manifest_sha256_value: str | None,
    collector: TrajectoryCollector,
    run_directory: Path,
    traces: ZstdJsonlTraceWriter,
    metrics: MetricLogger,
    logger: logging.Logger,
    checkpoint,
    checkpoint_manager: CheckpointManager,
    keep_best_valid: bool,
    schedule: TaskSchedule,
    hybrid_prefix_cuda_graph: bool,
    lora_shrink_split_k_one: bool,
):
    if plan.evaluation_kind == "none":
        return None
    if plan.evaluation_kind != "valid_seen":
        raise ValueError(f"unsupported evaluation kind: {plan.evaluation_kind}")
    if task_manifest_sha256_value is None:
        raise RuntimeError("valid_seen evaluation requires a registered manifest identity")
    evaluation_config = EvaluationConfig()
    phase = "valid_seen"
    selection_path = run_directory / "checkpoint_selection.json"
    curve_path = run_directory / "valid_seen_learning_curve.svg"
    valid_scores = load_checkpoint_scores(
        selection_path,
        expected_manifest_sha256=task_manifest_sha256_value,
    )
    if keep_best_valid:
        _sync_best_valid_checkpoint(
            checkpoint_manager=checkpoint_manager,
            run_directory=run_directory,
            scores=valid_scores,
        )
    monitoring_only = config.eval_batch_size != 8
    execution_mode = _rollout_execution_mode(
        hybrid_prefix_cuda_graph=hybrid_prefix_cuda_graph,
        lora_shrink_split_k_one=lora_shrink_split_k_one,
    )
    if valid_scores:
        write_valid_seen_learning_curve(
            curve_path,
            scores=valid_scores,
            eval_batch_size=config.eval_batch_size,
            monitoring_only=monitoring_only,
        )

    def evaluate(global_update: int) -> None:
        from tqdm.auto import tqdm

        evaluation_progress = tqdm(
            total=len(tasks),
            desc=f"{phase}@{global_update}",
            unit="task",
            dynamic_ncols=True,
            leave=False,
        )
        runner = EvaluationRunner(
            collector_factory=lambda: collector,
            config=evaluation_config,
            task_batch_size=config.eval_batch_size,
            master_seed=config.master_seed,
            on_progress=evaluation_progress.update,
        )
        try:
            with collector.rollout_session():
                run = runner.run(tasks, checkpoint_step=global_update)
        finally:
            evaluation_progress.close()
        trace_path = traces.write_evaluation(
            checkpoint_step=global_update,
            run=run,
            split=phase,
        )
        summary = run.summary
        values: dict[str, float | int | str | bool | None] = {
            "complete": summary.is_complete,
            "evaluated": summary.evaluated,
            "overall_success": summary.overall_success,
            "macro_success": summary.macro_success,
            "invalid_action_rate": summary.invalid_action_rate,
            "mean_steps": summary.mean_steps,
            "incomplete_reasons": ";".join(summary.incomplete_reasons),
            "trace": str(trace_path),
            "task_manifest_sha256": task_manifest_sha256_value,
            "eval_batch_size": config.eval_batch_size,
            "comparison_role": (
                "nonregistered_monitoring_curve"
                if monitoring_only
                else "registered_batch8"
            ),
            "execution_mode": execution_mode,
        }
        values.update(
            {
                f"success/{key}": value
                for key, value in summary.per_task_type_success.items()
            }
        )
        metrics.log(step=global_update, phase=phase, values=values)
        _write_json(
            run_directory / f"{phase}-{global_update:06d}-summary.json",
            values,
        )
        if not summary.is_complete:
            raise RuntimeError(
                f"{phase} evaluation is incomplete: {summary.incomplete_reasons}"
            )
        assert summary.macro_success is not None
        assert summary.overall_success is not None
        assert summary.invalid_action_rate is not None
        valid_scores.append(
            EvaluationCheckpointScore(
                step=global_update,
                macro_success=summary.macro_success,
                overall_success=summary.overall_success,
                invalid_action_rate=summary.invalid_action_rate,
                checkpoint=f"checkpoints/step-{global_update:06d}",
                eval_batch_size=config.eval_batch_size,
                comparison_role=(
                    "nonregistered_monitoring_curve"
                    if monitoring_only
                    else "registered_batch8"
                ),
                execution_mode=execution_mode,
            )
        )
        write_checkpoint_selection(
            selection_path,
            scores=valid_scores,
            task_manifest_sha256=task_manifest_sha256_value,
            eval_batch_size=config.eval_batch_size,
            comparison_role=(
                "nonregistered_monitoring_curve"
                if monitoring_only
                else "registered_batch8"
            ),
            execution_mode=execution_mode,
        )
        if keep_best_valid:
            _sync_best_valid_checkpoint(
                checkpoint_manager=checkpoint_manager,
                run_directory=run_directory,
                scores=valid_scores,
            )
        write_valid_seen_learning_curve(
            curve_path,
            scores=valid_scores,
            eval_batch_size=config.eval_batch_size,
            monitoring_only=monitoring_only,
        )
        logger.info(
            "%s update=%d macro=%s overall=%s",
            phase,
            global_update,
            summary.macro_success,
            summary.overall_success,
        )
        if global_update == 0:
            checkpoint(0, schedule)

    return evaluate


def _rollout_execution_mode(
    *,
    hybrid_prefix_cuda_graph: bool,
    lora_shrink_split_k_one: bool,
) -> str:
    base = "cuda_graph" if hybrid_prefix_cuda_graph else "eager"
    return f"{base}_split_k_one" if lora_shrink_split_k_one else base


def _checkpoint_is_permanent(
    *,
    update: int,
    max_updates: int,
    keep_best_valid: bool,
) -> bool:
    return (
        update == 0
        or update == max_updates
        or (not keep_best_valid and update % 25 == 0)
    )


def _sync_best_valid_checkpoint(
    *,
    checkpoint_manager: CheckpointManager,
    run_directory: Path,
    scores: Sequence[EvaluationCheckpointScore],
) -> None:
    if not scores:
        checkpoint_manager.set_best_checkpoint(None)
        return
    best = select_best_valid(scores)
    checkpoint_path = Path(
        best.checkpoint or f"checkpoints/step-{best.step:06d}"
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = (run_directory / checkpoint_path).resolve()
    else:
        checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.parent != checkpoint_manager.root:
        # Forked selection history may point at the immutable source run. The
        # destination manager must never delete or relabel source checkpoints.
        checkpoint_manager.set_best_checkpoint(None)
        return
    if not checkpoint_path.is_dir():
        if best.step == 0:
            # Fresh update-0 evaluation is written before its permanent
            # checkpoint. There is no retention gap because step 0 is permanent.
            checkpoint_manager.set_best_checkpoint(None)
            return
        raise RuntimeError(
            f"best valid checkpoint is missing from the active run: {checkpoint_path}"
        )
    checkpoint_manager.set_best_checkpoint(checkpoint_path)


def _require_finite_metrics(values: Mapping[str, float]) -> None:
    non_finite = [
        key for key, value in values.items() if not math.isfinite(float(value))
    ]
    if non_finite:
        raise RuntimeError(
            "training produced non-finite metrics: " + ", ".join(non_finite)
        )


def _plan_payload(plan: TrainingPlan) -> dict[str, object]:
    return {
        "profile": plan.profile.value,
        "max_updates": plan.max_updates,
        "task_groups_per_update": plan.task_groups_per_update,
        "rollouts_per_task": plan.rollouts_per_task,
        "trajectories_per_full_update": plan.trajectories_per_full_update,
        "action_minibatch_size": plan.action_minibatch_size,
        "checkpoint_every": plan.checkpoint_every,
        "evaluation_every": plan.evaluation_every,
        "evaluation_kind": plan.evaluation_kind,
    }


def _configure_training_logging(run_directory: Path) -> logging.Logger:
    logger = logging.getLogger("infoskill.training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(run_directory / "console.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _persist_logprob_alignment_failure(
    *,
    run_directory: Path,
    error: LogprobAlignmentError,
    mode: SkillMode,
    attempted_global_update: int,
) -> Path:
    path = run_directory / "rollout-recompute-alignment-failure.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "mode": mode.value,
            "attempted_global_update": attempted_global_update,
            **error.as_dict(),
        },
    )
    return path
