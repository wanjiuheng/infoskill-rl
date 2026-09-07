from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from infoskill.app_config import AppConfig
from infoskill.config import EvaluationConfig, SkillMode
from infoskill.training import TrainingProfile, resolve_training_plan


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = AppConfig.load(args.config)
    retrieval_mode = getattr(args, "retrieval_mode", None)
    if retrieval_mode is not None:
        config = replace(config, retrieval_mode=retrieval_mode)
    if args.command == "validate":
        mode = SkillMode(args.mode)
        _validate_paths(
            config,
            mode=mode,
            require_checkpoint=mode is SkillMode.INFO_SKILL,
        )
        print(json.dumps(config.as_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "eval":
        return _evaluate(config, args)
    if args.command == "raw-skill-ab":
        return _raw_skill_ab(config, args)
    if args.command == "checkpoint-effect":
        return _checkpoint_effect(config, args)
    if args.command == "grounding":
        return _grounding(config, args)
    if args.command == "train":
        return _train(config, args)
    parser.error(f"unsupported command: {args.command}")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="infoskill", description="INFO-SKILL experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate config and local paths")
    validate.add_argument("--config", required=True)
    validate.add_argument("--mode", choices=[mode.value for mode in SkillMode], default="no_skill")
    _add_retrieval_mode_argument(validate)
    evaluate = subparsers.add_parser("eval", help="run the complete ALFWorld valid_seen evaluation")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--mode", choices=[mode.value for mode in SkillMode], required=True)
    _add_retrieval_mode_argument(evaluate)
    evaluate.add_argument("--run-name")
    evaluate.add_argument("--checkpoint-step", type=int, default=0)
    evaluate.add_argument(
        "--backend",
        choices=("transformers", "verl"),
        default="transformers",
    )
    evaluate.add_argument("--num-gpus", type=int, default=1)
    evaluate.add_argument("--policy-checkpoint")
    evaluate.add_argument(
        "--environment-backend",
        choices=("individual", "native_batch"),
        default="native_batch",
    )
    evaluate.add_argument(
        "--persistent-rollout-session",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    evaluate.add_argument("--verbose-runtime-logs", action="store_true")
    raw_skill_ab = subparsers.add_parser(
        "raw-skill-ab",
        help="run the diagnostic raw-skill retrieval/prompt-format matrix",
    )
    raw_skill_ab.add_argument("--config", required=True)
    raw_skill_ab.add_argument("--num-gpus", type=int, required=True)
    raw_skill_ab.add_argument("--tasks-per-type", type=int, default=2)
    raw_skill_ab.add_argument("--run-name")
    raw_skill_ab.add_argument(
        "--environment-backend",
        choices=("individual", "native_batch"),
        default="native_batch",
    )
    raw_skill_ab.add_argument(
        "--persistent-rollout-session",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    raw_skill_ab.add_argument("--verbose-runtime-logs", action="store_true")
    checkpoint_effect = subparsers.add_parser(
        "checkpoint-effect",
        help="diagnose portable checkpoint propagation into FSDP and vLLM",
    )
    checkpoint_effect.add_argument("--config", required=True)
    checkpoint_effect.add_argument("--policy-checkpoint", required=True)
    checkpoint_effect.add_argument("--num-gpus", type=int, required=True)
    checkpoint_effect.add_argument("--run-name")
    checkpoint_effect.add_argument("--max-new-tokens", type=int, default=64)
    checkpoint_effect.add_argument("--verbose-runtime-logs", action="store_true")
    grounding = subparsers.add_parser("grounding", help="generate strict train-only expert labels")
    grounding.add_argument("--config", required=True)
    grounding.add_argument("--run-name")
    train = subparsers.add_parser("train", help="run INFO-SKILL-owned GRPO training")
    train.add_argument("--config", required=True)
    train.add_argument("--mode", choices=[mode.value for mode in SkillMode], required=True)
    _add_retrieval_mode_argument(train)
    train.add_argument(
        "--profile",
        choices=[profile.value for profile in TrainingProfile],
        default=TrainingProfile.SMOKE.value,
    )
    train.add_argument("--max-updates", type=int)
    train.add_argument("--num-gpus", type=int, required=True)
    train.add_argument(
        "--environment-workers",
        type=int,
        default=1,
        help="number of already-loaded ALFWorld step/close calls allowed in parallel",
    )
    train.add_argument(
        "--environment-backend",
        choices=("individual", "native_batch"),
        default="native_batch",
        help=(
            "TextWorld native multiprocessing batch (default) or individual "
            "batch_size=1 environments for rollback/diagnosis"
        ),
    )
    train.add_argument("--run-name")
    train.add_argument("--resume")
    train.add_argument(
        "--persistent-rollout-session",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep vLLM awake across environment steps within one rollout update",
    )
    train.add_argument(
        "--verbose-runtime-logs",
        action="store_true",
        help="forward verbose Ray/vLLM/FSDP worker logs to the terminal",
    )
    train.add_argument(
        "--cuda-memory-poll-interval-ms",
        type=int,
        default=0,
        help="diagnostic physical CUDA memory polling interval; 0 disables polling",
    )
    train.add_argument(
        "--policy-max-tokens-per-gpu",
        type=int,
        default=16_384,
        help="old/ref/actor dynamic micro-batch token budget per GPU",
    )
    train.add_argument(
        "--balance-policy-tokens-across-ranks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="balance token load across FSDP ranks within each GRPO minibatch",
    )
    train.add_argument("--dry-run", action="store_true")
    return parser


def _add_retrieval_mode_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retrieval-mode",
        choices=("embedding", "template"),
        help="override the YAML retrieval mode for raw_skill_prompt/infoskill",
    )


def _train(config: AppConfig, args: argparse.Namespace) -> int:
    mode = SkillMode(args.mode)
    if mode is SkillMode.INFO_SKILL:
        raise NotImplementedError(
            "infoskill training remains fail-fast until the distributed M1 "
            "policy and auxiliary update paths are complete"
        )
    if args.num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if args.environment_workers <= 0:
        raise ValueError("environment_workers must be positive")
    if args.environment_backend == "native_batch" and args.environment_workers != 1:
        raise ValueError(
            "native_batch owns its process count; environment_workers must remain 1"
        )
    if args.cuda_memory_poll_interval_ms < 0:
        raise ValueError("cuda_memory_poll_interval_ms must be non-negative")
    minimum_token_budget = config.max_prompt_tokens + config.max_response_tokens
    if args.policy_max_tokens_per_gpu < minimum_token_budget:
        raise ValueError(
            "policy_max_tokens_per_gpu must be at least max_prompt_tokens + "
            f"max_response_tokens ({minimum_token_budget})"
        )
    _validate_paths(
        config,
        mode=mode,
        require_checkpoint=False,
        require_training_runtime=True,
    )
    plan = resolve_training_plan(args.profile, max_updates=args.max_updates)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "action": "train",
                    "mode": mode.value,
                    "retrieval_mode": config.retrieval_mode,
                    "profile": plan.profile.value,
                    "max_updates": plan.max_updates,
                    "task_groups_per_update": plan.task_groups_per_update,
                    "rollouts_per_task": plan.rollouts_per_task,
                    "trajectories_per_full_update": plan.trajectories_per_full_update,
                    "action_minibatch_size": plan.action_minibatch_size,
                    "evaluation_kind": plan.evaluation_kind,
                    "num_gpus": args.num_gpus,
                    "environment_workers": args.environment_workers,
                    "environment_backend": args.environment_backend,
                    "persistent_rollout_session": args.persistent_rollout_session,
                    "verbose_runtime_logs": args.verbose_runtime_logs,
                    "cuda_memory_poll_interval_ms": (
                        args.cuda_memory_poll_interval_ms
                    ),
                    "policy_max_tokens_per_gpu": args.policy_max_tokens_per_gpu,
                    "balance_policy_tokens_across_ranks": (
                        args.balance_policy_tokens_across_ranks
                    ),
                    "resume": args.resume,
                    "resume_forked": bool(args.resume and args.run_name),
                    "dry_run": True,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    from infoskill.training.m0 import run_policy_training

    return run_policy_training(
        config=config,
        mode=mode,
        plan=plan,
        num_gpus=args.num_gpus,
        run_name=args.run_name,
        resume=args.resume,
        persistent_rollout_session=args.persistent_rollout_session,
        environment_workers=args.environment_workers,
        environment_backend=args.environment_backend,
        verbose_runtime_logs=args.verbose_runtime_logs,
        cuda_memory_poll_interval_ms=args.cuda_memory_poll_interval_ms,
        policy_max_tokens_per_gpu=args.policy_max_tokens_per_gpu,
        balance_policy_tokens_across_ranks=(
            args.balance_policy_tokens_across_ranks
        ),
    )


def _evaluate(config: AppConfig, args: argparse.Namespace) -> int:
    evaluation_started = time.perf_counter()
    mode = SkillMode(args.mode)
    if args.num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if args.checkpoint_step < 0:
        raise ValueError("checkpoint_step must be non-negative")
    if args.backend == "transformers" and args.num_gpus != 1:
        raise ValueError("the Transformers evaluation backend requires num_gpus=1")
    if args.backend == "verl" and mode is SkillMode.INFO_SKILL:
        raise NotImplementedError(
            "VERL infoskill evaluation requires the unfinished distributed M1 path"
        )
    if args.backend == "verl" and config.paths.policy_adapter is not None:
        raise ValueError(
            "VERL checkpoint evaluation loads portable state explicitly; "
            "paths.policy_adapter must be null"
        )
    if args.backend != "verl" and args.policy_checkpoint is not None:
        raise ValueError("policy_checkpoint requires backend=verl")
    _validate_paths(
        config,
        mode=mode,
        require_checkpoint=mode is SkillMode.INFO_SKILL,
        require_training_runtime=args.backend == "verl",
    )
    from tqdm.auto import tqdm

    from infoskill.builders import (
        audit_raw_skill_prompt_budget_for_model,
        build_raw_skill_setup,
        build_transformers_evaluation,
        build_verl_policy_evaluation,
    )
    from infoskill.evaluation import (
        EvaluationRunner,
        write_checkpoint_load,
        write_evaluation_provenance,
        write_evaluation_timing,
    )
    from infoskill.integrations.alfworld import discover_tasks, task_manifest_sha256
    from infoskill.persistence import (
        MetricLogger,
        ZstdJsonlTraceWriter,
        resolve_portable_checkpoint,
    )
    from infoskill.persistence.model_identity import (
        provenance_matches_pinned_model,
        verify_policy_model_identity,
    )

    task_discovery_started = time.perf_counter()
    tasks = discover_tasks(config.paths.alfworld_data, split="valid_seen")
    task_discovery_seconds = time.perf_counter() - task_discovery_started
    evaluation_config = EvaluationConfig()
    if len(tasks) != evaluation_config.total_tasks:
        raise RuntimeError(
            f"valid_seen discovery returned {len(tasks)} tasks instead of "
            f"{evaluation_config.total_tasks}"
        )
    valid_seen_manifest_sha256 = task_manifest_sha256(tasks)
    if valid_seen_manifest_sha256 != evaluation_config.manifest_sha256:
        raise RuntimeError(
            "valid_seen task manifest SHA256 does not match the registered "
            f"manifest: {valid_seen_manifest_sha256}"
        )
    checkpoint = (
        resolve_portable_checkpoint(args.policy_checkpoint)
        if args.policy_checkpoint is not None
        else None
    )
    evaluation_step = args.checkpoint_step
    if checkpoint is not None:
        if evaluation_step not in {0, checkpoint.global_update}:
            raise ValueError(
                "checkpoint_step differs from the portable checkpoint global update"
            )
        evaluation_step = checkpoint.global_update
    elif args.backend == "verl" and evaluation_step != 0:
        raise ValueError(
            "nonzero VERL checkpoint_step requires a portable policy_checkpoint"
        )

    skill_setup_started = time.perf_counter()
    raw_skill_setup = (
        build_raw_skill_setup(
            config,
            retrieval_queries={task.task_id: task.goal for task in tasks},
        )
        if mode is SkillMode.RAW_SKILL_PROMPT
        else None
    )
    raw_skill_provenance = (
        {
            **raw_skill_setup.provenance,
            **audit_raw_skill_prompt_budget_for_model(
                raw_skill_setup,
                model_path=config.paths.policy_model,
                max_prompt_tokens=config.max_prompt_tokens,
            ),
        }
        if raw_skill_setup is not None
        else None
    )
    skill_setup_seconds = time.perf_counter() - skill_setup_started

    policy_identity = None
    checkpoint_provenance_sha256 = None
    if args.backend == "verl":
        policy_identity = verify_policy_model_identity(
            config.paths.policy_model,
            model_id=config.policy_model_id,
        )
        if checkpoint is not None:
            provenance_path = checkpoint.directory / "provenance.json"
            if not provenance_path.is_file():
                raise RuntimeError(
                    f"portable checkpoint has no policy provenance: {provenance_path}"
                )
            checkpoint_provenance = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            checkpoint_provenance_sha256 = hashlib.sha256(
                provenance_path.read_bytes()
            ).hexdigest()
            if not isinstance(checkpoint_provenance, dict):
                raise RuntimeError(
                    f"portable checkpoint provenance is not an object: {provenance_path}"
                )
            if checkpoint_provenance.get("mode") != mode.value:
                raise RuntimeError(
                    "portable checkpoint mode differs from evaluation mode"
                )
            if raw_skill_setup is not None:
                raw_skill_setup.require_checkpoint_compatibility(
                    checkpoint_provenance
                )
            if not provenance_matches_pinned_model(
                checkpoint_provenance,
                model_id=config.policy_model_id or "",
            ):
                raise RuntimeError(
                    "portable checkpoint policy provenance differs from the "
                    "registered model"
                )

    run_directory = _run_directory(
        config,
        args.run_name
        or f"eval-{mode.value}-{args.backend}-step-{evaluation_step:06d}",
    )
    logger = _configure_logging(run_directory)
    resolved = config.as_dict()
    evaluation_runtime = {
        "backend": args.backend,
        "num_gpus": args.num_gpus,
        "environment_backend": args.environment_backend,
        "persistent_rollout_session": args.persistent_rollout_session,
        "policy_checkpoint": (
            str(checkpoint.directory) if checkpoint is not None else None
        ),
        "checkpoint_step": evaluation_step,
    }
    evaluation_manifest = {
        "split": evaluation_config.split,
        "task_count": evaluation_config.total_tasks,
        "sha256": valid_seen_manifest_sha256,
    }
    resolved["evaluation_runtime"] = evaluation_runtime
    resolved["evaluation_manifest"] = evaluation_manifest
    if raw_skill_provenance is not None:
        resolved["skill_conditioning"] = raw_skill_provenance
    if policy_identity is not None:
        resolved["policy_model_identity"] = policy_identity.as_dict()
    _write_json(run_directory / "resolved_config.json", resolved)
    write_evaluation_provenance(
        run_directory,
        mode=mode.value,
        evaluation_runtime=evaluation_runtime,
        evaluation_manifest=evaluation_manifest,
        policy_model=(
            policy_identity.as_dict() if policy_identity is not None else None
        ),
        skill_conditioning=raw_skill_provenance,
        checkpoint_provenance_sha256=checkpoint_provenance_sha256,
    )
    checkpoint_path = (
        str(checkpoint.directory) if checkpoint is not None else None
    )
    write_checkpoint_load(
        run_directory,
        backend=args.backend,
        checkpoint=checkpoint_path,
        checkpoint_step=evaluation_step,
        status=("pending" if checkpoint is not None else "not_requested"),
    )

    timing_seconds = {
        "task_discovery_seconds": task_discovery_seconds,
        "skill_setup_seconds": skill_setup_seconds,
        "backend_initialize_seconds": 0.0,
        "checkpoint_load_seconds": 0.0,
        "collector_build_seconds": 0.0,
        "rollout_seconds": 0.0,
        "runtime_close_seconds": 0.0,
        "trace_write_seconds": 0.0,
    }

    runtime = None
    try:
        if args.backend == "verl":
            from infoskill.integrations.verl import VerlRuntime, VerlRuntimeConfig

            if VerlRuntime is None or VerlRuntimeConfig is None:
                raise RuntimeError("the pinned VERL runtime is unavailable")
            logger.info(
                "Initializing VERL/vLLM evaluation runtime on %d GPU(s); worker logs=%s",
                args.num_gpus,
                "verbose" if args.verbose_runtime_logs else "quiet",
            )
            stage_started = time.perf_counter()
            runtime = VerlRuntime.start(
                VerlRuntimeConfig(
                    skillrl_source=config.paths.skillrl_source,
                    model_path=config.paths.policy_model,
                    num_gpus=args.num_gpus,
                    num_cpus=96,
                    max_prompt_tokens=config.max_prompt_tokens,
                    max_response_tokens=config.max_response_tokens,
                    total_training_steps=max(1, evaluation_step),
                    action_minibatch_size=256,
                    policy_max_tokens_per_gpu=16_384,
                    gpu_memory_utilization=0.45,
                    require_hybrid_prefix=False,
                    master_seed=config.master_seed,
                    persistent_rollout_session=args.persistent_rollout_session,
                    verbose_runtime_logs=args.verbose_runtime_logs,
                    cuda_memory_poll_interval_ms=0,
                    balance_policy_tokens_across_ranks=True,
                )
            )
            timing_seconds["backend_initialize_seconds"] = (
                time.perf_counter() - stage_started
            )
            if checkpoint is not None:
                stage_started = time.perf_counter()
                try:
                    worker_reports = (
                        runtime.load_portable_state(checkpoint.runtime_directory) or ()
                    )
                except Exception as error:
                    timing_seconds["checkpoint_load_seconds"] = (
                        time.perf_counter() - stage_started
                    )
                    write_checkpoint_load(
                        run_directory,
                        backend=args.backend,
                        checkpoint=checkpoint_path,
                        checkpoint_step=evaluation_step,
                        status="failed",
                        duration_seconds=timing_seconds[
                            "checkpoint_load_seconds"
                        ],
                        error=error,
                    )
                    raise
                timing_seconds["checkpoint_load_seconds"] = (
                    time.perf_counter() - stage_started
                )
                write_checkpoint_load(
                    run_directory,
                    backend=args.backend,
                    checkpoint=checkpoint_path,
                    checkpoint_step=evaluation_step,
                    status="loaded",
                    duration_seconds=timing_seconds["checkpoint_load_seconds"],
                    worker_reports=worker_reports,
                )
                logger.info(
                    "Loaded portable policy checkpoint: %s",
                    checkpoint.directory,
                )
            stage_started = time.perf_counter()
            collector = build_verl_policy_evaluation(
                config,
                mode=mode,
                backend=runtime,
                conditioner=(
                    raw_skill_setup.conditioner
                    if raw_skill_setup is not None
                    else None
                ),
                environment_backend=args.environment_backend,
            )
            timing_seconds["collector_build_seconds"] = (
                time.perf_counter() - stage_started
            )
        else:
            logger.info(
                "Loading Transformers policy and environment for mode=%s",
                mode.value,
            )
            stage_started = time.perf_counter()
            collector = build_transformers_evaluation(
                config,
                mode=mode,
                environment_backend=args.environment_backend,
                conditioner=(
                    raw_skill_setup.conditioner
                    if raw_skill_setup is not None
                    else None
                ),
            )
            timing_seconds["backend_initialize_seconds"] = (
                time.perf_counter() - stage_started
            )

        progress = tqdm(
            total=len(tasks),
            desc=f"valid_seen/{mode.value}@{evaluation_step}",
            unit="task",
            dynamic_ncols=True,
        )
        runner = EvaluationRunner(
            collector_factory=lambda: collector,
            config=evaluation_config,
            task_batch_size=config.eval_batch_size,
            master_seed=config.master_seed,
            on_progress=progress.update,
        )
        try:
            stage_started = time.perf_counter()
            with collector.rollout_session():
                run = runner.run(tasks, checkpoint_step=evaluation_step)
            timing_seconds["rollout_seconds"] = (
                time.perf_counter() - stage_started
            )
        finally:
            progress.close()
    finally:
        if runtime is not None:
            stage_started = time.perf_counter()
            runtime.close()
            timing_seconds["runtime_close_seconds"] = (
                time.perf_counter() - stage_started
            )
    stage_started = time.perf_counter()
    trace_path = ZstdJsonlTraceWriter(run_directory).write_evaluation(
        checkpoint_step=evaluation_step, run=run
    )
    timing_seconds["trace_write_seconds"] = time.perf_counter() - stage_started
    timing_seconds["total_seconds"] = time.perf_counter() - evaluation_started
    write_evaluation_timing(run_directory, timing_seconds)
    summary = run.summary
    metrics = MetricLogger(run_directory)
    values: dict[str, float | int | str | bool | None] = {
        "complete": summary.is_complete,
        "evaluated": summary.evaluated,
        "overall_success": summary.overall_success,
        "macro_success": summary.macro_success,
        "invalid_action_rate": summary.invalid_action_rate,
        "mean_steps": summary.mean_steps,
        "incomplete_reasons": ";".join(summary.incomplete_reasons),
        "task_manifest_sha256": valid_seen_manifest_sha256,
    }
    values.update(
        {f"perf/{key}": value for key, value in timing_seconds.items()}
    )
    values.update({f"success/{key}": value for key, value in summary.per_task_type_success.items()})
    metrics.log(step=evaluation_step, phase="valid_seen", values=values)
    summary_payload = _summary_payload(run)
    summary_payload["task_manifest_sha256"] = valid_seen_manifest_sha256
    summary_payload["timing_seconds"] = timing_seconds
    _write_json(run_directory / "valid_seen_summary.json", summary_payload)
    logger.info("Structured trace: %s", trace_path)
    logger.info(
        "Evaluation timing: setup=%.1fs runtime=%.1fs checkpoint=%.1fs "
        "rollout=%.1fs close=%.1fs total=%.1fs",
        timing_seconds["task_discovery_seconds"]
        + timing_seconds["skill_setup_seconds"],
        timing_seconds["backend_initialize_seconds"],
        timing_seconds["checkpoint_load_seconds"],
        timing_seconds["rollout_seconds"],
        timing_seconds["runtime_close_seconds"],
        timing_seconds["total_seconds"],
    )
    logger.info(
        "Evaluation summary:\n%s",
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
    )
    return 0 if summary.is_complete else 3


def _raw_skill_ab(config: AppConfig, args: argparse.Namespace) -> int:
    """Run a small controlled prompt/retrieval diagnostic on valid_seen."""

    diagnostic_started = time.perf_counter()
    if args.num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if args.tasks_per_type <= 0:
        raise ValueError("tasks_per_type must be positive")
    if config.paths.policy_adapter is not None:
        raise ValueError(
            "raw-skill-ab diagnoses the registered base policy; "
            "paths.policy_adapter must be null"
        )
    _validate_paths(
        config,
        mode=SkillMode.RAW_SKILL_PROMPT,
        require_checkpoint=False,
        require_training_runtime=True,
    )

    from tqdm.auto import tqdm

    from infoskill.builders import (
        audit_raw_skill_prompt_budget_for_model,
        build_raw_skill_setup,
        build_verl_policy_evaluation,
    )
    from infoskill.diagnostics import (
        RAW_SKILL_AB_VARIANTS,
        select_stratified_tasks,
        summarize_probe_groups,
    )
    from infoskill.evaluation import write_checkpoint_load
    from infoskill.integrations.alfworld import discover_tasks, task_manifest_sha256
    from infoskill.integrations.verl import VerlRuntime, VerlRuntimeConfig
    from infoskill.persistence import MetricLogger, ZstdJsonlTraceWriter
    from infoskill.persistence.model_identity import verify_policy_model_identity

    if VerlRuntime is None or VerlRuntimeConfig is None:
        raise RuntimeError("the pinned VERL runtime is unavailable")
    all_tasks = discover_tasks(config.paths.alfworld_data, split="valid_seen")
    evaluation_config = EvaluationConfig()
    full_manifest_sha256 = task_manifest_sha256(all_tasks)
    if (
        len(all_tasks) != evaluation_config.total_tasks
        or full_manifest_sha256 != evaluation_config.manifest_sha256
    ):
        raise RuntimeError(
            "raw-skill-ab requires the registered 140-task valid_seen manifest"
        )
    tasks = select_stratified_tasks(
        all_tasks,
        tasks_per_type=args.tasks_per_type,
    )
    selected_manifest_sha256 = task_manifest_sha256(tasks)
    retrieval_queries = {task.task_id: task.goal for task in tasks}

    setups = {}
    prompt_budgets = {}
    for variant in RAW_SKILL_AB_VARIANTS:
        setup = build_raw_skill_setup(
            config,
            retrieval_queries=retrieval_queries,
            retrieval_mode=variant.retrieval_mode,
            prompt_format=variant.prompt_format,
        )
        setups[variant.name] = setup
        prompt_budgets[variant.name] = audit_raw_skill_prompt_budget_for_model(
            setup,
            model_path=config.paths.policy_model,
            max_prompt_tokens=config.max_prompt_tokens,
        )

    policy_identity = verify_policy_model_identity(
        config.paths.policy_model,
        model_id=config.policy_model_id,
    )
    run_directory = _run_directory(
        config,
        args.run_name or f"raw-skill-ab-valid-seen-{len(tasks)}",
    )
    logger = _configure_logging(run_directory)
    runtime_settings = VerlRuntimeConfig(
        skillrl_source=config.paths.skillrl_source,
        model_path=config.paths.policy_model,
        num_gpus=args.num_gpus,
        num_cpus=96,
        max_prompt_tokens=config.max_prompt_tokens,
        max_response_tokens=config.max_response_tokens,
        total_training_steps=1,
        action_minibatch_size=256,
        policy_max_tokens_per_gpu=16_384,
        gpu_memory_utilization=0.45,
        require_hybrid_prefix=False,
        master_seed=config.master_seed,
        persistent_rollout_session=args.persistent_rollout_session,
        verbose_runtime_logs=args.verbose_runtime_logs,
        cuda_memory_poll_interval_ms=0,
        balance_policy_tokens_across_ranks=True,
    )
    runtime_payload = {
        "backend": "verl",
        "num_gpus": args.num_gpus,
        "environment_backend": args.environment_backend,
        "persistent_rollout_session": args.persistent_rollout_session,
        "policy_checkpoint": None,
        "checkpoint_step": 0,
    }
    variants_payload = [
        {
            "name": variant.name,
            "retrieval_mode": variant.retrieval_mode,
            "prompt_format": variant.prompt_format,
        }
        for variant in RAW_SKILL_AB_VARIANTS
    ]
    diagnostic_manifest = {
        "diagnostic_only": True,
        "reportable_as_valid_seen": False,
        "split": "valid_seen",
        "full_task_count": len(all_tasks),
        "full_task_manifest_sha256": full_manifest_sha256,
        "selected_task_count": len(tasks),
        "selected_task_manifest_sha256": selected_manifest_sha256,
        "tasks_per_type": args.tasks_per_type,
        "selected_tasks": [
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "goal": task.goal,
            }
            for task in tasks
        ],
        "variants": variants_payload,
    }
    resolved = config.as_dict()
    resolved["evaluation_runtime"] = runtime_payload
    resolved["raw_skill_ab_diagnostic"] = diagnostic_manifest
    resolved["policy_model_identity"] = policy_identity.as_dict()
    resolved["skill_conditioning"] = {
        name: {
            **setup.provenance,
            **prompt_budgets[name],
        }
        for name, setup in setups.items()
    }
    _write_json(run_directory / "resolved_config.json", resolved)
    _write_json(
        run_directory / "provenance.json",
        {
            "schema_version": 1,
            "artifact_kind": "raw_skill_prompt_ab_diagnostic",
            **diagnostic_manifest,
            "evaluation_runtime": runtime_payload,
            "policy_model": policy_identity.as_dict(),
            "skill_conditioning": resolved["skill_conditioning"],
        },
    )
    write_checkpoint_load(
        run_directory,
        backend="verl",
        checkpoint=None,
        checkpoint_step=0,
        status="not_requested",
    )

    logger.info(
        "Initializing one VERL/vLLM runtime for %d diagnostic variants on %d GPU(s)",
        len(RAW_SKILL_AB_VARIANTS),
        args.num_gpus,
    )
    initialize_started = time.perf_counter()
    runtime = VerlRuntime.start(runtime_settings)
    initialize_seconds = time.perf_counter() - initialize_started
    trace_writer = ZstdJsonlTraceWriter(run_directory)
    metric_logger = MetricLogger(run_directory)
    progress = tqdm(
        total=len(tasks) * len(RAW_SKILL_AB_VARIANTS),
        desc="raw-skill-ab/diagnostic",
        unit="task",
        dynamic_ncols=True,
    )
    variant_results = []
    try:
        for variant in RAW_SKILL_AB_VARIANTS:
            setup = setups[variant.name]
            collector = build_verl_policy_evaluation(
                config,
                mode=SkillMode.RAW_SKILL_PROMPT,
                backend=runtime,
                conditioner=setup.conditioner,
                environment_backend=args.environment_backend,
            )
            rollout_started = time.perf_counter()
            groups = []
            with collector.rollout_session():
                for start in range(0, len(tasks), config.eval_batch_size):
                    batch = tuple(tasks[start : start + config.eval_batch_size])
                    groups.extend(
                        collector.collect_task_groups(
                            batch,
                            rollouts_per_task=1,
                            master_seed=config.master_seed,
                            global_update=0,
                        )
                    )
                    progress.update(len(batch))
            rollout_seconds = time.perf_counter() - rollout_started
            summary = summarize_probe_groups(groups)
            trace_path = trace_writer.write_diagnostic_groups(
                label=f"raw-skill-ab-{variant.name}",
                groups=groups,
            )
            result = {
                "variant": variant.name,
                "retrieval_mode": variant.retrieval_mode,
                "prompt_format": variant.prompt_format,
                "rollout_seconds": rollout_seconds,
                "trace": str(trace_path),
                **summary,
            }
            variant_results.append(result)
            metric_logger.log(
                step=0,
                phase=f"raw_skill_ab/{variant.name}",
                values={
                    key: value
                    for key, value in result.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                },
            )
            logger.info(
                "variant=%s success=%.4f invalid=%.4f repeat=%.4f rollout=%.1fs",
                variant.name,
                summary["overall_success"],
                summary["invalid_action_rate"],
                summary["consecutive_repeat_rate"],
                rollout_seconds,
            )
    finally:
        progress.close()
        close_started = time.perf_counter()
        runtime.close()
        close_seconds = time.perf_counter() - close_started

    payload = {
        "schema_version": 1,
        **diagnostic_manifest,
        "runtime_initialize_seconds": initialize_seconds,
        "runtime_close_seconds": close_seconds,
        "total_seconds": time.perf_counter() - diagnostic_started,
        "results": variant_results,
    }
    _write_json(run_directory / "raw_skill_ab_summary.json", payload)
    logger.info(
        "Raw-skill A/B diagnostic complete:\n%s",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    return 0


def _checkpoint_effect(config: AppConfig, args: argparse.Namespace) -> int:
    """Run one-runtime before/after probes across the portable-checkpoint seam."""

    if args.num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if args.max_new_tokens <= 0 or args.max_new_tokens > config.max_response_tokens:
        raise ValueError(
            "max_new_tokens must be positive and no greater than max_response_tokens"
        )
    if config.paths.policy_adapter is not None:
        raise ValueError(
            "checkpoint-effect loads portable state explicitly; "
            "paths.policy_adapter must be null"
        )
    _validate_paths(
        config,
        mode=SkillMode.NO_SKILL,
        require_checkpoint=False,
        require_training_runtime=True,
    )

    from infoskill.checkpoint_effect import (
        build_checkpoint_effect_probes,
        classify_checkpoint_effect,
        collect_independent_checkpoint_effect_samples,
        compare_generation_results,
        compare_vllm_to_checkpoint_aggregate,
    )
    from infoskill.integrations.verl import VerlRuntime, VerlRuntimeConfig
    from infoskill.persistence import resolve_portable_checkpoint
    from infoskill.persistence.model_identity import (
        provenance_matches_pinned_model,
        verify_policy_model_identity,
    )

    checkpoint = resolve_portable_checkpoint(args.policy_checkpoint)
    provenance_path = checkpoint.directory / "provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(
            f"portable checkpoint has no policy provenance: {provenance_path}"
        )
    checkpoint_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(checkpoint_provenance, dict):
        raise RuntimeError(
            f"portable checkpoint provenance is not an object: {provenance_path}"
        )
    if checkpoint_provenance.get("mode") != SkillMode.NO_SKILL.value:
        raise RuntimeError("portable checkpoint mode is not no_skill")
    policy_identity = verify_policy_model_identity(
        config.paths.policy_model,
        model_id=config.policy_model_id,
    )
    if not provenance_matches_pinned_model(
        checkpoint_provenance,
        model_id=config.policy_model_id or "",
    ):
        raise RuntimeError(
            "portable checkpoint policy provenance differs from the registered model"
        )
    if VerlRuntime is None or VerlRuntimeConfig is None:
        raise RuntimeError("the pinned VERL runtime is unavailable")

    run_directory = _run_directory(
        config,
        args.run_name
        or f"checkpoint-effect-step-{checkpoint.global_update:06d}",
    )
    logger = _configure_logging(run_directory)
    probes = build_checkpoint_effect_probes(
        master_seed=config.master_seed,
        max_new_tokens=args.max_new_tokens,
    )
    resolved = config.as_dict()
    resolved["checkpoint_effect"] = {
        "policy_checkpoint": str(checkpoint.directory),
        "checkpoint_step": checkpoint.global_update,
        "num_gpus": args.num_gpus,
        "probe_count": len(probes),
        "max_new_tokens": args.max_new_tokens,
    }
    resolved["policy_model_identity"] = policy_identity.as_dict()
    _write_json(run_directory / "resolved_config.json", resolved)

    runtime_settings = VerlRuntimeConfig(
        skillrl_source=config.paths.skillrl_source,
        model_path=config.paths.policy_model,
        num_gpus=args.num_gpus,
        num_cpus=96,
        max_prompt_tokens=config.max_prompt_tokens,
        max_response_tokens=config.max_response_tokens,
        total_training_steps=max(1, checkpoint.global_update),
        action_minibatch_size=256,
        policy_max_tokens_per_gpu=16_384,
        gpu_memory_utilization=0.45,
        require_hybrid_prefix=False,
        master_seed=config.master_seed,
        persistent_rollout_session=True,
        verbose_runtime_logs=args.verbose_runtime_logs,
        cuda_memory_poll_interval_ms=0,
        balance_policy_tokens_across_ranks=True,
    )
    logger.info(
        "Initializing independent checkpoint-first and baseline runtimes on %d GPU(s)",
        args.num_gpus,
    )
    samples = collect_independent_checkpoint_effect_samples(
        runtime_factory=lambda: VerlRuntime.start(runtime_settings),
        checkpoint_runtime_directory=checkpoint.runtime_directory,
        probes=probes,
    )

    generation_comparison = compare_generation_results(
        samples.baseline_results,
        samples.checkpoint_results,
    )
    vllm_checkpoint_comparison = compare_vllm_to_checkpoint_aggregate(
        samples.actor_comparison,
        samples.checkpoint_vllm,
        lora_scaling=runtime_settings.lora_alpha / runtime_settings.lora_rank,
    )
    verdict = classify_checkpoint_effect(
        actor_snapshots=samples.actor_comparison,
        vllm_snapshots=samples.checkpoint_vllm,
        vllm_checkpoint_comparison=vllm_checkpoint_comparison,
        generation_comparison=generation_comparison,
    )
    report = {
        "schema_version": 1,
        "checkpoint": {
            "directory": str(checkpoint.directory),
            "global_update": checkpoint.global_update,
        },
        "probe": {
            "request_ids": [request.request_id for request in probes],
            "max_new_tokens": args.max_new_tokens,
            "deterministic": True,
            "runtime_topology": "independent_checkpoint_first_vs_baseline",
        },
        "actor_checkpoint_comparison": list(samples.actor_comparison),
        "baseline_vllm_lora": list(samples.baseline_vllm),
        "checkpoint_vllm_lora": list(samples.checkpoint_vllm),
        "vllm_checkpoint_aggregate_comparison": vllm_checkpoint_comparison,
        "generation_comparison": generation_comparison,
        "verdict": verdict,
    }
    output = run_directory / "checkpoint_effect.json"
    _write_json(output, report)
    logger.info(
        "checkpoint-effect classification=%s actor_match=%s vllm_match=%s output_effect=%s",
        verdict["classification"],
        verdict["actor_matches_checkpoint_on_all_ranks"],
        verdict["vllm_matches_checkpoint_aggregate_on_all_ranks"],
        verdict["generation_effect_visible"],
    )
    logger.info("Diagnostic report: %s", output)
    return 0 if verdict["passed"] else 5


def _grounding(config: AppConfig, args: argparse.Namespace) -> int:
    _validate_paths(
        config,
        mode=SkillMode.INFO_SKILL,
        require_checkpoint=False,
    )
    from tqdm.auto import tqdm

    from infoskill.integrations.alfworld import (
        AlfworldEnvironmentFactory,
        StrictExpertReplay,
        build_grounding_manifest,
        discover_tasks,
        load_handcoded_expert,
        write_grounding_artifacts,
    )
    from infoskill.skills import (
        EmbeddingRetriever,
        FixedSkillLibrary,
        SentenceTransformerEncoder,
        TemplateRetriever,
    )

    run_directory = _run_directory(config, args.run_name or "grounding")
    logger = _configure_logging(run_directory)
    tasks = discover_tasks(config.paths.alfworld_data, split="train")
    library = FixedSkillLibrary.load(config.paths.skill_bank)
    if config.retrieval_mode == "embedding":
        retriever = EmbeddingRetriever(
            library,
            SentenceTransformerEncoder(config.paths.semantic_model, device="cuda:0"),
            general_top_k=config.general_top_k,
            task_top_k=config.task_top_k,
            mistake_count=config.mistake_count,
        )
    else:
        retriever = TemplateRetriever(
            library,
            general_count=config.general_top_k,
            task_count=config.task_top_k,
            mistake_count=config.mistake_count,
        )
    factory = AlfworldEnvironmentFactory.from_paths(
        alfworld_source=config.paths.alfworld_source,
        config_path=config.paths.alfworld_config,
        data_root=config.paths.alfworld_data,
        max_steps=150,
    )
    replay = StrictExpertReplay(max_replay_steps=150, persist_horizon=config.max_steps)
    expert = load_handcoded_expert(alfworld_source=config.paths.alfworld_source, max_steps=200)
    results = []
    for index, task in enumerate(tqdm(tasks, desc="train/expert-replay", unit="task", dynamic_ncols=True)):
        candidates = retriever.retrieve(task.goal).skill_ids
        environment = factory.create(task, rollout_id=0, seed=_stable_seed(config.master_seed, task.task_id))
        result = replay.run(
            task=task,
            environment=environment,
            expert=expert,  # type: ignore[arg-type]
            candidate_skill_ids=candidates,
        )
        results.append((task.task_type, result))
    manifest = build_grounding_manifest(
        results=results,
        source_checksums={
            "skill_bank": library.source_sha256,
            "train_task_manifest": _task_manifest_checksum(tasks),
            "infoskill_source": _source_checksum(),
        },
        code_revision=_source_checksum()[:16],
        max_replay_steps=150,
        persist_horizon=config.max_steps,
    )
    write_grounding_artifacts(output_directory=run_directory, results=results, manifest=manifest)
    logger.info("Grounding manifest:\n%s", json.dumps(manifest.__dict__ if hasattr(manifest, "__dict__") else _dataclass_dict(manifest), ensure_ascii=False, indent=2))
    return 0 if manifest.formal_gate_passed else 4


def _validate_paths(
    config: AppConfig,
    *,
    mode: SkillMode,
    require_checkpoint: bool,
    require_training_runtime: bool = False,
) -> None:
    required = {
        "policy_model": config.paths.policy_model,
        "alfworld_source": config.paths.alfworld_source,
        "alfworld_data": config.paths.alfworld_data,
        "alfworld_config": config.paths.alfworld_config,
        "output_root": config.paths.output_root,
    }
    if mode is not SkillMode.NO_SKILL:
        required["skill_bank"] = config.paths.skill_bank
        required["skill_bank_manifest"] = (
            config.paths.skill_bank_manifest
            or "configs/alfworld_skill_bank_manifest.json"
        )
    if mode is SkillMode.INFO_SKILL or (
        mode is SkillMode.RAW_SKILL_PROMPT
        and config.retrieval_mode == "embedding"
    ):
        required["semantic_model"] = config.paths.semantic_model
    if require_training_runtime:
        required["skillrl_source"] = config.paths.skillrl_source
    if config.paths.policy_adapter:
        required["policy_adapter"] = config.paths.policy_adapter
    if require_checkpoint:
        if not config.paths.infoskill_checkpoint:
            raise ValueError("infoskill evaluation requires paths.infoskill_checkpoint")
        required["infoskill_checkpoint"] = config.paths.infoskill_checkpoint
    missing = [f"{name}={path}" for name, path in required.items() if not Path(path).expanduser().exists()]
    if missing:
        raise FileNotFoundError("configured local paths do not exist: " + ", ".join(missing))


def _run_directory(config: AppConfig, name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(config.paths.output_root).expanduser() / f"{stamp}-{name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _configure_logging(run_directory: Path) -> logging.Logger:
    logger = logging.getLogger("infoskill")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_directory / "console.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _summary_payload(run: object) -> dict[str, object]:
    summary = run.summary  # type: ignore[attr-defined]
    return {
        "is_complete": summary.is_complete,
        "evaluated": summary.evaluated,
        "overall_success": summary.overall_success,
        "macro_success": summary.macro_success,
        "invalid_action_rate": summary.invalid_action_rate,
        "mean_steps": summary.mean_steps,
        "per_task_type_success": summary.per_task_type_success,
        "incomplete_reasons": list(summary.incomplete_reasons),
    }


def _stable_seed(master_seed: int, task_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"expert|{master_seed}|{task_id}".encode()).digest()[:8], "big") % (2**31)


def _task_manifest_checksum(tasks: object) -> str:
    digest = hashlib.sha256()
    for task in tasks:  # type: ignore[union-attr]
        digest.update(f"{task.task_id}\0{task.task_type}\0{task.goal}\n".encode("utf-8"))
    return digest.hexdigest()


def _source_checksum() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dataclass_dict(value: object) -> dict[str, object]:
    from dataclasses import asdict

    return asdict(value)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
