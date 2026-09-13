from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from infoskill.app_config import AppConfig
from infoskill.config import (
    DEFAULT_POLICY_MAX_TOKENS_PER_GPU,
    EvaluationConfig,
    SkillMode,
)
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
            require_checkpoint=False,
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
    if args.command == "grounding-timeout-rescue":
        return _grounding_timeout_rescue(config, args)
    if args.command == "grounding-expert-diagnostic":
        return _grounding_expert_diagnostic(config, args)
    if args.command == "grounding-planner-pilot":
        return _grounding_planner_pilot(config, args)
    if args.command == "grounding-planner-parity":
        return _grounding_planner_parity(config, args)
    if args.command == "grounding-planner-loop-diagnostic":
        return _grounding_planner_loop_diagnostic(config, args)
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
    _add_raw_skill_prompt_format_argument(evaluate)
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
    evaluate.add_argument(
        "--grouped-infoskill-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "experimental M1 evaluation optimization; batch conditioning "
            "across active tasks after exact parity validation"
        ),
    )
    evaluate.add_argument(
        "--diagnostic-task-manifest",
        help=(
            "fixed non-reportable valid_seen subset used only for controlled "
            "performance and parity diagnostics"
        ),
    )
    evaluate.add_argument(
        "--eval-batch-size",
        type=int,
        help="explicit evaluation batch size; omission preserves the YAML default",
    )
    evaluate.add_argument(
        "--cuda-memory-poll-interval-ms",
        type=int,
        default=0,
        help="diagnostic physical CUDA memory polling interval; 0 disables polling",
    )
    evaluate.add_argument("--verbose-runtime-logs", action="store_true")
    raw_skill_ab = subparsers.add_parser(
        "raw-skill-ab",
        help="run the diagnostic raw-skill retrieval/prompt-format matrix",
    )
    raw_skill_ab.add_argument("--config", required=True)
    raw_skill_ab.add_argument("--num-gpus", type=int, required=True)
    raw_skill_ab.add_argument("--tasks-per-type", type=int, default=2)
    raw_skill_ab.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "embedding-skillrl",
            "template-full",
            "template-skillrl",
            "skillrl-rl-exact",
            "skillrl-sft-exact",
            "skillrl-sft-shell-no-skills-deterministic",
            "no-skill-sampled-t0.4",
            "skillrl-sft-exact-sampled-t0.4",
            "unified-no-skill-deterministic",
            "unified-empty-skills-deterministic",
            "unified-template-skills-deterministic",
            "unified-embedding-skills-deterministic",
        ),
        help=(
            "run only the selected diagnostic variants; omission preserves "
            "the original three-cell raw-skill matrix"
        ),
    )
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
    grounding.add_argument(
        "--resume-run",
        help=(
            "existing grounding run directory whose committed shards should "
            "be validated and reused"
        ),
    )
    grounding.add_argument(
        "--worker-batch-size",
        type=int,
        default=64,
        help=(
            "number of expert replays per short-lived worker process; "
            "bounds TextWorld/Fast Downward temporary resources"
        ),
    )
    grounding.add_argument("--worker-processes", type=int, default=1)
    grounding.add_argument(
        "--replay-backend",
        choices=("individual", "native_batch"),
        default="individual",
    )
    grounding.add_argument("--native-batch-size", type=int, default=4)
    grounding.add_argument(
        "--worker-inactivity-timeout-seconds",
        type=float,
        default=300.0,
        help=(
            "terminate a worker after this many seconds without a completed "
            "task, then isolate unfinished tasks"
        ),
    )
    grounding_rescue = subparsers.add_parser(
        "grounding-timeout-rescue",
        help=(
            "retry only expert_wall_timeout rows from a committed formal "
            "grounding run and derive a checksum-audited merged dataset"
        ),
    )
    grounding_rescue.add_argument("--config", required=True)
    grounding_rescue.add_argument("--source-grounding-run", required=True)
    grounding_rescue.add_argument("--run-name")
    grounding_rescue.add_argument(
        "--resume-run",
        help="existing timeout-rescue run whose committed shards should be reused",
    )
    grounding_rescue.add_argument(
        "--finalize-committed-rescue-run",
        help=(
            "read a checksum-validated snapshot of committed rescue shards and "
            "write a separate derived formal run without waiting for every retry"
        ),
    )
    grounding_rescue.add_argument("--worker-processes", type=int, default=4)
    grounding_rescue.add_argument(
        "--worker-inactivity-timeout-seconds",
        type=float,
        default=600.0,
    )
    grounding_diagnostic = subparsers.add_parser(
        "grounding-expert-diagnostic",
        help="compare strict handcoded replay with the ALFWorld planner",
    )
    grounding_diagnostic.add_argument("--config", required=True)
    grounding_diagnostic.add_argument("--source-grounding-run", required=True)
    grounding_diagnostic.add_argument("--tasks-per-type", type=int, default=3)
    grounding_diagnostic.add_argument("--max-replay-steps", type=int, default=150)
    grounding_diagnostic.add_argument("--run-name")
    planner_pilot = subparsers.add_parser(
        "grounding-planner-pilot",
        help="run a balanced, non-formal ALFWorld planner grounding pilot",
    )
    planner_pilot.add_argument("--config", required=True)
    planner_pilot.add_argument("--tasks-per-type", type=int, default=50)
    planner_pilot.add_argument("--worker-batch-size", type=int, default=64)
    planner_pilot.add_argument("--worker-processes", type=int, default=1)
    planner_pilot.add_argument(
        "--replay-backend",
        choices=("individual", "native_batch"),
        default="individual",
    )
    planner_pilot.add_argument("--native-batch-size", type=int, default=4)
    planner_pilot.add_argument("--max-replay-steps", type=int, default=150)
    planner_pilot.add_argument("--run-name")
    planner_parity = subparsers.add_parser(
        "grounding-planner-parity",
        help="compare serial and parallel planner replay on identical tasks",
    )
    planner_parity.add_argument("--config", required=True)
    planner_parity.add_argument("--tasks-per-type", type=int, default=2)
    planner_parity.add_argument("--worker-batch-size", type=int, default=6)
    planner_parity.add_argument("--parallel-workers", type=int, default=2)
    planner_parity.add_argument(
        "--candidate-backend",
        choices=(
            "process_parallel",
            "native_batch",
            "native_batch_parallel",
        ),
        default="process_parallel",
    )
    planner_parity.add_argument("--native-batch-size", type=int, default=4)
    planner_parity.add_argument("--minimum-speedup", type=float, default=0.0)
    planner_parity.add_argument("--max-replay-steps", type=int, default=150)
    planner_parity.add_argument("--run-name")
    planner_loop = subparsers.add_parser(
        "grounding-planner-loop-diagnostic",
        help="trace pilot failures and long successful controls at a larger horizon",
    )
    planner_loop.add_argument("--config", required=True)
    planner_loop.add_argument("--source-pilot-run", required=True)
    planner_loop.add_argument("--successful-two-object-controls", type=int, default=6)
    planner_loop.add_argument("--max-replay-steps", type=int, default=300)
    planner_loop.add_argument("--run-name")
    train = subparsers.add_parser("train", help="run INFO-SKILL-owned GRPO training")
    train.add_argument("--config", required=True)
    train.add_argument("--mode", choices=[mode.value for mode in SkillMode], required=True)
    _add_retrieval_mode_argument(train)
    _add_raw_skill_prompt_format_argument(train)
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
    train.add_argument(
        "--eval-batch-size",
        type=int,
        help=(
            "explicit batch size for periodic valid_seen monitoring; omission "
            "preserves the YAML default"
        ),
    )
    train.add_argument("--run-name")
    train.add_argument("--resume")
    train.add_argument(
        "--segment-end-update",
        type=int,
        help=(
            "pause after committing this global update; this bounds one "
            "invocation without changing the registered training target"
        ),
    )
    train.add_argument(
        "--grounding-data",
        help=(
            "override paths.grounding_data for infoskill training without "
            "editing the shared YAML"
        ),
    )
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
        default=DEFAULT_POLICY_MAX_TOKENS_PER_GPU,
        help="old/ref/actor dynamic micro-batch token budget per GPU",
    )
    train.add_argument(
        "--balance-policy-tokens-across-ranks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="balance token load across FSDP ranks within each GRPO minibatch",
    )
    train.add_argument(
        "--skip-unused-old-logprob-entropy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "experimental M1-only optimization: do not compute entropy in "
            "the old-logprob pass because PPO does not consume it"
        ),
    )
    train.add_argument(
        "--rollout-max-batched-tokens",
        type=int,
        default=16_384,
        help="experimental vLLM scheduler token capacity per generation batch",
    )
    train.add_argument(
        "--hybrid-prefix-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "experimental M1-only candidate: use persistent hybrid-prefix "
            "embedding buffers with vLLM CUDA Graph"
        ),
    )
    train.add_argument(
        "--fuse-kl-ppo-forward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "experimental M1-only algorithm candidate: reuse the PPO actor "
            "forward for KL; KL then also regularizes the projector"
        ),
    )
    train.add_argument("--dry-run", action="store_true")
    return parser


def _add_retrieval_mode_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retrieval-mode",
        choices=("embedding", "template"),
        help="override the YAML retrieval mode for raw_skill_prompt/infoskill",
    )


def _add_raw_skill_prompt_format_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--raw-skill-prompt-format",
        choices=("compact", "full"),
        default="full",
        help=(
            "model-visible serialization for raw_skill_prompt; full is the "
            "registered default and compact is an explicit measured ablation"
        ),
    )


def _train(config: AppConfig, args: argparse.Namespace) -> int:
    mode = SkillMode(args.mode)
    if args.eval_batch_size is not None:
        if args.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        config = replace(config, eval_batch_size=args.eval_batch_size)
    if args.grounding_data is not None:
        config = replace(
            config,
            paths=replace(config.paths, grounding_data=args.grounding_data),
        )
    if mode is SkillMode.INFO_SKILL and not config.paths.grounding_data:
        raise ValueError(
            "infoskill training requires paths.grounding_data to point at a "
            "completed train-only grounding run"
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
    if args.rollout_max_batched_tokens < minimum_token_budget:
        raise ValueError(
            "rollout_max_batched_tokens must be at least max_prompt_tokens + "
            f"max_response_tokens ({minimum_token_budget})"
        )
    if args.skip_unused_old_logprob_entropy and mode is not SkillMode.INFO_SKILL:
        raise ValueError(
            "skip_unused_old_logprob_entropy is registered only for infoskill"
        )
    if args.hybrid_prefix_cuda_graph and mode is not SkillMode.INFO_SKILL:
        raise ValueError(
            "hybrid_prefix_cuda_graph is registered only for infoskill"
        )
    if args.fuse_kl_ppo_forward and mode is not SkillMode.INFO_SKILL:
        raise ValueError(
            "fuse_kl_ppo_forward is registered only for infoskill"
        )
    if (
        args.rollout_max_batched_tokens != 16_384
        and mode is not SkillMode.INFO_SKILL
    ):
        raise ValueError(
            "rollout_max_batched_tokens overrides are registered only for infoskill"
        )
    _validate_paths(
        config,
        mode=mode,
        require_checkpoint=False,
        require_training_runtime=True,
        require_grounding=mode is SkillMode.INFO_SKILL,
    )
    plan = resolve_training_plan(args.profile, max_updates=args.max_updates)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "action": "train",
                    "mode": mode.value,
                    "retrieval_mode": config.retrieval_mode,
                    "raw_skill_prompt_format": (
                        args.raw_skill_prompt_format
                        if mode is SkillMode.RAW_SKILL_PROMPT
                        else None
                    ),
                    "infoskill_auxiliary_enabled": mode is SkillMode.INFO_SKILL,
                    "infoskill_latent_mode": (
                        "sample" if mode is SkillMode.INFO_SKILL else None
                    ),
                    "infoskill_soft_prefix_length": (
                        5 if mode is SkillMode.INFO_SKILL else None
                    ),
                    "grounding_data": (
                        config.paths.grounding_data
                        if mode is SkillMode.INFO_SKILL
                        else None
                    ),
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
                    "eval_batch_size": config.eval_batch_size,
                    "persistent_rollout_session": args.persistent_rollout_session,
                    "verbose_runtime_logs": args.verbose_runtime_logs,
                    "cuda_memory_poll_interval_ms": (
                        args.cuda_memory_poll_interval_ms
                    ),
                    "policy_max_tokens_per_gpu": args.policy_max_tokens_per_gpu,
                    "balance_policy_tokens_across_ranks": (
                        args.balance_policy_tokens_across_ranks
                    ),
                    "skip_unused_old_logprob_entropy": (
                        args.skip_unused_old_logprob_entropy
                    ),
                    "rollout_max_batched_tokens": (
                        args.rollout_max_batched_tokens
                    ),
                    "hybrid_prefix_cuda_graph": (
                        args.hybrid_prefix_cuda_graph
                    ),
                    "fuse_kl_ppo_forward": args.fuse_kl_ppo_forward,
                    "resume": args.resume,
                    "resume_forked": bool(args.resume and args.run_name),
                    "segment_end_update": args.segment_end_update,
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
        segment_end_update=args.segment_end_update,
        persistent_rollout_session=args.persistent_rollout_session,
        environment_workers=args.environment_workers,
        environment_backend=args.environment_backend,
        verbose_runtime_logs=args.verbose_runtime_logs,
        cuda_memory_poll_interval_ms=args.cuda_memory_poll_interval_ms,
        policy_max_tokens_per_gpu=args.policy_max_tokens_per_gpu,
        balance_policy_tokens_across_ranks=(
            args.balance_policy_tokens_across_ranks
        ),
        skip_unused_old_logprob_entropy=(
            args.skip_unused_old_logprob_entropy
        ),
        rollout_max_batched_tokens=args.rollout_max_batched_tokens,
        hybrid_prefix_cuda_graph=args.hybrid_prefix_cuda_graph,
        fuse_kl_ppo_forward=args.fuse_kl_ppo_forward,
        raw_skill_prompt_format=args.raw_skill_prompt_format,
    )


def _evaluate(config: AppConfig, args: argparse.Namespace) -> int:
    evaluation_started = time.perf_counter()
    mode = SkillMode(args.mode)
    if args.num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if args.checkpoint_step < 0:
        raise ValueError("checkpoint_step must be non-negative")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    if args.cuda_memory_poll_interval_ms < 0:
        raise ValueError("cuda_memory_poll_interval_ms must be non-negative")
    if args.diagnostic_task_manifest is not None and mode is not SkillMode.INFO_SKILL:
        raise ValueError("the fixed pressure diagnostic is registered only for infoskill")
    if args.backend == "transformers" and args.num_gpus != 1:
        raise ValueError("the Transformers evaluation backend requires num_gpus=1")
    if mode is SkillMode.INFO_SKILL and args.backend != "verl":
        raise ValueError(
            "infoskill evaluation requires backend=verl so the portable M1 "
            "modules and Hybrid Prefix Input are loaded"
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
        require_checkpoint=False,
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
    full_valid_seen_manifest_sha256 = task_manifest_sha256(tasks)
    if full_valid_seen_manifest_sha256 != evaluation_config.manifest_sha256:
        raise RuntimeError(
            "valid_seen task manifest SHA256 does not match the registered "
            f"manifest: {full_valid_seen_manifest_sha256}"
        )
    diagnostic_manifest = None
    if args.diagnostic_task_manifest is not None:
        from infoskill.diagnostics import (
            diagnostic_denominators,
            load_pressure_task_manifest,
        )

        tasks, diagnostic_manifest = load_pressure_task_manifest(
            args.diagnostic_task_manifest,
            available_tasks=tasks,
            full_task_manifest_sha256=full_valid_seen_manifest_sha256,
        )
        selected_manifest_sha256 = task_manifest_sha256(tasks)
        evaluation_config = EvaluationConfig(
            denominators=diagnostic_denominators(tasks),
            manifest_sha256=selected_manifest_sha256,
        )
    valid_seen_manifest_sha256 = task_manifest_sha256(tasks)
    effective_eval_batch_size = args.eval_batch_size or config.eval_batch_size
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
    skill_setup = (
        build_raw_skill_setup(
            config,
            retrieval_queries={task.task_id: task.goal for task in tasks},
            prompt_format=args.raw_skill_prompt_format,
        )
        if mode in {SkillMode.RAW_SKILL_PROMPT, SkillMode.INFO_SKILL}
        else None
    )
    if skill_setup is None:
        skill_provenance = None
    elif mode is SkillMode.RAW_SKILL_PROMPT:
        skill_provenance = {
            **skill_setup.provenance,
            **audit_raw_skill_prompt_budget_for_model(
                skill_setup,
                model_path=config.paths.policy_model,
                max_prompt_tokens=config.max_prompt_tokens,
            ),
        }
    else:
        skill_provenance = {
            **skill_setup.provenance,
            "prompt_format": "continuous_soft_prefix_v1",
            "soft_prefix_length": 5,
            "latent_dim": 32,
            "latent_train_mode": "sample",
            "latent_eval_mode": "mean",
            "dynamic_skill_updates": False,
        }
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
            if skill_setup is not None:
                skill_setup.require_checkpoint_compatibility(
                    checkpoint_provenance,
                    expected_prompt_format=(
                        "continuous_soft_prefix_v1"
                        if mode is SkillMode.INFO_SKILL
                        else None
                    ),
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
        "grouped_infoskill_conditioning": (
            args.grouped_infoskill_conditioning
            if mode is SkillMode.INFO_SKILL
            else False
        ),
        "eval_batch_size": effective_eval_batch_size,
        "cuda_memory_poll_interval_ms": args.cuda_memory_poll_interval_ms,
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
    if diagnostic_manifest is not None:
        evaluation_manifest.update(
            {
                "diagnostic_only": True,
                "reportable_as_valid_seen": False,
                "pressure_manifest_sha256": diagnostic_manifest["sha256"],
                "full_task_count": 140,
                "full_task_manifest_sha256": full_valid_seen_manifest_sha256,
            }
        )
    resolved["evaluation_runtime"] = evaluation_runtime
    resolved["evaluation_manifest"] = evaluation_manifest
    if skill_provenance is not None:
        resolved["skill_conditioning"] = skill_provenance
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
        skill_conditioning=skill_provenance,
        checkpoint_provenance_sha256=checkpoint_provenance_sha256,
        artifact_kind=(
            "infoskill_eval_batch_pressure_diagnostic"
            if diagnostic_manifest is not None
            else "valid_seen_evaluation"
        ),
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
                    policy_max_tokens_per_gpu=DEFAULT_POLICY_MAX_TOKENS_PER_GPU,
                    gpu_memory_utilization=0.45,
                    require_hybrid_prefix=mode is SkillMode.INFO_SKILL,
                    soft_prefix_length=5,
                    master_seed=config.master_seed,
                    persistent_rollout_session=args.persistent_rollout_session,
                    verbose_runtime_logs=args.verbose_runtime_logs,
                    cuda_memory_poll_interval_ms=args.cuda_memory_poll_interval_ms,
                    balance_policy_tokens_across_ranks=True,
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
                    enable_infoskill_auxiliary=False,
                    infoskill_history_length=config.history_length,
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
            conditioner = (
                skill_setup.conditioner if skill_setup is not None else None
            )
            if mode is SkillMode.INFO_SKILL:
                from infoskill.conditioning import RuntimeInfoSkillConditioner

                if skill_setup is None or skill_setup.retriever is None:
                    raise RuntimeError("infoskill retrieval setup is incomplete")
                conditioner = RuntimeInfoSkillConditioner(
                    retriever=skill_setup.retriever,
                    runtime=runtime,
                    latent_mode="mean",
                    query_by_task_id={
                        task.task_id: task.goal for task in tasks
                    },
                    group_across_tasks=args.grouped_infoskill_conditioning,
                )
            stage_started = time.perf_counter()
            collector = build_verl_policy_evaluation(
                config,
                mode=mode,
                backend=runtime,
                conditioner=conditioner,
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
                    skill_setup.conditioner
                    if skill_setup is not None
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
            task_batch_size=effective_eval_batch_size,
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
            if runtime is not None and args.cuda_memory_poll_interval_ms > 0:
                performance = dict(run.performance_metrics or {})
                performance.update(runtime.rollout_memory_metrics())
                run = replace(run, performance_metrics=performance)
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
        checkpoint_step=evaluation_step,
        run=run,
        split=(
            "diagnostic_valid_seen"
            if diagnostic_manifest is not None
            else "valid_seen"
        ),
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
    values.update(run.performance_metrics or {})
    values.update({f"success/{key}": value for key, value in summary.per_task_type_success.items()})
    metrics.log(
        step=evaluation_step,
        phase=(
            "diagnostic_valid_seen"
            if diagnostic_manifest is not None
            else "valid_seen"
        ),
        values=values,
    )
    summary_payload = _summary_payload(run)
    summary_payload["task_manifest_sha256"] = valid_seen_manifest_sha256
    summary_payload["timing_seconds"] = timing_seconds
    summary_payload["rollout_performance"] = dict(run.performance_metrics or {})
    summary_name = (
        "diagnostic_summary.json"
        if diagnostic_manifest is not None
        else "valid_seen_summary.json"
    )
    if diagnostic_manifest is not None:
        summary_payload.update(
            {
                "diagnostic_only": True,
                "reportable_as_valid_seen": False,
                "pressure_manifest_sha256": diagnostic_manifest["sha256"],
                "eval_batch_size": effective_eval_batch_size,
            }
        )
    _write_json(run_directory / summary_name, summary_payload)
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
    rollout_performance = run.performance_metrics or {}
    logger.info(
        "Evaluation rollout breakdown: conditioning=%.1fs generation=%.1fs "
        "environment=%.1fs action-resolution=%.1fs",
        rollout_performance.get("perf/rollout_conditioning_seconds", 0.0),
        rollout_performance.get("perf/rollout_backend_generate_seconds", 0.0),
        sum(
            rollout_performance.get(key, 0.0)
            for key in (
                "perf/environment_create_seconds",
                "perf/environment_reset_seconds",
                "perf/environment_step_seconds",
                "perf/environment_close_seconds",
            )
        ),
        rollout_performance.get("perf/rollout_action_resolution_seconds", 0.0),
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
        build_skillrl_grpo_prompt_setup,
        build_skillrl_sft_no_skills_prompt_setup,
        build_skillrl_sft_prompt_setup,
        build_verl_policy_evaluation,
    )
    from infoskill.diagnostics import (
        compare_unified_prompt_controls,
        resolve_raw_skill_diagnostic_variants,
        select_stratified_tasks,
        summarize_probe_groups,
        validate_unified_skill_causal_gate,
    )
    from infoskill.conditioning import RawSkillPromptConditioner
    from infoskill.evaluation import write_checkpoint_load
    from infoskill.integrations.alfworld import discover_tasks, task_manifest_sha256
    from infoskill.integrations.verl import VerlRuntime, VerlRuntimeConfig
    from infoskill.persistence import MetricLogger, ZstdJsonlTraceWriter
    from infoskill.persistence.model_identity import verify_policy_model_identity
    from infoskill.rollout import GenerationParameters
    from infoskill.skills import EmptyRetriever

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
    variants = resolve_raw_skill_diagnostic_variants(args.variants)
    unified_skill_causal = validate_unified_skill_causal_gate(
        variants,
        tasks_per_type=args.tasks_per_type,
        policy_model_id=config.policy_model_id,
        master_seed=config.master_seed,
        history_length=config.history_length,
        max_steps=config.max_steps,
    )
    tasks = select_stratified_tasks(
        all_tasks,
        tasks_per_type=args.tasks_per_type,
    )
    selected_manifest_sha256 = task_manifest_sha256(tasks)
    retrieval_queries = {task.task_id: task.goal for task in tasks}

    conditioners = {}
    prompt_budgets = {}
    conditioning_provenance = {}
    for variant in variants:
        if variant.policy_mode == "no_skill":
            conditioner = None
            conditioning_provenance[variant.name] = {
                "retrieval_mode": None,
                "prompt_format": "unified_no_skill",
                "skills_injected": False,
                "history_length": config.history_length,
                "policy_prompt_schema_version": 1,
                "diagnostic_only": True,
            }
            prompt_budgets[variant.name] = {
                "raw_skill_block_count": 0,
                "raw_skill_block_tokens_min": 0,
                "raw_skill_block_tokens_max": 0,
                "raw_skill_prompt_token_limit": config.max_prompt_tokens,
                "raw_skill_block_fits_prompt_limit": True,
            }
        elif variant.prompt_format == "unified_empty_skills":
            conditioner = RawSkillPromptConditioner(
                EmptyRetriever(),
                history_length=config.history_length,
                query_by_task_id=retrieval_queries,
                prompt_format="full",
            )
            conditioning_provenance[variant.name] = {
                "retrieval_schema_version": 1,
                "retrieval_mode": "empty",
                "prompt_format": "unified_empty_skills",
                "skills_injected": False,
                "retrieval_query_count": len(set(retrieval_queries.values())),
                "history_length": config.history_length,
                "policy_prompt_schema_version": 1,
                "diagnostic_only": True,
            }
            prompt_budgets[variant.name] = {
                "raw_skill_block_count": 0,
                "raw_skill_block_tokens_min": 0,
                "raw_skill_block_tokens_max": 0,
                "raw_skill_prompt_token_limit": config.max_prompt_tokens,
                "raw_skill_block_fits_prompt_limit": True,
            }
        else:
            if variant.prompt_format == "skillrl_rl_exact":
                setup = build_skillrl_grpo_prompt_setup(config)
            elif variant.prompt_format == "skillrl_sft_exact":
                setup = build_skillrl_sft_prompt_setup(config)
            elif variant.prompt_format == "skillrl_sft_no_skills":
                setup = build_skillrl_sft_no_skills_prompt_setup(config)
            else:
                if variant.retrieval_mode is None:
                    raise ValueError(
                        f"diagnostic variant {variant.name} requires retrieval"
                    )
                setup = build_raw_skill_setup(
                    config,
                    retrieval_queries=retrieval_queries,
                    retrieval_mode=variant.retrieval_mode,
                    prompt_format=variant.prompt_format,
                )
            conditioner = setup.conditioner
            conditioning_provenance[variant.name] = dict(setup.provenance)
            prompt_budgets[variant.name] = audit_raw_skill_prompt_budget_for_model(
                setup,
                model_path=config.paths.policy_model,
                max_prompt_tokens=config.max_prompt_tokens,
            )
        conditioners[variant.name] = conditioner

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
        policy_max_tokens_per_gpu=DEFAULT_POLICY_MAX_TOKENS_PER_GPU,
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
            "policy_mode": variant.policy_mode,
            "generation": {
                "do_sample": variant.do_sample,
                "temperature": variant.temperature,
                "top_p": variant.top_p,
                "max_new_tokens": config.max_response_tokens,
                "master_seed": config.master_seed,
            },
        }
        for variant in variants
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
            **conditioning_provenance[name],
            **prompt_budgets[name],
        }
        for name in conditioners
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
        len(variants),
        args.num_gpus,
    )
    initialize_started = time.perf_counter()
    runtime = VerlRuntime.start(runtime_settings)
    initialize_seconds = time.perf_counter() - initialize_started
    trace_writer = ZstdJsonlTraceWriter(run_directory)
    metric_logger = MetricLogger(run_directory)
    progress = tqdm(
        total=len(tasks) * len(variants),
        desc="raw-skill-ab/diagnostic",
        unit="task",
        dynamic_ncols=True,
    )
    variant_results = []
    groups_by_variant = {}
    try:
        for variant in variants:
            conditioner = conditioners[variant.name]
            mode = SkillMode(variant.policy_mode)
            collector = build_verl_policy_evaluation(
                config,
                mode=mode,
                backend=runtime,
                conditioner=conditioner,
                environment_backend=args.environment_backend,
                history_limit=(
                    5
                    if variant.prompt_format
                    in {"skillrl_sft_exact", "skillrl_sft_no_skills"}
                    else None
                ),
                generation_parameters=GenerationParameters(
                    do_sample=variant.do_sample,
                    temperature=variant.temperature,
                    top_p=variant.top_p,
                    max_new_tokens=config.max_response_tokens,
                ),
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
            groups_by_variant[variant.name] = tuple(groups)
            trace_path = trace_writer.write_diagnostic_groups(
                label=f"raw-skill-ab-{variant.trace_slug}",
                groups=groups,
            )
            result = {
                "variant": variant.name,
                "retrieval_mode": variant.retrieval_mode,
                "prompt_format": variant.prompt_format,
                "policy_mode": variant.policy_mode,
                "do_sample": variant.do_sample,
                "temperature": variant.temperature,
                "top_p": variant.top_p,
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

    control_prompt_parity = None
    if unified_skill_causal:
        control_prompt_parity = compare_unified_prompt_controls(
            groups_by_variant["unified-no-skill-deterministic"],
            groups_by_variant["unified-empty-skills-deterministic"],
        )

    payload = {
        "schema_version": 1,
        **diagnostic_manifest,
        "runtime_initialize_seconds": initialize_seconds,
        "runtime_close_seconds": close_seconds,
        "total_seconds": time.perf_counter() - diagnostic_started,
        "results": variant_results,
    }
    if control_prompt_parity is not None:
        payload["control_prompt_parity"] = control_prompt_parity
    _write_json(run_directory / "raw_skill_ab_summary.json", payload)
    logger.info(
        "Raw-skill A/B diagnostic complete:\n%s",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    if control_prompt_parity is not None and not control_prompt_parity["passed"]:
        logger.error(
            "Unified prompt control parity failed; diagnostic is invalid"
        )
        return 4
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
        policy_max_tokens_per_gpu=DEFAULT_POLICY_MAX_TOKENS_PER_GPU,
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
    if (
        args.worker_batch_size <= 0
        or args.worker_processes <= 0
        or args.native_batch_size <= 0
        or args.worker_inactivity_timeout_seconds <= 0
    ):
        raise ValueError("grounding worker sizes must be positive")
    if args.resume_run and args.run_name:
        raise ValueError("grounding resume-run cannot be combined with run-name")
    from tqdm.auto import tqdm

    from infoskill.integrations.alfworld import (
        GroundingWorkItem,
        build_grounding_manifest,
        discover_tasks,
        prepare_alfworld_expert_type_binding,
        run_bounded_grounding,
        write_grounding_artifacts,
    )
    from infoskill.skills import (
        FixedSkillLibrary,
        PrecomputedEmbeddingRetriever,
        SentenceTransformerEncoder,
        TemplateRetriever,
    )

    if args.resume_run:
        run_directory = Path(args.resume_run).expanduser().resolve()
        if not run_directory.is_dir():
            raise FileNotFoundError(
                f"grounding resume run does not exist: {run_directory}"
            )
    else:
        run_directory = _run_directory(config, args.run_name or "grounding")
    logger = _configure_logging(run_directory)
    tasks = discover_tasks(config.paths.alfworld_data, split="train")
    expert_binding = prepare_alfworld_expert_type_binding(
        config.paths.alfworld_source,
        requested_expert_type="planner",
    )
    library = FixedSkillLibrary.load(config.paths.skill_bank)
    if config.retrieval_mode == "embedding":
        encoder = SentenceTransformerEncoder(
            config.paths.semantic_model,
            device="cuda:0",
            show_progress_bar=True,
        )
        try:
            retriever = PrecomputedEmbeddingRetriever(
                library,
                encoder,
                queries=[task.goal for task in tasks],
                general_top_k=config.general_top_k,
                task_top_k=config.task_top_k,
                mistake_count=config.mistake_count,
            )
        finally:
            encoder.close()
    else:
        retriever = TemplateRetriever(
            library,
            general_count=config.general_top_k,
            task_count=config.task_top_k,
            mistake_count=config.mistake_count,
        )
    work_items = tuple(
        GroundingWorkItem(
            task=task,
            candidate_skill_ids=retriever.retrieve(task.goal).skill_ids,
            seed=_stable_seed(config.master_seed, task.task_id),
        )
        for task in tasks
    )
    source_checksum = _source_checksum()
    with tqdm(
        total=len(tasks),
        desc="train/expert-replay",
        unit="task",
        dynamic_ncols=True,
    ) as progress:
        results, lifecycle = run_bounded_grounding(
            work_items=work_items,
            config_path=args.config,
            run_directory=run_directory,
            worker_batch_size=args.worker_batch_size,
            worker_processes=args.worker_processes,
            max_replay_steps=150,
            persist_horizon=config.max_steps,
            expert_type="planner",
            replay_backend=args.replay_backend,
            native_batch_size=args.native_batch_size,
            worker_inactivity_timeout_seconds=(
                args.worker_inactivity_timeout_seconds
            ),
            source_checksum=source_checksum,
            on_progress=progress.update,
        )
    manifest = build_grounding_manifest(
        results=results,
        source_checksums={
            "skill_bank": library.source_sha256,
            "train_task_manifest": _task_manifest_checksum(tasks),
            "infoskill_source": source_checksum,
        },
        code_revision=source_checksum[:16],
        max_replay_steps=150,
        persist_horizon=config.max_steps,
        expert_type="planner",
        expert_binding=expert_binding,
    )
    write_grounding_artifacts(output_directory=run_directory, results=results, manifest=manifest)
    _write_json(run_directory / "grounding-lifecycle.json", _dataclass_dict(lifecycle))
    logger.info(
        "Grounding workers: backend=%s count=%d concurrency=%d peak=%d "
        "environment_slots=%d batch_size=%d tasks=%d temp_cleanup=%s "
        "min_free=%.2f GiB",
        lifecycle.replay_backend,
        lifecycle.worker_processes_started,
        lifecycle.worker_concurrency,
        lifecycle.peak_worker_processes,
        lifecycle.peak_environment_slots,
        lifecycle.worker_batch_size,
        lifecycle.processed_tasks,
        lifecycle.temporary_directories_cleaned,
        lifecycle.minimum_free_disk_bytes / 1024**3,
    )
    logger.info(
        "Grounding expert identity: requested=%s effective=%s guard=%s corrected=%s",
        expert_binding["requested_expert_type"],
        expert_binding["effective_expert_type"],
        expert_binding["compatibility_guard_active"],
        expert_binding["positional_binding_corrected"],
    )
    logger.info("Grounding manifest:\n%s", json.dumps(manifest.__dict__ if hasattr(manifest, "__dict__") else _dataclass_dict(manifest), ensure_ascii=False, indent=2))
    return 0 if manifest.formal_gate_passed else 4


def _grounding_timeout_rescue(
    config: AppConfig,
    args: argparse.Namespace,
) -> int:
    """Retry only committed planner timeouts and derive a formal dataset."""

    _validate_paths(
        config,
        mode=SkillMode.INFO_SKILL,
        require_checkpoint=False,
    )
    if args.worker_processes <= 0 or args.worker_inactivity_timeout_seconds <= 0:
        raise ValueError("grounding rescue worker settings must be positive")
    if args.resume_run and args.run_name:
        raise ValueError("grounding rescue resume-run cannot be combined with run-name")
    if args.resume_run and args.finalize_committed_rescue_run:
        raise ValueError(
            "grounding rescue resume-run cannot be combined with committed "
            "snapshot finalization"
        )

    from tqdm.auto import tqdm

    from infoskill.integrations.alfworld import (
        GroundingWorkItem,
        build_grounding_manifest,
        discover_tasks,
        grounding_work_items_sha256,
        load_available_committed_grounding_results,
        load_committed_grounding_results,
        merge_timeout_grounding_results,
        prepare_alfworld_expert_type_binding,
        run_bounded_grounding,
        select_timeout_work_items,
        sha256_file,
        write_grounding_artifacts,
        write_serialized_grounding_results,
    )
    from infoskill.skills import (
        FixedSkillLibrary,
        PrecomputedEmbeddingRetriever,
        SentenceTransformerEncoder,
        TemplateRetriever,
    )

    source_run = Path(args.source_grounding_run).expanduser().resolve()
    source_manifest_path = source_run / "manifest.json"
    source_resume_path = source_run / "grounding-resume.json"
    if not source_manifest_path.is_file() or not source_resume_path.is_file():
        raise FileNotFoundError(
            "source grounding run must contain manifest.json and "
            "grounding-resume.json"
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_resume = json.loads(source_resume_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != 2:
        raise ValueError("grounding rescue requires a schema v2 source manifest")
    if source_manifest.get("source_split") != "train":
        raise ValueError("grounding rescue source must use the train split")
    if source_manifest.get("expert_identity_gate_passed") is not True:
        raise ValueError("grounding rescue source failed the planner identity gate")
    if source_manifest.get("expert_type") != "planner":
        raise ValueError("grounding rescue source is not planner-generated")
    source_failures = source_manifest.get("formal_gate_failures")
    if source_failures != ["success_coverage_below_threshold"]:
        raise ValueError(
            "grounding rescue only accepts a source whose sole formal failure "
            "is success coverage"
        )
    source_checksums = source_manifest.get("source_checksums")
    if not isinstance(source_checksums, dict):
        raise ValueError("grounding rescue source lacks source checksums")
    if source_resume.get("source_checksum") != source_checksums.get(
        "infoskill_source"
    ):
        raise ValueError(
            "source manifest and resume plan disagree on INFO-SKILL source"
        )

    tasks = discover_tasks(config.paths.alfworld_data, split="train")
    task_manifest_checksum = _task_manifest_checksum(tasks)
    if source_manifest.get("total_games") != len(tasks):
        raise ValueError("source grounding task count differs from current train data")
    if source_checksums.get("train_task_manifest") != task_manifest_checksum:
        raise ValueError("source grounding train task manifest has changed")

    expert_binding = prepare_alfworld_expert_type_binding(
        config.paths.alfworld_source,
        requested_expert_type="planner",
    )
    library = FixedSkillLibrary.load(config.paths.skill_bank)
    if source_checksums.get("skill_bank") != library.source_sha256:
        raise ValueError("source grounding skill bank has changed")
    if config.retrieval_mode == "embedding":
        encoder = SentenceTransformerEncoder(
            config.paths.semantic_model,
            device="cuda:0",
            show_progress_bar=True,
        )
        try:
            retriever = PrecomputedEmbeddingRetriever(
                library,
                encoder,
                queries=[task.goal for task in tasks],
                general_top_k=config.general_top_k,
                task_top_k=config.task_top_k,
                mistake_count=config.mistake_count,
            )
        finally:
            encoder.close()
    else:
        retriever = TemplateRetriever(
            library,
            general_count=config.general_top_k,
            task_count=config.task_top_k,
            mistake_count=config.mistake_count,
        )
    all_work_items = tuple(
        GroundingWorkItem(
            task=task,
            candidate_skill_ids=retriever.retrieve(task.goal).skill_ids,
            seed=_stable_seed(config.master_seed, task.task_id),
        )
        for task in tasks
    )
    current_work_items_sha256 = grounding_work_items_sha256(all_work_items)
    if source_resume.get("work_items_sha256") != current_work_items_sha256:
        raise ValueError(
            "source grounding work items changed (task, retrieval, or seed)"
        )
    source_results = load_committed_grounding_results(source_run, all_work_items)
    rescue_work_items = select_timeout_work_items(all_work_items, source_results)
    result_successes = sum(result.succeeded for _, result in source_results)
    result_reasons = Counter(
        result.quarantine_reason or "unknown"
        for _, result in source_results
        if not result.succeeded
    )
    if source_manifest.get("successful_games") != result_successes:
        raise ValueError(
            "source manifest success count differs from committed shard results"
        )
    if source_manifest.get("quarantined_games") != (
        len(source_results) - result_successes
    ):
        raise ValueError(
            "source manifest quarantine count differs from committed shard results"
        )
    manifest_reasons = source_manifest.get("quarantine_reasons")
    if not isinstance(manifest_reasons, dict) or manifest_reasons != dict(
        sorted(result_reasons.items())
    ):
        raise ValueError(
            "source manifest quarantine reasons differ from committed shard results"
        )
    expected_timeouts = manifest_reasons.get("expert_wall_timeout")
    if expected_timeouts != len(rescue_work_items):
        raise ValueError(
            "source manifest timeout count differs from committed shard results"
        )
    if not rescue_work_items:
        raise ValueError("source grounding run has no timeout rows to rescue")

    committed_rescue_run = None
    if args.finalize_committed_rescue_run:
        committed_rescue_run = Path(
            args.finalize_committed_rescue_run
        ).expanduser().resolve()
        if not committed_rescue_run.is_dir():
            raise FileNotFoundError(
                "committed timeout-rescue run does not exist: "
                f"{committed_rescue_run}"
            )
        run_directory = _run_directory(
            config,
            args.run_name or "grounding-timeout-rescue-finalized",
        )
    elif args.resume_run:
        run_directory = Path(args.resume_run).expanduser().resolve()
        if not run_directory.is_dir():
            raise FileNotFoundError(
                f"grounding rescue run does not exist: {run_directory}"
            )
    else:
        run_directory = _run_directory(
            config,
            args.run_name or "grounding-timeout-rescue",
        )
    logger = _configure_logging(run_directory)
    source_manifest_sha256 = sha256_file(source_manifest_path)
    current_source_checksum = _source_checksum()
    rescue_plan_source_checksum = hashlib.sha256(
        (
            f"{current_source_checksum}\0{source_manifest_sha256}\0"
            f"{current_work_items_sha256}"
        ).encode("utf-8")
    ).hexdigest()
    if committed_rescue_run is not None:
        rescue_resume_path = committed_rescue_run / "grounding-resume.json"
        if not rescue_resume_path.is_file():
            raise FileNotFoundError(
                "committed timeout-rescue run lacks grounding-resume.json: "
                f"{committed_rescue_run}"
            )
        rescue_resume = json.loads(
            rescue_resume_path.read_text(encoding="utf-8")
        )
        rescue_plan_checks = {
            "worker_batch_size": rescue_resume.get("worker_batch_size") == 1,
            "expert_type": rescue_resume.get("expert_type") == "planner",
            "replay_backend": rescue_resume.get("replay_backend") == "individual",
            "native_batch_size": rescue_resume.get("native_batch_size") == 1,
            "max_replay_steps": rescue_resume.get("max_replay_steps")
            == int(source_manifest["max_replay_steps"]),
            "persist_horizon": rescue_resume.get("persist_horizon")
            == int(source_manifest["persist_horizon"]),
            "source_checksum": isinstance(
                rescue_resume.get("source_checksum"), str
            ),
        }
        failed_rescue_plan_checks = [
            name for name, passed in rescue_plan_checks.items() if not passed
        ]
        if failed_rescue_plan_checks:
            raise ValueError(
                "committed timeout-rescue plan is incompatible with the "
                f"formal source: {failed_rescue_plan_checks}"
            )
        rescue_results = load_available_committed_grounding_results(
            committed_rescue_run,
            rescue_work_items,
        )
        if not rescue_results:
            raise ValueError("committed timeout-rescue snapshot contains no results")
        lifecycle_payload = {
            "schema_version": 1,
            "mode": "committed_rescue_snapshot",
            "source_rescue_run": str(committed_rescue_run),
            "source_rescue_plan_sha256": rescue_resume.get("plan_sha256"),
            "snapshot_committed_tasks": len(rescue_results),
        }
        report_worker_processes = rescue_resume.get("worker_processes")
        report_timeout_seconds = rescue_resume.get(
            "worker_inactivity_timeout_seconds"
        )
        report_temporary_cleaned = None
        report_minimum_free_disk_bytes = None
    else:
        with tqdm(
            total=len(rescue_work_items),
            desc="train/planner-timeout-rescue",
            unit="task",
            dynamic_ncols=True,
        ) as progress:
            rescue_results, lifecycle = run_bounded_grounding(
                work_items=rescue_work_items,
                config_path=args.config,
                run_directory=run_directory,
                worker_batch_size=1,
                worker_processes=args.worker_processes,
                max_replay_steps=int(source_manifest["max_replay_steps"]),
                persist_horizon=int(source_manifest["persist_horizon"]),
                expert_type="planner",
                replay_backend="individual",
                native_batch_size=1,
                worker_inactivity_timeout_seconds=(
                    args.worker_inactivity_timeout_seconds
                ),
                source_checksum=rescue_plan_source_checksum,
                on_progress=progress.update,
            )
        lifecycle_payload = _dataclass_dict(lifecycle)
        report_worker_processes = args.worker_processes
        report_timeout_seconds = args.worker_inactivity_timeout_seconds
        report_temporary_cleaned = lifecycle.temporary_directories_cleaned
        report_minimum_free_disk_bytes = lifecycle.minimum_free_disk_bytes
    merge = merge_timeout_grounding_results(source_results, rescue_results)
    derived_manifest = build_grounding_manifest(
        results=merge.results,
        source_checksums={
            "base_grounding_manifest": source_manifest_sha256,
            "base_infoskill_source": str(source_checksums["infoskill_source"]),
            "infoskill_source": current_source_checksum,
            "skill_bank": library.source_sha256,
            "train_task_manifest": task_manifest_checksum,
        },
        code_revision=current_source_checksum[:16],
        max_replay_steps=int(source_manifest["max_replay_steps"]),
        persist_horizon=int(source_manifest["persist_horizon"]),
        expert_type="planner",
        expert_binding=expert_binding,
    )
    write_serialized_grounding_results(
        run_directory / "rescue-results.jsonl",
        rescue_results,
    )
    write_grounding_artifacts(
        output_directory=run_directory,
        results=merge.results,
        manifest=derived_manifest,
    )
    _write_json(
        run_directory / "grounding-rescue-lifecycle.json",
        lifecycle_payload,
    )
    rescue_report = {
        "schema_version": 1,
        "source_grounding_run": str(source_run),
        "source_manifest_sha256": source_manifest_sha256,
        "source_work_items_sha256": current_work_items_sha256,
        "source_results_validated": len(source_results),
        "attempted_tasks": len(merge.attempted_task_ids),
        "rescued_tasks": len(merge.rescued_task_ids),
        "remaining_timeout_tasks": len(merge.remaining_timeout_task_ids),
        "attempted_task_ids": list(merge.attempted_task_ids),
        "rescued_task_ids": list(merge.rescued_task_ids),
        "remaining_timeout_task_ids": list(
            merge.remaining_timeout_task_ids
        ),
        "source_successful_games": source_manifest["successful_games"],
        "derived_successful_games": derived_manifest.successful_games,
        "source_success_coverage": source_manifest["success_coverage"],
        "derived_success_coverage": derived_manifest.success_coverage,
        "derived_formal_gate_passed": derived_manifest.formal_gate_passed,
        "derived_formal_gate_failures": list(
            derived_manifest.formal_gate_failures
        ),
        "committed_rescue_snapshot_run": (
            str(committed_rescue_run) if committed_rescue_run is not None else None
        ),
        "worker_processes": report_worker_processes,
        "worker_inactivity_timeout_seconds": report_timeout_seconds,
        "temporary_directories_cleaned": report_temporary_cleaned,
        "minimum_free_disk_bytes": report_minimum_free_disk_bytes,
    }
    _write_json(run_directory / "grounding-rescue-report.json", rescue_report)
    logger.info(
        "Grounding timeout rescue: attempted=%d rescued=%d remaining=%d "
        "coverage=%.6f formal_gate=%s min_free=%.2f GiB",
        len(merge.attempted_task_ids),
        len(merge.rescued_task_ids),
        len(merge.remaining_timeout_task_ids),
        derived_manifest.success_coverage,
        derived_manifest.formal_gate_passed,
        (
            report_minimum_free_disk_bytes / 1024**3
            if report_minimum_free_disk_bytes is not None
            else float("nan")
        ),
    )
    logger.info("Grounding rescue report: %s", run_directory / "grounding-rescue-report.json")
    return 0 if derived_manifest.formal_gate_passed else 4


def _grounding_expert_diagnostic(
    config: AppConfig,
    args: argparse.Namespace,
) -> int:
    _validate_paths(
        config,
        mode=SkillMode.NO_SKILL,
        require_checkpoint=False,
    )
    if args.tasks_per_type <= 0 or args.max_replay_steps <= 0:
        raise ValueError("diagnostic task count and replay limit must be positive")
    from tqdm.auto import tqdm

    from infoskill.integrations.alfworld import discover_tasks, sha256_file
    from infoskill.integrations.alfworld.grounding_diagnostic import (
        run_expert_diagnostic,
        select_quarantined_tasks,
    )

    source = Path(args.source_grounding_run).expanduser().resolve()
    manifest_path = source / "manifest.json"
    quarantine_path = source / "quarantine.jsonl"
    if not manifest_path.is_file() or not quarantine_path.is_file():
        raise FileNotFoundError(
            "source grounding run requires manifest.json and quarantine.jsonl"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("source_split") != "train":
        raise ValueError("source grounding manifest must describe the train split")
    quarantine_rows = tuple(
        json.loads(line)
        for line in quarantine_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if any(not isinstance(row, dict) for row in quarantine_rows):
        raise ValueError("source quarantine rows must be JSON objects")
    tasks = discover_tasks(config.paths.alfworld_data, split="train")
    source_checksums = manifest.get("source_checksums", {})
    expected_task_checksum = (
        source_checksums.get("train_task_manifest")
        if isinstance(source_checksums, dict)
        else None
    )
    current_task_checksum = _task_manifest_checksum(tasks)
    if expected_task_checksum != current_task_checksum:
        raise ValueError(
            "source grounding run does not match the current train task manifest"
        )
    selected = select_quarantined_tasks(
        tasks=tasks,
        quarantine_rows=quarantine_rows,
        tasks_per_type=args.tasks_per_type,
        selection_seed=config.master_seed,
    )
    run_directory = _run_directory(
        config,
        args.run_name or "grounding-expert-diagnostic",
    )
    logger = _configure_logging(run_directory)
    with tqdm(
        total=len(selected) * 2,
        desc="grounding/expert-diagnostic",
        unit="replay",
        dynamic_ncols=True,
    ) as progress:
        report = run_expert_diagnostic(
            config=config,
            tasks=tasks,
            quarantine_rows=quarantine_rows,
            tasks_per_type=args.tasks_per_type,
            selection_seed=config.master_seed,
            max_replay_steps=args.max_replay_steps,
            persist_horizon=config.max_steps,
            seed_for_task=lambda task_id: _stable_seed(
                config.master_seed,
                task_id,
            ),
            on_progress=progress.update,
        )
    report.update(
        {
            "source_grounding_run": str(source),
            "source_manifest_sha256": sha256_file(manifest_path),
            "train_task_manifest_sha256": current_task_checksum,
            "code_revision": _source_checksum()[:16],
            "max_replay_steps": args.max_replay_steps,
            "persist_horizon": config.max_steps,
        }
    )
    output = run_directory / "expert-diagnostic.json"
    _write_json(output, report)
    summary = report["summary"]
    if not isinstance(summary, dict):
        raise TypeError("expert diagnostic summary must be a mapping")
    handcoded_summary = summary["handcoded"]
    planner_summary = summary["planner"]
    if not isinstance(handcoded_summary, dict) or not isinstance(
        planner_summary,
        dict,
    ):
        raise TypeError("expert diagnostic variants must be mappings")
    logger.info(
        "Expert diagnostic complete: selected=%d handcoded_success=%d "
        "planner_success=%d planner_rescues=%d",
        report["selected_task_count"],
        handcoded_summary["success_count"],
        planner_summary["success_count"],
        summary["planner_rescue_count"],
    )
    logger.info("Diagnostic report: %s", output)
    return 0


def _grounding_planner_pilot(
    config: AppConfig,
    args: argparse.Namespace,
) -> int:
    _validate_paths(
        config,
        mode=SkillMode.NO_SKILL,
        require_checkpoint=False,
    )
    if (
        args.tasks_per_type <= 0
        or args.worker_batch_size <= 0
        or args.worker_processes <= 0
        or args.max_replay_steps <= 0
        or args.native_batch_size <= 0
    ):
        raise ValueError("planner pilot sizes and replay limit must be positive")
    from tqdm.auto import tqdm

    from infoskill.integrations.alfworld import (
        GroundingWorkItem,
        build_planner_pilot_report,
        compact_result_payload,
        discover_tasks,
        prepare_alfworld_expert_type_binding,
        run_bounded_grounding,
        select_stratified_tasks,
        write_planner_pilot_results,
    )

    all_tasks = discover_tasks(config.paths.alfworld_data, split="train")
    expert_binding = prepare_alfworld_expert_type_binding(
        config.paths.alfworld_source,
        requested_expert_type="planner",
    )
    selected = select_stratified_tasks(
        tasks=all_tasks,
        tasks_per_type=args.tasks_per_type,
        selection_seed=config.master_seed,
    )
    seeds = {
        task.task_id: _stable_seed(config.master_seed, task.task_id)
        for task in selected
    }
    work_items = tuple(
        GroundingWorkItem(
            task=task,
            candidate_skill_ids=(),
            seed=seeds[task.task_id],
        )
        for task in selected
    )
    run_directory = _run_directory(
        config,
        args.run_name or "grounding-planner-pilot",
    )
    logger = _configure_logging(run_directory)
    with tqdm(
        total=len(selected),
        desc="train/planner-pilot",
        unit="task",
        dynamic_ncols=True,
    ) as progress:
        results, lifecycle = run_bounded_grounding(
            work_items=work_items,
            config_path=args.config,
            run_directory=run_directory,
            worker_batch_size=args.worker_batch_size,
            worker_processes=args.worker_processes,
            max_replay_steps=args.max_replay_steps,
            persist_horizon=config.max_steps,
            expert_type="planner",
            replay_backend=args.replay_backend,
            native_batch_size=args.native_batch_size,
            on_progress=progress.update,
        )

    result_by_id = {result.task_id: result for _, result in results}
    rows = tuple(
        compact_result_payload(
            task=task,
            seed=seeds[task.task_id],
            result=result_by_id[task.task_id],
        )
        for task in selected
    )
    source_checksum = _source_checksum()
    report = build_planner_pilot_report(
        results=results,
        tasks_per_type=args.tasks_per_type,
        selection_seed=config.master_seed,
        train_task_manifest_sha256=_task_manifest_checksum(all_tasks),
        code_revision=source_checksum,
        max_replay_steps=args.max_replay_steps,
        persist_horizon=config.max_steps,
        expert_binding=expert_binding,
    )
    write_planner_pilot_results(
        run_directory / "planner-pilot-results.jsonl",
        rows,
    )
    _write_json(run_directory / "planner-pilot.json", report)
    _write_json(
        run_directory / "grounding-lifecycle.json",
        _dataclass_dict(lifecycle),
    )
    logger.info(
        "Planner pilot complete: selected=%d success=%d coverage=%.4f "
        "over_%d_steps=%d gate=%s backend=%s concurrency=%d peak=%d slots=%d",
        report["selected_games"],
        report["successful_games"],
        report["success_coverage"],
        config.max_steps,
        report["over_persist_horizon"],
        report["pilot_gate_passed"],
        lifecycle.replay_backend,
        lifecycle.worker_concurrency,
        lifecycle.peak_worker_processes,
        lifecycle.peak_environment_slots,
    )
    logger.info(
        "Planner identity: requested=%s effective=%s guard=%s corrected=%s",
        expert_binding["requested_expert_type"],
        expert_binding["effective_expert_type"],
        expert_binding["compatibility_guard_active"],
        expert_binding["positional_binding_corrected"],
    )
    logger.info(
        "Pilot-only report: %s (a full 3,553-task formal run is still required)",
        run_directory / "planner-pilot.json",
    )
    return 0


def _grounding_planner_parity(
    config: AppConfig,
    args: argparse.Namespace,
) -> int:
    """Fail closed unless serial and parallel planner replays are identical."""

    _validate_paths(
        config,
        mode=SkillMode.NO_SKILL,
        require_checkpoint=False,
    )
    if (
        args.tasks_per_type <= 0
        or args.worker_batch_size <= 0
        or args.max_replay_steps <= 0
        or args.native_batch_size <= 0
        or args.minimum_speedup < 0
        or (
            args.candidate_backend
            in {"process_parallel", "native_batch_parallel"}
            and args.parallel_workers < 2
        )
    ):
        raise ValueError(
            "planner parity sizes must be positive; parallel candidates "
            "must use at least 2 workers"
        )
    from tqdm.auto import tqdm

    from infoskill.integrations.alfworld import (
        GroundingWorkItem,
        build_grounding_parity_report,
        discover_tasks,
        prepare_alfworld_expert_type_binding,
        run_bounded_grounding,
        select_stratified_tasks,
        write_serialized_grounding_results,
    )

    all_tasks = discover_tasks(config.paths.alfworld_data, split="train")
    expert_binding = prepare_alfworld_expert_type_binding(
        config.paths.alfworld_source,
        requested_expert_type="planner",
    )
    selected = select_stratified_tasks(
        tasks=all_tasks,
        tasks_per_type=args.tasks_per_type,
        selection_seed=config.master_seed,
    )
    if (
        args.candidate_backend
        in {"process_parallel", "native_batch_parallel"}
        and len(selected) <= args.worker_batch_size
    ):
        raise ValueError(
            "planner parity requires more selected tasks than worker batch size "
            "so at least two worker processes can overlap"
        )
    work_items = tuple(
        GroundingWorkItem(
            task=task,
            candidate_skill_ids=(),
            seed=_stable_seed(config.master_seed, task.task_id),
        )
        for task in selected
    )
    run_directory = _run_directory(
        config,
        args.run_name or "grounding-planner-parity",
    )
    logger = _configure_logging(run_directory)
    serial_directory = run_directory / "serial"
    parallel_directory = run_directory / "parallel"
    with tqdm(
        total=2 * len(selected),
        desc="train/planner-parity",
        unit="task",
        dynamic_ncols=True,
    ) as progress:
        started = time.perf_counter()
        serial_results, serial_lifecycle = run_bounded_grounding(
            work_items=work_items,
            config_path=args.config,
            run_directory=serial_directory,
            worker_batch_size=args.worker_batch_size,
            worker_processes=1,
            max_replay_steps=args.max_replay_steps,
            persist_horizon=config.max_steps,
            expert_type="planner",
            on_progress=progress.update,
        )
        serial_seconds = time.perf_counter() - started

        started = time.perf_counter()
        parallel_results, parallel_lifecycle = run_bounded_grounding(
            work_items=work_items,
            config_path=args.config,
            run_directory=parallel_directory,
            worker_batch_size=args.worker_batch_size,
            worker_processes=(
                args.parallel_workers
                if args.candidate_backend
                in {"process_parallel", "native_batch_parallel"}
                else 1
            ),
            max_replay_steps=args.max_replay_steps,
            persist_horizon=config.max_steps,
            expert_type="planner",
            replay_backend=(
                "native_batch"
                if args.candidate_backend
                in {"native_batch", "native_batch_parallel"}
                else "individual"
            ),
            native_batch_size=args.native_batch_size,
            on_progress=progress.update,
        )
        parallel_seconds = time.perf_counter() - started

    source_checksum = _source_checksum()
    report = build_grounding_parity_report(
        serial_results=serial_results,
        parallel_results=parallel_results,
        serial_lifecycle=serial_lifecycle,
        parallel_lifecycle=parallel_lifecycle,
        serial_seconds=serial_seconds,
        parallel_seconds=parallel_seconds,
        tasks_per_type=args.tasks_per_type,
        selection_seed=config.master_seed,
        train_task_manifest_sha256=_task_manifest_checksum(all_tasks),
        code_revision=source_checksum,
        expert_binding=expert_binding,
        minimum_speedup=args.minimum_speedup,
    )
    write_serialized_grounding_results(
        run_directory / "serial-results.jsonl",
        serial_results,
    )
    write_serialized_grounding_results(
        run_directory / "parallel-results.jsonl",
        parallel_results,
    )
    _write_json(
        run_directory / "serial-lifecycle.json",
        _dataclass_dict(serial_lifecycle),
    )
    _write_json(
        run_directory / "parallel-lifecycle.json",
        _dataclass_dict(parallel_lifecycle),
    )
    _write_json(run_directory / "planner-parity.json", report)
    logger.info(
        "Planner serial/parallel parity: passed=%s exact_fields=%s "
        "serial=%.1fs candidate=%.1fs speedup=%.2fx backend=%s "
        "peak_workers=%d peak_environment_slots=%d performance_gate=%s",
        report["passed"],
        all(report["field_checks"].values()),
        serial_seconds,
        parallel_seconds,
        report["speedup"],
        parallel_lifecycle.replay_backend,
        parallel_lifecycle.peak_worker_processes,
        parallel_lifecycle.peak_environment_slots,
        report["performance_passed"],
    )
    logger.info("Parity report: %s", run_directory / "planner-parity.json")
    return 0 if report["passed"] else 5


def _grounding_planner_loop_diagnostic(
    config: AppConfig,
    args: argparse.Namespace,
) -> int:
    _validate_paths(
        config,
        mode=SkillMode.NO_SKILL,
        require_checkpoint=False,
    )
    if args.successful_two_object_controls <= 0 or args.max_replay_steps <= 0:
        raise ValueError("planner loop diagnostic sizes must be positive")
    from tqdm.auto import tqdm

    from infoskill.integrations.alfworld import (
        build_planner_loop_report,
        discover_tasks,
        run_planner_loop_diagnostic,
        select_loop_diagnostic_rows,
        sha256_file,
        write_planner_loop_rows,
    )

    source = Path(args.source_pilot_run).expanduser().resolve()
    source_report_path = source / "planner-pilot.json"
    source_results_path = source / "planner-pilot-results.jsonl"
    if not source_report_path.is_file() or not source_results_path.is_file():
        raise FileNotFoundError(
            "source planner pilot requires planner-pilot.json and "
            "planner-pilot-results.jsonl"
        )
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if not isinstance(source_report, dict) or source_report.get("pilot_only") is not True:
        raise ValueError("source run is not an explicit planner-only pilot")
    if source_report.get("expert_identity_gate_passed") is not True:
        raise ValueError(
            "source planner pilot did not pass the effective planner identity gate"
        )
    source_rows = tuple(
        json.loads(line)
        for line in source_results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if any(not isinstance(row, dict) for row in source_rows):
        raise ValueError("source planner pilot rows must be JSON objects")
    if len(source_rows) != int(source_report.get("selected_games", -1)):
        raise ValueError("source planner pilot report/result counts disagree")
    source_max_steps = int(source_report.get("max_replay_steps", 0))
    if source_max_steps <= 0:
        raise ValueError("source planner pilot has an invalid replay horizon")
    if args.max_replay_steps <= source_max_steps:
        raise ValueError(
            "planner loop diagnostic max replay steps must exceed the source pilot"
        )

    tasks = discover_tasks(config.paths.alfworld_data, split="train")
    current_manifest = _task_manifest_checksum(tasks)
    source_checksums = source_report.get("source_checksums")
    if not isinstance(source_checksums, dict) or source_checksums.get(
        "train_task_manifest"
    ) != current_manifest:
        raise ValueError("source planner pilot does not match current train tasks")
    selected_rows = select_loop_diagnostic_rows(
        rows=source_rows,
        successful_two_object_controls=args.successful_two_object_controls,
        persist_horizon=config.max_steps,
    )
    run_directory = _run_directory(
        config,
        args.run_name or "grounding-planner-loop-diagnostic",
    )
    logger = _configure_logging(run_directory)
    task_by_id = {task.task_id: task for task in tasks}
    minimum_free = shutil.disk_usage(run_directory).free
    temporary_path: Path | None = None
    old_tempdir = tempfile.tempdir
    temporary_variables = {
        name: os.environ.get(name) for name in ("TMPDIR", "TMP", "TEMP")
    }
    try:
        with tempfile.TemporaryDirectory(
            prefix="planner-loop-diagnostic-",
            dir=run_directory,
        ) as temporary:
            temporary_path = Path(temporary)
            tempfile.tempdir = temporary
            for name in temporary_variables:
                os.environ[name] = temporary

            def update_progress(count: int) -> None:
                nonlocal minimum_free
                minimum_free = min(
                    minimum_free,
                    shutil.disk_usage(run_directory).free,
                )
                progress.update(count)

            with tqdm(
                total=len(selected_rows),
                desc="grounding/planner-loop-diagnostic",
                unit="task",
                dynamic_ncols=True,
            ) as progress:
                rows = run_planner_loop_diagnostic(
                    config=config,
                    tasks=task_by_id,
                    selected_rows=selected_rows,
                    max_replay_steps=args.max_replay_steps,
                    persist_horizon=config.max_steps,
                    on_progress=update_progress,
                )
    finally:
        tempfile.tempdir = old_tempdir
        for name, value in temporary_variables.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    minimum_free = min(minimum_free, shutil.disk_usage(run_directory).free)
    temporary_cleaned = temporary_path is not None and not temporary_path.exists()
    source_checksum = _source_checksum()
    report = build_planner_loop_report(
        rows=rows,
        source_pilot_run=str(source),
        source_pilot_sha256=sha256_file(source_report_path),
        source_results_sha256=sha256_file(source_results_path),
        train_task_manifest_sha256=current_manifest,
        code_revision=source_checksum,
        source_max_replay_steps=source_max_steps,
        diagnostic_max_replay_steps=args.max_replay_steps,
        successful_two_object_controls=args.successful_two_object_controls,
        temporary_directory_cleaned=temporary_cleaned,
        minimum_free_disk_bytes=minimum_free,
    )
    write_planner_loop_rows(
        run_directory / "planner-loop-traces.jsonl",
        rows,
    )
    _write_json(run_directory / "planner-loop-diagnostic.json", report)
    logger.info(
        "Planner loop diagnostic complete: selected=%d rescued=%d "
        "still_failed=%d terminal_cycles=%d control_regressions=%d",
        report["selected_tasks"],
        report["rescued_after_source_horizon"],
        report["still_failed_at_diagnostic_horizon"],
        report["terminal_cycles_detected"],
        report["control_regressions"],
    )
    logger.info("Diagnostic report: %s", run_directory / "planner-loop-diagnostic.json")
    return 0


def _validate_paths(
    config: AppConfig,
    *,
    mode: SkillMode,
    require_checkpoint: bool,
    require_training_runtime: bool = False,
    require_grounding: bool = False,
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
    if require_grounding:
        if mode is not SkillMode.INFO_SKILL:
            raise ValueError("grounding data is only valid for infoskill training")
        if not config.paths.grounding_data:
            raise ValueError("infoskill training requires paths.grounding_data")
        required["grounding_data"] = config.paths.grounding_data
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
