from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from infoskill.app_config import AppConfig
from infoskill.config import EvaluationConfig, SkillMode
from infoskill.training import TrainingProfile, resolve_training_plan


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = AppConfig.load(args.config)
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
    evaluate = subparsers.add_parser("eval", help="run the complete ALFWorld valid_seen evaluation")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--mode", choices=[mode.value for mode in SkillMode], required=True)
    evaluate.add_argument("--run-name")
    evaluate.add_argument("--checkpoint-step", type=int, default=0)
    grounding = subparsers.add_parser("grounding", help="generate strict train-only expert labels")
    grounding.add_argument("--config", required=True)
    grounding.add_argument("--run-name")
    train = subparsers.add_parser("train", help="run INFO-SKILL-owned GRPO training")
    train.add_argument("--config", required=True)
    train.add_argument("--mode", choices=[mode.value for mode in SkillMode], required=True)
    train.add_argument(
        "--profile",
        choices=[profile.value for profile in TrainingProfile],
        default=TrainingProfile.SMOKE.value,
    )
    train.add_argument("--max-updates", type=int)
    train.add_argument("--num-gpus", type=int, required=True)
    train.add_argument("--run-name")
    train.add_argument("--resume")
    train.add_argument("--dry-run", action="store_true")
    return parser


def _train(config: AppConfig, args: argparse.Namespace) -> int:
    mode = SkillMode(args.mode)
    if mode is not SkillMode.NO_SKILL:
        raise NotImplementedError(
            "the first training slice supports only mode=no_skill; "
            "raw_skill_prompt and infoskill remain fail-fast"
        )
    if args.num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if args.resume and args.run_name:
        raise ValueError("run_name cannot be changed while resuming a training run")
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
                    "profile": plan.profile.value,
                    "max_updates": plan.max_updates,
                    "task_groups_per_update": plan.task_groups_per_update,
                    "rollouts_per_task": plan.rollouts_per_task,
                    "trajectories_per_full_update": plan.trajectories_per_full_update,
                    "action_minibatch_size": plan.action_minibatch_size,
                    "evaluation_kind": plan.evaluation_kind,
                    "num_gpus": args.num_gpus,
                    "resume": args.resume,
                    "dry_run": True,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    from infoskill.training.m0 import run_m0_training

    return run_m0_training(
        config=config,
        plan=plan,
        num_gpus=args.num_gpus,
        run_name=args.run_name,
        resume=args.resume,
    )


def _evaluate(config: AppConfig, args: argparse.Namespace) -> int:
    mode = SkillMode(args.mode)
    _validate_paths(
        config,
        mode=mode,
        require_checkpoint=mode is SkillMode.INFO_SKILL,
    )
    from tqdm.auto import tqdm

    from infoskill.builders import build_transformers_evaluation
    from infoskill.evaluation import EvaluationRunner
    from infoskill.integrations.alfworld import discover_tasks
    from infoskill.persistence import MetricLogger, ZstdJsonlTraceWriter

    run_directory = _run_directory(config, args.run_name or f"eval-{mode.value}")
    logger = _configure_logging(run_directory)
    _write_json(run_directory / "resolved_config.json", config.as_dict())
    tasks = discover_tasks(config.paths.alfworld_data, split="valid_seen")
    if len(tasks) != 140:
        raise RuntimeError(f"valid_seen discovery returned {len(tasks)} tasks instead of 140")
    logger.info("Loading local policy and environment for mode=%s", mode.value)
    collector = build_transformers_evaluation(config, mode=mode)
    progress = tqdm(total=len(tasks), desc=f"valid_seen/{mode.value}", unit="task", dynamic_ncols=True)
    runner = EvaluationRunner(
        collector_factory=lambda: collector,
        config=EvaluationConfig(),
        task_batch_size=config.eval_batch_size,
        master_seed=config.master_seed,
        on_progress=progress.update,
    )
    try:
        run = runner.run(tasks, checkpoint_step=args.checkpoint_step)
    finally:
        progress.close()
    trace_path = ZstdJsonlTraceWriter(run_directory).write_evaluation(
        checkpoint_step=args.checkpoint_step, run=run
    )
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
    }
    values.update({f"success/{key}": value for key, value in summary.per_task_type_success.items()})
    metrics.log(step=args.checkpoint_step, phase="valid_seen", values=values)
    _write_json(run_directory / "valid_seen_summary.json", _summary_payload(run))
    logger.info("Structured trace: %s", trace_path)
    logger.info("Evaluation summary:\n%s", json.dumps(_summary_payload(run), ensure_ascii=False, indent=2))
    return 0 if summary.is_complete else 3


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
        required["semantic_model"] = config.paths.semantic_model
        required["skill_bank"] = config.paths.skill_bank
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
