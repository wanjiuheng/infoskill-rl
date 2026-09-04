from __future__ import annotations

import json
import logging
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from infoskill.app_config import AppConfig
from infoskill.conditioning import NoSkillConditioner
from infoskill.config import EvaluationConfig, TaskDenominator
from infoskill.episode import TaskSpec, TrajectoryCollector
from infoskill.evaluation import (
    EvaluationCheckpointScore,
    EvaluationRunner,
    select_best_valid,
)
from infoskill.integrations.alfworld import (
    AlfworldEnvironmentFactory,
    build_train_monitor_manifest,
    discover_tasks,
    write_train_monitor_manifest,
)
from infoskill.learning import group_relative_advantages
from infoskill.persistence import (
    CheckpointManager,
    MetricLogger,
    TrainerCheckpointState,
    ZstdJsonlTraceWriter,
)
from infoskill.rollout import GenerationParameters

from .plan import TrainingPlan, TrainingProfile
from .schedule import TaskSchedule
from .trainer import InfoSkillTrainer, UpdateMetrics


EXPECTED_TRAIN_TASKS = 3_553
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def run_m0_training(
    *,
    config: AppConfig,
    plan: TrainingPlan,
    num_gpus: int,
    run_name: str | None,
    resume: str | None,
) -> int:
    """Run the token-only M0 vertical slice through the pinned VERL runtime."""

    if config.paths.policy_adapter is not None:
        raise ValueError("formal M0 must start from the unmodified base policy")
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")

    all_train_tasks = discover_tasks(config.paths.alfworld_data, split="train")
    if len(all_train_tasks) != EXPECTED_TRAIN_TASKS:
        raise RuntimeError(
            f"train discovery returned {len(all_train_tasks)} tasks instead of "
            f"{EXPECTED_TRAIN_TASKS}"
        )
    monitor = build_train_monitor_manifest(
        all_train_tasks, master_seed=config.master_seed
    )
    monitor_ids = set(monitor.monitor_task_ids)
    scheduled_tasks = (
        all_train_tasks
        if plan.include_monitor_tasks
        else tuple(task for task in all_train_tasks if task.task_id not in monitor_ids)
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

    run_directory, checkpoint_to_load = _resolve_run_directory(
        output_root=config.paths.output_root,
        profile=plan.profile,
        run_name=run_name,
        resume=resume,
    )
    logger = _configure_training_logging(run_directory)
    resolved = {
        "schema_version": 1,
        "mode": "no_skill",
        "num_gpus": num_gpus,
        "app_config": config.as_dict(),
        "training_plan": _plan_payload(plan),
    }
    if checkpoint_to_load is None:
        _write_json(run_directory / "resolved_config.json", resolved)
        write_train_monitor_manifest(
            run_directory / "train_monitor_manifest.json", monitor
        )
    else:
        _validate_resume_config(run_directory, resolved)

    schedule = TaskSchedule(
        scheduled_tasks,
        master_seed=config.master_seed,
        passes=1,
    )
    checkpoints = CheckpointManager(
        run_directory / "checkpoints",
        keep_recent=2,
    )
    restored = (
        checkpoints.load_trainer_state(checkpoint_to_load)
        if checkpoint_to_load is not None
        else None
    )
    if restored is not None:
        schedule.restore(restored.schedule)

    provenance = {
        "schema_version": 1,
        "mode": "no_skill",
        "train_task_count": len(all_train_tasks),
        "scheduled_task_count": len(scheduled_tasks),
        "train_task_manifest_sha256": monitor.source_manifest_sha256,
        "skillrl_expected_commit": "8e66726ed866a4e0a7f053586a41022798192e6c",
    }
    _write_json(run_directory / "provenance.json", provenance)

    from infoskill.integrations.verl import VerlRuntime, VerlRuntimeConfig

    if VerlRuntime is None or VerlRuntimeConfig is None:
        raise RuntimeError("the pinned VERL runtime is unavailable")
    runtime = VerlRuntime.start(
        VerlRuntimeConfig(
            skillrl_source=config.paths.skillrl_source,
            model_path=config.paths.policy_model,
            num_gpus=num_gpus,
            num_cpus=96,
            max_prompt_tokens=config.max_prompt_tokens,
            max_response_tokens=config.max_response_tokens,
            total_training_steps=plan.max_updates,
            action_minibatch_size=plan.action_minibatch_size,
            gpu_memory_utilization=0.45,
            require_hybrid_prefix=False,
            master_seed=config.master_seed,
        )
    )
    try:
        factory = AlfworldEnvironmentFactory.from_paths(
            alfworld_source=config.paths.alfworld_source,
            config_path=config.paths.alfworld_config,
            data_root=config.paths.alfworld_data,
            max_steps=config.max_steps,
        )
        training_collector = _collector(
            config, factory=factory, runtime=runtime, training=True
        )
        evaluation_collector = _collector(
            config, factory=factory, runtime=runtime, training=False
        )
        if checkpoint_to_load is not None:
            runtime.load_portable_state(checkpoint_to_load / "runtime")

        traces = ZstdJsonlTraceWriter(run_directory)
        metrics = MetricLogger(run_directory)
        from tqdm.auto import tqdm

        initial_update = restored.global_update if restored is not None else 0
        progress = tqdm(
            total=plan.max_updates,
            initial=initial_update,
            desc=f"M0/{plan.profile.value}",
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
                    update == 0
                    or update == plan.max_updates
                    or update % 25 == 0
                ),
            )
            logger.info("Committed checkpoint: %s", path)

        def update_callback(
            update: UpdateMetrics,
            groups: tuple,
        ) -> None:
            _require_finite_metrics(update.values)
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
            values: dict[str, float | int | str | bool | None] = dict(
                update.values
            )
            values["schedule/cursor"] = schedule.cursor
            values["schedule/total"] = schedule.total
            values["trace"] = str(trace_path)
            metrics.log(step=update.global_update, phase="train", values=values)
            logger.info(
                "update=%d/%d success=%.4f reward=%.4f invalid=%.4f",
                update.global_update,
                plan.max_updates,
                update.values.get("rollout/success_rate", float("nan")),
                update.values.get("rollout/mean_reward", float("nan")),
                update.values.get("rollout/invalid_action_rate", float("nan")),
            )
            progress.update(1)

        evaluate = _evaluation_callback(
            config=config,
            plan=plan,
            all_train_tasks=all_train_tasks,
            monitor_task_ids=monitor_ids,
            collector=evaluation_collector,
            run_directory=run_directory,
            traces=traces,
            metrics=metrics,
            logger=logger,
            checkpoint=checkpoint,
            schedule=schedule,
        )
        trainer = InfoSkillTrainer(
            collector=training_collector,
            runtime=runtime,
            schedule=schedule,
            task_groups_per_update=plan.task_groups_per_update,
            rollouts_per_task=plan.rollouts_per_task,
            master_seed=config.master_seed,
            auxiliary_enabled=False,
            on_update=update_callback,
            on_evaluate=evaluate,
            on_checkpoint=checkpoint,
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
            "status": "complete",
            "mode": "no_skill",
            "profile": plan.profile.value,
            "global_update": trainer.global_update,
            "task_cursor": schedule.cursor,
            "max_updates": plan.max_updates,
        }
        _write_json(run_directory / "training_summary.json", summary)
        logger.info("M0 training segment complete: %s", json.dumps(summary))
        return 0
    except Exception:
        logger.exception("M0 training failed")
        raise
    finally:
        runtime.close()


def _collector(
    config: AppConfig,
    *,
    factory: AlfworldEnvironmentFactory,
    runtime: object,
    training: bool,
) -> TrajectoryCollector:
    parameters = GenerationParameters(
        do_sample=training,
        temperature=1.0 if training else 0.0,
        top_p=1.0,
        max_new_tokens=config.max_response_tokens,
    )
    return TrajectoryCollector(
        environment_factory=factory,
        conditioner=NoSkillConditioner(),
        rollout_backend=runtime,  # type: ignore[arg-type]
        max_steps=config.max_steps,
        history_limit=config.history_length,
        invalid_action_penalty=0.01,
        generation_parameters=parameters,
    )


def _evaluation_callback(
    *,
    config: AppConfig,
    plan: TrainingPlan,
    all_train_tasks: Sequence[TaskSpec],
    monitor_task_ids: set[str],
    collector: TrajectoryCollector,
    run_directory: Path,
    traces: ZstdJsonlTraceWriter,
    metrics: MetricLogger,
    logger: logging.Logger,
    checkpoint,
    schedule: TaskSchedule,
):
    if plan.evaluation_kind == "none":
        return None
    if plan.evaluation_kind == "valid_seen":
        tasks = discover_tasks(config.paths.alfworld_data, split="valid_seen")
        evaluation_config = EvaluationConfig()
        phase = "valid_seen"
    elif plan.evaluation_kind == "train_monitor":
        tasks = tuple(
            task for task in all_train_tasks if task.task_id in monitor_task_ids
        )
        counts = Counter(task.task_type for task in tasks)
        evaluation_config = EvaluationConfig(
            split="train_monitor",
            denominators=tuple(
                TaskDenominator(task_type, count)
                for task_type, count in sorted(counts.items())
            ),
        )
        phase = "train_monitor"
    else:
        raise ValueError(f"unsupported evaluation kind: {plan.evaluation_kind}")
    valid_scores = (
        _load_valid_scores(run_directory)
        if phase == "valid_seen"
        else []
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
        if phase == "valid_seen":
            assert summary.macro_success is not None
            assert summary.overall_success is not None
            assert summary.invalid_action_rate is not None
            valid_scores.append(
                EvaluationCheckpointScore(
                    step=global_update,
                    macro_success=summary.macro_success,
                    overall_success=summary.overall_success,
                    invalid_action_rate=summary.invalid_action_rate,
                )
            )
            best = select_best_valid(valid_scores)
            _write_json(
                run_directory / "checkpoint_selection.json",
                {
                    "schema_version": 1,
                    "disclosure": "validation-selected performance on valid_seen",
                    "rule": [
                        "max_macro_success",
                        "max_overall_success",
                        "min_invalid_action_rate",
                        "earliest_update",
                    ],
                    "evaluations": [
                        _checkpoint_score_payload(score) for score in valid_scores
                    ],
                    "last": _checkpoint_score_payload(valid_scores[-1]),
                    "best_valid": _checkpoint_score_payload(best),
                },
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


def _checkpoint_score_payload(
    score: EvaluationCheckpointScore,
) -> dict[str, float | int | str]:
    return {
        "step": score.step,
        "macro_success": score.macro_success,
        "overall_success": score.overall_success,
        "invalid_action_rate": score.invalid_action_rate,
        "checkpoint": f"checkpoints/step-{score.step:06d}",
    }


def _load_valid_scores(run_directory: Path) -> list[EvaluationCheckpointScore]:
    path = run_directory / "checkpoint_selection.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("evaluations")
    if not isinstance(records, list):
        raise RuntimeError(f"invalid checkpoint selection history: {path}")
    scores: list[EvaluationCheckpointScore] = []
    seen_steps: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError(f"invalid checkpoint selection record: {path}")
        score = EvaluationCheckpointScore(
            step=int(record["step"]),
            macro_success=float(record["macro_success"]),
            overall_success=float(record["overall_success"]),
            invalid_action_rate=float(record["invalid_action_rate"]),
        )
        if score.step in seen_steps:
            raise RuntimeError(f"duplicate checkpoint selection step: {score.step}")
        seen_steps.add(score.step)
        scores.append(score)
    return scores


def _resolve_run_directory(
    *,
    output_root: str,
    profile: TrainingProfile,
    run_name: str | None,
    resume: str | None,
) -> tuple[Path, Path | None]:
    if resume is not None:
        checkpoint = Path(resume).expanduser().resolve()
        if checkpoint.parent.name != "checkpoints":
            raise ValueError("resume path must be a run checkpoints/step-* directory")
        return checkpoint.parent.parent, checkpoint
    name = run_name or f"m0-{profile.value}"
    if not _RUN_NAME.fullmatch(name):
        raise ValueError("run_name must contain only letters, digits, '.', '_' or '-'")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output_root).expanduser().resolve() / f"{stamp}-{name}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory, None


def _validate_resume_config(
    run_directory: Path, current: Mapping[str, object]
) -> None:
    path = run_directory / "resolved_config.json"
    if not path.is_file():
        raise RuntimeError(f"resume run has no resolved config: {path}")
    previous = json.loads(path.read_text(encoding="utf-8"))
    previous_without_gpus = dict(previous)
    current_without_gpus = dict(current)
    previous_without_gpus.pop("num_gpus", None)
    current_without_gpus.pop("num_gpus", None)
    if previous_without_gpus != current_without_gpus:
        raise RuntimeError("resume configuration differs from the original run")


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
        "include_monitor_tasks": plan.include_monitor_tasks,
    }


def _configure_training_logging(run_directory: Path) -> logging.Logger:
    logger = logging.getLogger("infoskill.training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
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
