from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from infoskill.app_config import AppConfig
from infoskill.episode import TaskSpec

from .expert_replay import ExpertReplayResult, StrictExpertReplay
from .factory import AlfworldEnvironmentFactory
from .grounding_io import grounding_result_payload
from .handcoded_expert import load_handcoded_expert
from .planner_batch_replay import StrictPlannerBatchReplay
from .planner_expert import PlannerPayloadExpert


_PROGRESS_MARKER = "INFO_SKILL_GROUNDING_PROGRESS"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="bounded INFO-SKILL grounding worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-replay-steps", type=int, required=True)
    parser.add_argument("--persist-horizon", type=int, required=True)
    parser.add_argument(
        "--expert-type",
        choices=("handcoded", "planner"),
        required=True,
    )
    parser.add_argument(
        "--replay-backend",
        choices=("individual", "native_batch"),
        default="individual",
    )
    parser.add_argument("--native-batch-size", type=int, default=4)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = AppConfig.load(args.config)
    factory = AlfworldEnvironmentFactory.from_paths(
        alfworld_source=config.paths.alfworld_source,
        config_path=config.paths.alfworld_config,
        data_root=config.paths.alfworld_data,
        max_steps=args.max_replay_steps,
        expert_type=args.expert_type,
    )
    replay = StrictExpertReplay(
        max_replay_steps=args.max_replay_steps,
        persist_horizon=args.persist_horizon,
    )
    if args.expert_type == "planner":
        expert = PlannerPayloadExpert()
    else:
        expert = load_handcoded_expert(
            alfworld_source=config.paths.alfworld_source,
            max_steps=max(200, args.max_replay_steps),
        )
    input_path = Path(args.input)
    output_path = Path(args.output)
    payloads = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items = tuple(
        (
            TaskSpec(**payload["task"]),
            tuple(payload["candidate_skill_ids"]),
            int(payload["seed"]),
        )
        for payload in payloads
    )
    with output_path.open("w", encoding="utf-8") as destination:
        def emit(task_type: str, result: ExpertReplayResult) -> None:
            destination.write(
                json.dumps(
                    grounding_result_payload(task_type, result),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            destination.flush()
            print(_PROGRESS_MARKER, flush=True)

        if args.replay_backend == "native_batch":
            if args.expert_type != "planner":
                raise ValueError("native batch grounding requires planner expert")
            if args.native_batch_size < 2:
                raise ValueError("native batch size must be at least 2")
            for start in range(0, len(items), args.native_batch_size):
                group = items[start : start + args.native_batch_size]
                if len(group) == 1:
                    task, candidate_skill_ids, seed = group[0]
                    emit(
                        task.task_type,
                        _run_individual(
                            factory=factory,
                            replay=replay,
                            expert=expert,
                            task=task,
                            candidate_skill_ids=candidate_skill_ids,
                            seed=seed,
                        ),
                    )
                    continue
                tasks = tuple(item[0] for item in group)
                candidates = tuple(item[1] for item in group)
                seeds = tuple(item[2] for item in group)
                environment = factory.create_batch(tasks, seeds=seeds)
                batch_results = StrictPlannerBatchReplay(
                    max_replay_steps=args.max_replay_steps,
                    persist_horizon=args.persist_horizon,
                ).run(
                    tasks=tasks,
                    environment=environment,
                    candidate_skill_ids=candidates,
                )
                for task, result in zip(tasks, batch_results):
                    emit(task.task_type, result)
        else:
            for task, candidate_skill_ids, seed in items:
                emit(
                    task.task_type,
                    _run_individual(
                        factory=factory,
                        replay=replay,
                        expert=expert,
                        task=task,
                        candidate_skill_ids=candidate_skill_ids,
                        seed=seed,
                    ),
                )
    return 0


def _run_individual(
    *,
    factory: AlfworldEnvironmentFactory,
    replay: StrictExpertReplay,
    expert: object,
    task: TaskSpec,
    candidate_skill_ids: tuple[str, ...],
    seed: int,
) -> ExpertReplayResult:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        environment = factory.create(task, rollout_id=0, seed=seed)
    except Exception as error:
        return ExpertReplayResult(
            task_id=task.task_id,
            succeeded=False,
            samples=(),
            total_steps=0,
            quarantine_reason=f"factory_exception:{type(error).__name__}",
            exception_stage="factory_create",
            exception_type=type(error).__name__,
            exception_message=" ".join(str(error).split())[:1000],
        )
    return replay.run(
        task=task,
        environment=environment,
        expert=expert,  # type: ignore[arg-type]
        candidate_skill_ids=candidate_skill_ids,
    )


if __name__ == "__main__":
    raise SystemExit(main())
