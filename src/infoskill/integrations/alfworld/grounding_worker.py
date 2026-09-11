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
        default="handcoded",
    )
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
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            payload = json.loads(line)
            task = TaskSpec(**payload["task"])
            candidate_skill_ids = tuple(payload["candidate_skill_ids"])
            seed = int(payload["seed"])
            # A task-owned RNG stream makes output independent of process and
            # shard boundaries, matching the paired-randomness protocol.
            random.seed(seed)
            np.random.seed(seed % (2**32))
            try:
                environment = factory.create(task, rollout_id=0, seed=seed)
            except Exception as error:
                result = ExpertReplayResult(
                    task_id=task.task_id,
                    succeeded=False,
                    samples=(),
                    total_steps=0,
                    quarantine_reason=f"factory_exception:{type(error).__name__}",
                    exception_stage="factory_create",
                    exception_type=type(error).__name__,
                    exception_message=" ".join(str(error).split())[:1000],
                )
            else:
                result = replay.run(
                    task=task,
                    environment=environment,
                    expert=expert,  # type: ignore[arg-type]
                    candidate_skill_ids=candidate_skill_ids,
                )
            destination.write(
                json.dumps(
                    grounding_result_payload(task.task_type, result),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            destination.flush()
            print(_PROGRESS_MARKER, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
