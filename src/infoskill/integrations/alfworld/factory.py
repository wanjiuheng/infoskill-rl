from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Mapping

from infoskill.episode import TaskSpec

from .batch_environment import AlfworldEnvironmentBatch
from .environment import AlfworldEnvironment
from .expert_type_guard import prepare_alfworld_expert_type_binding
from .parser_guard import install_textworld_parser_guards


def _configured_copy(
    base_config: Mapping[str, object],
    *,
    data_root: Path,
    max_steps: int,
    expert_type: str | None = None,
) -> dict:
    config = copy.deepcopy(dict(base_config))
    dataset = config.setdefault("dataset", {})
    logic = config.setdefault("logic", {})
    env = config.setdefault("env", {})
    general = config.setdefault("general", {})
    dagger = config.setdefault("dagger", {}).setdefault("training", {})
    rl = config.setdefault("rl", {}).setdefault("training", {})
    if not all(isinstance(section, dict) for section in (dataset, logic, env, general, dagger, rl)):
        raise ValueError("ALFWorld config sections must be mappings")

    dataset["data_path"] = str(data_root / "json_2.1.1" / "train")
    dataset["eval_id_data_path"] = str(data_root / "json_2.1.1" / "valid_seen")
    dataset["eval_ood_data_path"] = str(data_root / "json_2.1.1" / "valid_unseen")
    dataset["num_train_games"] = -1
    dataset["num_eval_games"] = -1
    logic["domain"] = str(data_root / "logic" / "alfred.pddl")
    logic["grammar"] = str(data_root / "logic" / "alfred.twl2")
    env["type"] = "AlfredTWEnv"
    env["domain_randomization"] = False
    if expert_type is not None:
        env["expert_type"] = expert_type
    general["use_cuda"] = False
    dagger["max_nb_steps_per_episode"] = max_steps
    rl["max_nb_steps_per_episode"] = max_steps
    return config


class AlfworldEnvironmentFactory:
    """Create one targeted TextWorld game without scanning an entire split."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        max_steps: int,
        base_config: Mapping[str, object],
        environment_class: type,
        expert_type: str | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if expert_type not in {None, "handcoded", "planner"}:
            raise ValueError("expert_type must be handcoded, planner, or None")
        self._data_root = Path(data_root).expanduser().resolve()
        self._max_steps = max_steps
        self._base_config = dict(base_config)
        self._environment_class = environment_class
        self._expert_type = expert_type

    @classmethod
    def from_paths(
        cls,
        *,
        alfworld_source: str | Path,
        config_path: str | Path,
        data_root: str | Path,
        max_steps: int,
        expert_type: str | None = None,
    ) -> "AlfworldEnvironmentFactory":
        source = str(Path(alfworld_source).expanduser().resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        try:
            import yaml
            if expert_type is not None:
                # This probe runs in every short-lived grounding worker.  It
                # therefore guards the process that actually constructs the
                # TextWorld wrapper, not merely its parent coordinator.
                prepare_alfworld_expert_type_binding(
                    source,
                    requested_expert_type=expert_type,
                )
            from alfworld.agents.environment import get_environment
            install_textworld_parser_guards()
        except ImportError as error:
            raise RuntimeError(
                "ALFWorld runtime dependencies are missing; install the locked server environment"
            ) from error
        path = Path(config_path).expanduser().resolve()
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"ALFWorld config root must be a mapping: {path}")
        environment_class = get_environment("AlfredTWEnv")
        return cls(
            data_root=data_root,
            max_steps=max_steps,
            base_config=payload,
            environment_class=environment_class,
            expert_type=expert_type,
        )

    def create(self, task: TaskSpec, *, rollout_id: int, seed: int) -> AlfworldEnvironment:
        del rollout_id  # The semantic seed and task identity already distinguish the environment instance.
        if task.environment_path is None:
            raise ValueError("ALFWorld TaskSpec requires environment_path")
        split_to_mode = {
            "train": "train",
            "valid_seen": "eval_in_distribution",
            "valid_unseen": "eval_out_of_distribution",
        }
        try:
            train_eval = split_to_mode[task.split]
        except KeyError as error:
            raise ValueError(f"unsupported ALFWorld split: {task.split}") from error

        definition = self._environment_class.__new__(self._environment_class)
        definition.config = _configured_copy(
            self._base_config,
            data_root=self._data_root,
            max_steps=self._max_steps,
            expert_type=self._expert_type,
        )
        definition.config["general"]["random_seed"] = seed
        definition.train_eval = train_eval
        definition.game_files = [task.environment_path]
        definition.num_games = 1
        raw_environment = definition.init_env(batch_size=1)
        if hasattr(raw_environment, "seed"):
            raw_environment.seed(seed)
        return AlfworldEnvironment(raw_environment, task=task)

    def create_batch(
        self,
        tasks: tuple[TaskSpec, ...],
        *,
        seeds: tuple[int, ...],
    ) -> AlfworldEnvironmentBatch:
        """Create one TextWorld AsyncBatchEnv with a stable task per process slot."""

        if len(tasks) < 2:
            raise ValueError("native ALFWorld batching requires at least two tasks")
        if len(seeds) != len(tasks):
            raise ValueError("one semantic seed is required for every ALFWorld batch slot")
        if any(task.environment_path is None for task in tasks):
            raise ValueError("every ALFWorld TaskSpec requires environment_path")
        splits = {task.split for task in tasks}
        if len(splits) != 1:
            raise ValueError("one ALFWorld native batch cannot mix data splits")
        split_to_mode = {
            "train": "train",
            "valid_seen": "eval_in_distribution",
            "valid_unseen": "eval_out_of_distribution",
        }
        try:
            train_eval = split_to_mode[next(iter(splits))]
        except KeyError as error:
            raise ValueError(f"unsupported ALFWorld split: {next(iter(splits))}") from error

        definition = self._environment_class.__new__(self._environment_class)
        definition.config = _configured_copy(
            self._base_config,
            data_root=self._data_root,
            max_steps=self._max_steps,
            expert_type=self._expert_type,
        )
        # Kept for provenance parity with the single-instance factory. With domain
        # randomization disabled, task identity and transition semantics are fixed by
        # the explicit game paths below rather than a shared mutable RNG.
        definition.config["general"]["random_seed"] = seeds[0]
        definition.train_eval = train_eval
        definition.game_files = [str(task.environment_path) for task in tasks]
        definition.num_games = len(tasks)
        raw_environment = definition.init_env(batch_size=len(tasks))
        return AlfworldEnvironmentBatch(raw_environment, tasks=tasks)
