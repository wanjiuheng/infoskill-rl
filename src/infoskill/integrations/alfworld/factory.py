from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Mapping

from infoskill.episode import TaskSpec

from .environment import AlfworldEnvironment


def _configured_copy(base_config: Mapping[str, object], *, data_root: Path, max_steps: int) -> dict:
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
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._data_root = Path(data_root).expanduser().resolve()
        self._max_steps = max_steps
        self._base_config = dict(base_config)
        self._environment_class = environment_class

    @classmethod
    def from_paths(
        cls,
        *,
        alfworld_source: str | Path,
        config_path: str | Path,
        data_root: str | Path,
        max_steps: int,
    ) -> "AlfworldEnvironmentFactory":
        source = str(Path(alfworld_source).expanduser().resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        try:
            import yaml
            from alfworld.agents.environment import get_environment
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
        )
        definition.config["general"]["random_seed"] = seed
        definition.train_eval = train_eval
        definition.game_files = [task.environment_path]
        definition.num_games = 1
        raw_environment = definition.init_env(batch_size=1)
        if hasattr(raw_environment, "seed"):
            raw_environment.seed(seed)
        return AlfworldEnvironment(raw_environment, task=task)
