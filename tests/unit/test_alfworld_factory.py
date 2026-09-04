from __future__ import annotations

import unittest

from infoskill.episode import TaskSpec
from infoskill.integrations.alfworld import AlfworldEnvironmentFactory

from .test_alfworld_environment import _RawBatchSizeOneEnvironment


class _PinnedAlfredTWEnv:
    last_definition = None

    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("factory must not scan the full split through AlfredTWEnv.__init__")

    def init_env(self, batch_size: int):
        self.__class__.last_definition = self
        assert batch_size == 1
        raw = _RawBatchSizeOneEnvironment()
        raw.seed_value = None

        def seed(value: int) -> None:
            raw.seed_value = value

        raw.seed = seed
        self.raw = raw
        return raw


class AlfworldEnvironmentFactoryTests(unittest.TestCase):
    def test_factory_targets_one_game_without_scanning_the_split(self) -> None:
        base_config = {
            "dataset": {},
            "logic": {},
            "env": {"type": "AlfredTWEnv", "domain_randomization": True},
            "general": {"use_cuda": True},
            "dagger": {"training": {}},
        }
        factory = AlfworldEnvironmentFactory(
            data_root="/data",
            max_steps=30,
            base_config=base_config,
            environment_class=_PinnedAlfredTWEnv,
        )
        task = TaskSpec(
            task_id="game-1",
            split="train",
            task_type="pick_and_place_simple",
            goal="put an apple in the fridge",
            environment_path="/data/game.tw-pddl",
        )

        environment = factory.create(task, rollout_id=3, seed=99)
        definition = _PinnedAlfredTWEnv.last_definition

        self.assertIsNotNone(environment)
        self.assertEqual(definition.game_files, ["/data/game.tw-pddl"])
        self.assertEqual(definition.train_eval, "train")
        self.assertEqual(definition.config["dagger"]["training"]["max_nb_steps_per_episode"], 30)
        self.assertFalse(definition.config["env"]["domain_randomization"])
        self.assertEqual(definition.raw.seed_value, 99)


if __name__ == "__main__":
    unittest.main()
