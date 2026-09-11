from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep

from infoskill.episode import TaskSpec
from infoskill.integrations.alfworld import AlfworldEnvironmentFactory
from infoskill.integrations.alfworld.parser_guard import (
    install_textworld_parser_guard,
    install_textworld_parser_guards,
)

from .test_alfworld_environment import _RawBatchSizeOneEnvironment


class _PinnedAlfredTWEnv:
    last_definition = None

    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("factory must not scan the full split through AlfredTWEnv.__init__")

    def init_env(self, batch_size: int):
        self.__class__.last_definition = self
        assert batch_size >= 1
        if batch_size > 1:
            from .test_alfworld_batch_environment import _RawBatchEnvironment

            raw = _RawBatchEnvironment(tuple(self.game_files))
            self.raw = raw
            return raw
        raw = _RawBatchSizeOneEnvironment()
        raw.seed_value = None

        def seed(value: int) -> None:
            raw.seed_value = value

        raw.seed = seed
        self.raw = raw
        return raw


class AlfworldEnvironmentFactoryTests(unittest.TestCase):
    def test_textworld_parser_guard_serializes_shared_parser_calls(self) -> None:
        class FakeTextgenModule:
            pass

        module = FakeTextgenModule()
        active_lock = Lock()
        start_barrier = Barrier(2)
        active_calls = 0

        def unsafe_parse(value: str) -> str:
            nonlocal active_calls
            with active_lock:
                active_calls += 1
                overlap = active_calls > 1
            try:
                sleep(0.02)
                if overlap:
                    raise RuntimeError("shared parser call overlapped")
                return value
            finally:
                with active_lock:
                    active_calls -= 1

        module._parse_and_convert = unsafe_parse
        self.assertTrue(install_textworld_parser_guard(module))
        guarded_parse = module._parse_and_convert
        self.assertFalse(install_textworld_parser_guard(module))
        self.assertIs(module._parse_and_convert, guarded_parse)

        def parse_together(value: str) -> str:
            start_barrier.wait(timeout=1.0)
            return module._parse_and_convert(value)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(parse_together, ("a", "b")))

        self.assertEqual(results, ("a", "b"))

    def test_textworld_parser_guards_cover_textgen_and_logic_modules(self) -> None:
        class FakeParserModule:
            pass

        active_lock = Lock()
        start_barrier = Barrier(2)
        active_calls = 0

        def module_with_parser(name: str) -> object:
            module = FakeParserModule()

            def unsafe_parse(value: str) -> str:
                nonlocal active_calls
                with active_lock:
                    active_calls += 1
                    overlap = active_calls > 1
                try:
                    sleep(0.02)
                    if overlap:
                        raise RuntimeError("TextWorld parser modules overlapped")
                    return f"{name}:{value}"
                finally:
                    with active_lock:
                        active_calls -= 1

            module._parse_and_convert = unsafe_parse
            return module

        textgen = module_with_parser("textgen")
        logic = module_with_parser("logic")
        self.assertEqual(
            install_textworld_parser_guards(textgen=textgen, logic=logic),
            (True, True),
        )

        def parse_together(item: tuple[object, str]) -> str:
            module, value = item
            start_barrier.wait(timeout=1.0)
            return module._parse_and_convert(value)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(
                executor.map(parse_together, ((textgen, "a"), (logic, "b")))
            )

        self.assertEqual(results, ("textgen:a", "logic:b"))

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

    def test_factory_builds_one_native_batch_with_exact_slot_order(self) -> None:
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
        tasks = tuple(
            TaskSpec(
                task_id=f"game-{index}",
                split="train",
                task_type="pick_and_place_simple",
                goal="goal",
                environment_path=f"/data/game-{index}.tw-pddl",
            )
            for index in range(2)
        )

        batch = factory.create_batch(tasks, seeds=(10, 11))
        states = batch.reset()
        definition = _PinnedAlfredTWEnv.last_definition

        self.assertEqual(definition.game_files, [task.environment_path for task in tasks])
        self.assertEqual(definition.raw.loaded_gamefiles, definition.game_files)
        self.assertEqual([state.task_id for state in states], ["game-0", "game-1"])

    def test_factory_can_request_planner_expert_without_mutating_base_config(self) -> None:
        base_config = {
            "dataset": {},
            "logic": {},
            "env": {"type": "AlfredTWEnv", "expert_type": "handcoded"},
            "general": {},
            "dagger": {"training": {}},
        }
        factory = AlfworldEnvironmentFactory(
            data_root="/data",
            max_steps=30,
            base_config=base_config,
            environment_class=_PinnedAlfredTWEnv,
            expert_type="planner",
        )
        task = TaskSpec(
            task_id="game-1",
            split="train",
            task_type="pick_and_place_simple",
            goal="put an apple in the fridge",
            environment_path="/data/game.tw-pddl",
        )

        factory.create(task, rollout_id=0, seed=7)
        definition = _PinnedAlfredTWEnv.last_definition

        self.assertEqual(definition.config["env"]["expert_type"], "planner")
        self.assertEqual(base_config["env"]["expert_type"], "handcoded")


if __name__ == "__main__":
    unittest.main()
