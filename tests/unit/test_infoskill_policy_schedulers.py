from __future__ import annotations

import unittest


class InfoSkillPolicySchedulerTests(unittest.TestCase):
    def test_steps_actor_and_projector_together_only_after_policy_step(self) -> None:
        try:
            from infoskill.integrations.verl.worker import (
                _CoordinatedPolicySchedulers,
            )
        except ModuleNotFoundError as error:
            if error.name in {"torch", "verl", "peft", "safetensors"}:
                self.skipTest(f"server dependency is unavailable: {error.name}")
            raise

        actor = _Actor()
        scheduler = _CoordinatedPolicySchedulers(actor)

        scheduler.step()
        self.assertEqual(actor.infoskill_actor_scheduler.steps, 0)
        self.assertEqual(actor.infoskill_projector_scheduler.steps, 0)

        actor.infoskill_update_applied = True
        scheduler.step()
        self.assertEqual(actor.infoskill_actor_scheduler.steps, 1)
        self.assertEqual(actor.infoskill_projector_scheduler.steps, 1)
        self.assertEqual(scheduler.get_last_lr(), [0.25])


class _Scheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def get_last_lr(self):
        return [0.25]


class _Actor:
    def __init__(self) -> None:
        self.infoskill_actor_scheduler = _Scheduler()
        self.infoskill_projector_scheduler = _Scheduler()
        self.infoskill_update_applied = False


if __name__ == "__main__":
    unittest.main()
