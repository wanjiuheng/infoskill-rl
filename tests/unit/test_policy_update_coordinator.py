from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local docs-only environment
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class PolicyUpdateCoordinatorTests(unittest.TestCase):
    def test_combines_norm_clips_once_and_steps_both_optimizers(self) -> None:
        from infoskill.learning import PolicyUpdateCoordinator

        actor = torch.nn.Parameter(torch.tensor([10.0]))
        projector = torch.nn.Parameter(torch.tensor([20.0]))
        actor.grad = torch.tensor([3.0])
        projector.grad = torch.tensor([4.0])
        actor_optimizer = torch.optim.SGD([actor], lr=1.0)
        projector_optimizer = torch.optim.SGD([projector], lr=1.0)
        actor_scheduler = torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lambda _: 1.0)
        projector_scheduler = torch.optim.lr_scheduler.LambdaLR(
            projector_optimizer, lambda _: 1.0
        )
        coordinator = PolicyUpdateCoordinator(
            actor_parameters=(actor,),
            projector_parameters=(projector,),
            actor_optimizer=actor_optimizer,
            projector_optimizer=projector_optimizer,
            actor_scheduler=actor_scheduler,
            projector_scheduler=projector_scheduler,
            max_grad_norm=1.0,
        )

        metrics = coordinator.step(actor_global_grad_norm=torch.tensor(3.0))

        self.assertAlmostEqual(actor.item(), 9.4, places=6)
        self.assertAlmostEqual(projector.item(), 19.2, delta=1e-6)
        self.assertAlmostEqual(metrics["policy/combined_grad_norm_before_clip"], 5.0)
        self.assertAlmostEqual(metrics["policy/clip_coefficient"], 0.2, places=6)
        self.assertEqual(metrics["policy/optimizer_step_applied"], 1.0)
        self.assertEqual(actor_scheduler.last_epoch, 1)
        self.assertEqual(projector_scheduler.last_epoch, 1)

    def test_nonfinite_side_skips_both_optimizers_and_schedulers(self) -> None:
        from infoskill.learning import PolicyUpdateCoordinator

        actor = torch.nn.Parameter(torch.tensor([10.0]))
        projector = torch.nn.Parameter(torch.tensor([20.0]))
        actor.grad = torch.tensor([1.0])
        projector.grad = torch.tensor([float("nan")])
        actor_optimizer = torch.optim.SGD([actor], lr=1.0)
        projector_optimizer = torch.optim.SGD([projector], lr=1.0)
        actor_scheduler = torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lambda _: 1.0)
        projector_scheduler = torch.optim.lr_scheduler.LambdaLR(
            projector_optimizer, lambda _: 1.0
        )
        coordinator = PolicyUpdateCoordinator(
            actor_parameters=(actor,),
            projector_parameters=(projector,),
            actor_optimizer=actor_optimizer,
            projector_optimizer=projector_optimizer,
            actor_scheduler=actor_scheduler,
            projector_scheduler=projector_scheduler,
        )

        metrics = coordinator.step(actor_global_grad_norm=torch.tensor(1.0))

        self.assertEqual(actor.item(), 10.0)
        self.assertEqual(projector.item(), 20.0)
        self.assertEqual(metrics["policy/optimizer_step_applied"], 0.0)
        self.assertEqual(metrics["policy/optimizer_skip_nonfinite"], 1.0)
        self.assertEqual(actor_scheduler.last_epoch, 0)
        self.assertEqual(projector_scheduler.last_epoch, 0)


if __name__ == "__main__":
    unittest.main()
