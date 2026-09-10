from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local docs-only environment
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class PolicyPrefixRecomputeTests(unittest.TestCase):
    def test_replaces_only_prefix_slots_and_detaches_replayed_latent(self) -> None:
        from infoskill.integrations.verl.policy_prefix import (
            recompute_policy_inputs_embeds,
        )

        embedding = torch.nn.Embedding(8, 2)
        with torch.no_grad():
            embedding.weight.copy_(torch.arange(16).reshape(8, 2))
        projector = _Projector()
        input_ids = torch.tensor([[0, 0, 3, 4], [0, 5, 6, 7]])
        attention = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
        prefix_mask = torch.tensor(
            [[True, True, False, False], [False, True, True, False]]
        )
        latent = torch.tensor([[1.0], [2.0]], requires_grad=True)

        inputs_embeds = recompute_policy_inputs_embeds(
            embedding=embedding,
            projector=projector,
            input_ids=input_ids,
            attention_mask=attention,
            prefix_mask=prefix_mask,
            replay_latents=latent,
        )

        self.assertEqual(
            inputs_embeds.detach().tolist(),
            [
                [[1.0, 2.0], [2.0, 4.0], [6.0, 7.0], [8.0, 9.0]],
                [[0.0, 1.0], [2.0, 4.0], [4.0, 8.0], [14.0, 15.0]],
            ],
        )
        inputs_embeds.sum().backward()
        self.assertIsNone(latent.grad)
        self.assertTrue(torch.equal(projector.weight.grad, torch.tensor([9.0, 9.0])))

    def test_can_detach_current_prefix_for_actor_reference_kl(self) -> None:
        from infoskill.integrations.verl.policy_prefix import (
            recompute_policy_inputs_embeds,
        )

        embedding = torch.nn.Embedding(8, 2)
        projector = _Projector()
        latent = torch.tensor([[1.0]], requires_grad=True)
        inputs_embeds = recompute_policy_inputs_embeds(
            embedding=embedding,
            projector=projector,
            input_ids=torch.tensor([[0, 0, 3, 4]]),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            prefix_mask=torch.tensor([[True, True, False, False]]),
            replay_latents=latent,
            detach_projector_output=True,
        )

        inputs_embeds.sum().backward()

        self.assertIsNone(latent.grad)
        self.assertIsNone(projector.weight.grad)
        self.assertIsNotNone(embedding.weight.grad)

    def test_rejects_noncontiguous_or_unattended_prefix_slots(self) -> None:
        from infoskill.integrations.verl.policy_prefix import (
            recompute_policy_inputs_embeds,
        )

        embedding = torch.nn.Embedding(8, 2)
        projector = _Projector()
        latent = torch.tensor([[1.0]])

        with self.assertRaisesRegex(ValueError, "contiguous"):
            recompute_policy_inputs_embeds(
                embedding=embedding,
                projector=projector,
                input_ids=torch.tensor([[0, 3, 0, 4]]),
                attention_mask=torch.ones((1, 4), dtype=torch.long),
                prefix_mask=torch.tensor([[True, False, True, False]]),
                replay_latents=latent,
            )
        with self.assertRaisesRegex(ValueError, "attended"):
            recompute_policy_inputs_embeds(
                embedding=embedding,
                projector=projector,
                input_ids=torch.tensor([[0, 0, 3, 4]]),
                attention_mask=torch.tensor([[0, 1, 1, 1]]),
                prefix_mask=torch.tensor([[True, True, False, False]]),
                replay_latents=latent,
            )


if torch is not None:

    class _Projector(torch.nn.Module):
        prefix_length = 2
        policy_hidden_size = 2

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

        def forward(self, latent):
            row = latent * self.weight
            return torch.stack((row, row * 2), dim=1)


if __name__ == "__main__":
    unittest.main()
