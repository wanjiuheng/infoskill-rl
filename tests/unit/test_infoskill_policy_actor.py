from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local docs-only environment
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class InfoSkillPolicyActorTests(unittest.TestCase):
    def test_embedding_injection_preserves_projector_gradient(self) -> None:
        from infoskill.integrations.verl.policy_actor import (
            PrefixEmbeddingInjector,
            _current_prefix_and_hook_mask,
        )

        embedding = torch.nn.Embedding(8, 2)
        projector = _Projector()
        injector = PrefixEmbeddingInjector(embedding)
        prefix, mask = _current_prefix_and_hook_mask(
            projector=projector,
            replay_latents=torch.tensor([[2.0]], requires_grad=True),
            prefix_mask=torch.tensor([[True, True, False]]),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            remove_padding=False,
            detach_projector_output=False,
        )

        with injector.inject(prefix=prefix, prefix_mask=mask):
            output = embedding(torch.tensor([[0, 0, 3]]))
        output.sum().backward()

        self.assertTrue(torch.equal(projector.weight.grad, torch.tensor([6.0, 6.0])))
        self.assertEqual(embedding.weight.grad[0].abs().sum().item(), 0.0)
        self.assertGreater(embedding.weight.grad[3].abs().sum().item(), 0.0)

    def test_remove_padding_mask_follows_attended_row_major_order(self) -> None:
        from infoskill.integrations.verl.policy_actor import (
            _current_prefix_and_hook_mask,
        )

        prefix, mask = _current_prefix_and_hook_mask(
            projector=_Projector(),
            replay_latents=torch.tensor([[1.0], [2.0]]),
            prefix_mask=torch.tensor(
                [[False, True, True, False], [True, True, False, False]]
            ),
            attention_mask=torch.tensor(
                [[False, True, True, True], [True, True, True, False]]
            ),
            remove_padding=True,
            detach_projector_output=True,
        )

        self.assertEqual(tuple(prefix.shape), (2, 2, 2))
        self.assertFalse(prefix.requires_grad)
        self.assertEqual(
            mask.tolist(),
            [[True, True, False, True, True, False]],
        )

    def test_injector_clears_state_after_failed_model_forward(self) -> None:
        from infoskill.integrations.verl.policy_actor import PrefixEmbeddingInjector

        embedding = torch.nn.Embedding(4, 2)
        injector = PrefixEmbeddingInjector(embedding)
        with self.assertRaisesRegex(RuntimeError, "unexpected sequence layout"):
            with injector.inject(
                prefix=torch.ones((1, 2, 2)),
                prefix_mask=torch.ones((1, 3), dtype=torch.bool),
            ):
                embedding(torch.tensor([[0, 1]]))

        output = embedding(torch.tensor([[0, 1]]))
        self.assertEqual(tuple(output.shape), (1, 2, 2))


if torch is not None:

    class _Projector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

        def forward(self, latent):
            row = latent * self.weight
            return torch.stack((row, row * 2), dim=1)


if __name__ == "__main__":
    unittest.main()
