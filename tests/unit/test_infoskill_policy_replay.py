from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local docs-only environment
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class InfoSkillPolicyReplayTests(unittest.TestCase):
    def test_prefixed_batch_preserves_exact_latent_and_explicit_positions(self) -> None:
        from infoskill.integrations.verl.policy_replay import (
            PolicyReplayExample,
            build_policy_replay_tensors,
        )

        examples = (
            PolicyReplayExample(
                prompt_ids=(10, 11),
                response_ids=(20,),
                response_logprobs=(-0.25,),
                advantage=1.5,
                latent=torch.tensor([0.1, 0.2]),
                rollout_prefix=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            ),
            PolicyReplayExample(
                prompt_ids=(12,),
                response_ids=(21, 22),
                response_logprobs=(-0.5, -0.75),
                advantage=-1.0,
                latent=torch.tensor([0.3, 0.4]),
                rollout_prefix=torch.ones(3, 4),
            ),
        )

        batch = build_policy_replay_tensors(examples, pad_token_id=0)

        self.assertEqual(batch["prompts"].tolist(), [[0, 0, 0, 10, 11], [0, 0, 0, 0, 12]])
        self.assertEqual(
            batch["input_ids"].tolist(),
            [[0, 0, 0, 10, 11, 20, 0], [0, 0, 0, 0, 12, 21, 22]],
        )
        self.assertEqual(
            batch["attention_mask"].tolist(),
            [[1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1]],
        )
        self.assertEqual(
            batch["infoskill_prefix_mask"].tolist(),
            [
                [True, True, True, False, False, False, False],
                [False, True, True, True, False, False, False],
            ],
        )
        self.assertTrue(
            torch.equal(
                batch["infoskill_replay_latents"],
                torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch["infoskill_rollout_prefixes"][0],
                examples[0].rollout_prefix,
            )
        )
        self.assertEqual(
            batch["position_ids"].tolist(),
            [[0, 1, 2, 3, 4, 5, 0], [0, 0, 1, 2, 3, 4, 5]],
        )

    def test_batch_cannot_mix_prefixed_and_token_only_examples(self) -> None:
        from infoskill.integrations.verl.policy_replay import (
            PolicyReplayExample,
            build_policy_replay_tensors,
        )

        examples = (
            PolicyReplayExample((1,), (2,), (-0.1,), 1.0),
            PolicyReplayExample(
                (1,),
                (2,),
                (-0.1,),
                1.0,
                latent=torch.zeros(2),
                rollout_prefix=torch.zeros(3, 4),
            ),
        )

        with self.assertRaisesRegex(ValueError, "mix"):
            build_policy_replay_tensors(examples, pad_token_id=0)

    def test_prefix_geometry_must_be_identical_across_examples(self) -> None:
        from infoskill.integrations.verl.policy_replay import (
            PolicyReplayExample,
            build_policy_replay_tensors,
        )

        examples = (
            PolicyReplayExample((1,), (2,), (-0.1,), 1.0, torch.zeros(2), torch.zeros(3, 4)),
            PolicyReplayExample((1,), (2,), (-0.1,), 1.0, torch.zeros(2), torch.zeros(5, 4)),
        )

        with self.assertRaisesRegex(ValueError, "prefix geometry"):
            build_policy_replay_tensors(examples, pad_token_id=0)


if __name__ == "__main__":
    unittest.main()
