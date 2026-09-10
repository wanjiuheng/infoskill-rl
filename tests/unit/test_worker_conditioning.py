from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local docs-only environment
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class InfoSkillWorkerConditionerTests(unittest.TestCase):
    def test_seeded_worker_conditioning_returns_exact_cpu_replay(self) -> None:
        from infoskill.conditioning import InfoSkillConditioningWorkItem
        from infoskill.integrations.verl.worker_conditioning import (
            InfoSkillWorkerConditioner,
        )

        compressor = _Compressor()
        projector = _Projector()
        engine = InfoSkillWorkerConditioner(
            library=_Library(),
            semantic_encoder=_SemanticEncoder(),
            feature_cache=_FeatureCache(),
            compressor=compressor,
            projector=projector,
        )
        items = (
            InfoSkillConditioningWorkItem(
                compression_view="state one",
                candidate_skill_ids=("skill-1",),
                latent_seed=13,
                latent_mode="sample",
            ),
            InfoSkillConditioningWorkItem(
                compression_view="state two",
                candidate_skill_ids=("skill-1",),
                latent_seed=17,
                latent_mode="sample",
            ),
        )

        first = engine.condition(items)
        second = engine.condition(items)

        self.assertTrue(
            torch.equal(
                first[0].replay_trace.epsilon,
                second[0].replay_trace.epsilon,
            )
        )
        self.assertTrue(
            torch.equal(
                first[1].replay_trace.epsilon,
                second[1].replay_trace.epsilon,
            )
        )
        self.assertFalse(
            torch.equal(
                first[0].replay_trace.epsilon,
                first[1].replay_trace.epsilon,
            )
        )
        self.assertEqual(tuple(first[0].soft_prefix.shape), (2, 3))
        self.assertEqual(tuple(first[0].replay_trace.state_tokens.shape), (1, 3))
        self.assertEqual(first[0].soft_prefix.device.type, "cpu")
        self.assertFalse(first[0].soft_prefix.requires_grad)

    def test_mean_conditioning_records_zero_epsilon(self) -> None:
        from infoskill.conditioning import InfoSkillConditioningWorkItem
        from infoskill.integrations.verl.worker_conditioning import (
            InfoSkillWorkerConditioner,
        )

        engine = InfoSkillWorkerConditioner(
            library=_Library(),
            semantic_encoder=_SemanticEncoder(),
            feature_cache=_FeatureCache(),
            compressor=_Compressor(),
            projector=_Projector(),
        )
        items = (
            InfoSkillConditioningWorkItem(
                compression_view="state one",
                candidate_skill_ids=("skill-1",),
                latent_seed=13,
                latent_mode="mean",
            ),
            InfoSkillConditioningWorkItem(
                compression_view="state two",
                candidate_skill_ids=("skill-1",),
                latent_seed=17,
                latent_mode="mean",
            ),
        )

        outputs = engine.condition(items)

        self.assertTrue(
            torch.equal(
                outputs[0].replay_trace.latent,
                torch.tensor([1.0, 2.0]),
            )
        )
        self.assertTrue(torch.count_nonzero(outputs[0].replay_trace.epsilon) == 0)


if torch is not None:

    class _Features:
        def __init__(self, tokens, valid) -> None:
            self.tokens = tokens
            self.valid = valid


    class _SemanticEncoder:
        def encode_tokens(self, texts):
            if list(texts) != ["state one", "state two"]:
                raise AssertionError(texts)
            return _Features(
                torch.tensor(
                    [
                        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
                        [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    ]
                ),
                torch.tensor([[True, False], [True, True]]),
            )


    class _Library:
        def get(self, skill_id):
            if skill_id != "skill-1":
                raise KeyError(skill_id)
            return skill_id


    class _FeatureCache:
        def heterogeneous_skill_batch(self, groups):
            if groups != [("skill-1",), ("skill-1",)]:
                raise AssertionError(groups)
            return (
                torch.ones((2, 1, 1, 3)),
                torch.ones((2, 1, 1), dtype=torch.bool),
                torch.zeros((2, 1), dtype=torch.long),
            )


    class _Outputs:
        def __init__(self, summary, mu, logvar, latent, epsilon) -> None:
            self.state_summary = summary
            self.posterior_mu = mu
            self.posterior_logvar = logvar
            self.latent = latent
            self.epsilon = epsilon


    class _Compressor(torch.nn.Module):
        latent_dim = 2

        def forward(
            self,
            *,
            state_tokens,
            latent_mode,
            replay_epsilon,
            **kwargs,
        ):
            summary = state_tokens[:, 0]
            mu = summary[:, :2]
            logvar = torch.zeros_like(mu)
            epsilon = (
                torch.zeros_like(mu)
                if latent_mode == "mean"
                else replay_epsilon
            )
            return _Outputs(summary, mu, logvar, mu + epsilon, epsilon)


    class _Projector(torch.nn.Module):
        prefix_length = 2
        policy_hidden_size = 3

        def forward(self, latent):
            return (
                torch.cat((latent, latent[:, :1]), dim=-1)
                .unsqueeze(1)
                .repeat(1, 2, 1)
            )


if __name__ == "__main__":
    unittest.main()
