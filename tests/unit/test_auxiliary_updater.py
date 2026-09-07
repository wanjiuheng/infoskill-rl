from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "requires torch")
class AuxiliaryUpdaterTests(unittest.TestCase):
    def _fixture(self):
        from infoskill.learning import (
            AuxiliaryTrainingBatch,
            AuxiliaryUpdater,
            CompressionReplayBatch,
            OfflineGroundingBatch,
            OnlineAuxiliaryBatch,
        )
        from infoskill.models import (
            ExecutableGroundingHead,
            FidelityPredictor,
            InfoSkillCompressor,
            StateConditionedPrior,
        )

        torch.manual_seed(7)
        semantic_width = 6
        latent_dim = 4
        compressor = InfoSkillCompressor(
            semantic_width,
            model_width=8,
            latent_dim=latent_dim,
            attention_layers=1,
            attention_heads=2,
            max_candidate_skills=2,
        )
        prior = StateConditionedPrior(state_width=8, latent_dim=latent_dim)
        fidelity = FidelityPredictor(state_width=8, latent_dim=latent_dim)
        grounding = ExecutableGroundingHead(
            semantic_width=semantic_width,
            state_width=8,
            latent_dim=latent_dim,
            key_width=8,
        )
        modules = (compressor, prior, fidelity, grounding)
        optimizer = torch.optim.AdamW(
            [parameter for module in modules for parameter in module.parameters()],
            lr=1e-3,
        )
        updater = AuxiliaryUpdater(
            compressor=compressor,
            prior=prior,
            fidelity=fidelity,
            grounding=grounding,
            optimizer=optimizer,
        )

        def replay(batch_size: int):
            return CompressionReplayBatch(
                state_tokens=torch.randn(batch_size, 3, semantic_width),
                state_valid=torch.ones(batch_size, 3, dtype=torch.bool),
                skill_tokens=torch.randn(batch_size, 2, 2, semantic_width),
                skill_valid=torch.ones(batch_size, 2, 2, dtype=torch.bool),
                skill_kind_ids=torch.tensor([[0, 1]]).expand(batch_size, -1),
                epsilon=torch.randn(batch_size, latent_dim),
            )

        online = OnlineAuxiliaryBatch(
            replay=replay(4),
            trajectory_index=torch.tensor([0, 0, 1, 1]),
            fidelity_target=torch.tensor([-0.5, -0.5, 0.5, 0.5]),
        )
        offline = OfflineGroundingBatch(
            replay=replay(3),
            command_embeddings=torch.randn(3, 3, semantic_width),
            command_valid=torch.tensor(
                [[True, True, False], [True, True, True], [True, False, False]]
            ),
            grounding_target=torch.tensor([1, 2, 0]),
        )
        return updater, AuxiliaryTrainingBatch(online=online, offline=offline), compressor

    def test_update_changes_parameters_and_reports_registered_metrics(self) -> None:
        updater, batch, compressor = self._fixture()
        before = compressor.posterior_mu.weight.detach().clone()

        metrics = updater.update(batch)

        self.assertFalse(torch.equal(before, compressor.posterior_mu.weight.detach()))
        self.assertEqual(metrics["aux/optimizer_step_applied"], 1.0)
        self.assertEqual(metrics["aux/online_step_count"], 4.0)
        self.assertEqual(metrics["aux/online_trajectory_count"], 2.0)
        self.assertEqual(metrics["aux/offline_sample_count"], 3.0)
        for value in metrics.values():
            self.assertTrue(torch.isfinite(torch.tensor(value)))

    def test_invalid_grounding_target_is_rejected_before_optimizer_step(self) -> None:
        from dataclasses import replace

        updater, batch, compressor = self._fixture()
        before = compressor.posterior_mu.weight.detach().clone()
        invalid_offline = replace(
            batch.offline,
            grounding_target=torch.tensor([2, 2, 0]),
        )

        with self.assertRaisesRegex(ValueError, "must point to a valid command"):
            updater.update(replace(batch, offline=invalid_offline))

        self.assertTrue(torch.equal(before, compressor.posterior_mu.weight.detach()))

    def test_float32_modules_accept_bfloat16_frozen_features(self) -> None:
        from dataclasses import replace

        updater, batch, _ = self._fixture()

        def bfloat16(replay):
            return replace(
                replay,
                state_tokens=replay.state_tokens.bfloat16(),
                skill_tokens=replay.skill_tokens.bfloat16(),
            )

        converted = replace(
            batch,
            online=replace(batch.online, replay=bfloat16(batch.online.replay)),
            offline=replace(
                batch.offline,
                replay=bfloat16(batch.offline.replay),
                command_embeddings=batch.offline.command_embeddings.bfloat16(),
            ),
        )

        metrics = updater.update(converted)

        self.assertEqual(metrics["aux/optimizer_step_applied"], 1.0)


if __name__ == "__main__":
    unittest.main()
