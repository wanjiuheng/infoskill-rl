from __future__ import annotations

import unittest
from dataclasses import replace

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "requires torch")
class DistributedAuxiliaryUpdaterTests(unittest.TestCase):
    def test_accumulates_weighted_microbatches_into_one_atomic_step(self) -> None:
        from infoskill.integrations.verl.auxiliary_updater import (
            DistributedAuxiliaryUpdater,
        )
        from infoskill.learning import (
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

        torch.manual_seed(3)
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
        parameters = [
            parameter
            for module in (compressor, prior, fidelity, grounding)
            for parameter in module.parameters()
        ]
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda _: 1.0,
        )
        updater = DistributedAuxiliaryUpdater(
            compressor=compressor,
            prior=prior,
            fidelity=fidelity,
            grounding=grounding,
            optimizer=optimizer,
            scheduler=scheduler,
            world_size=1,
            all_reduce=lambda _: None,
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
            trajectory_index=torch.tensor([0, 0, 1, 99]),
            fidelity_target=torch.tensor([-0.5, -0.5, 0.5, 9.0]),
            step_weight=torch.tensor([0.5, 0.5, 1.0, 0.0]),
        )
        offline = OfflineGroundingBatch(
            replay=replay(3),
            command_embeddings=torch.randn(3, 3, semantic_width),
            command_valid=torch.ones(3, 3, dtype=torch.bool),
            grounding_target=torch.tensor([0, 1, 2]),
            sample_weight=torch.tensor([1.0, 1.0, 0.0]),
        )
        before = compressor.posterior_mu.weight.detach().clone()
        scheduler_step_before = scheduler.last_epoch

        metrics = updater.update(
            online_batches=(
                replace(
                    online,
                    replay=_slice_replay(online.replay, slice(0, 2)),
                    trajectory_index=online.trajectory_index[:2],
                    fidelity_target=online.fidelity_target[:2],
                    step_weight=online.step_weight[:2],
                ),
                replace(
                    online,
                    replay=_slice_replay(online.replay, slice(2, 4)),
                    trajectory_index=online.trajectory_index[2:],
                    fidelity_target=online.fidelity_target[2:],
                    step_weight=online.step_weight[2:],
                ),
            ),
            offline_batches=(offline,),
            global_trajectory_count=2,
            global_offline_count=2,
        )

        self.assertFalse(torch.equal(before, compressor.posterior_mu.weight))
        self.assertEqual(scheduler.last_epoch, scheduler_step_before + 1)
        self.assertEqual(metrics["aux/optimizer_step_applied"], 1.0)
        self.assertEqual(metrics["aux/online_step_count"], 3.0)
        self.assertEqual(metrics["aux/online_trajectory_count"], 2.0)
        self.assertEqual(metrics["aux/offline_sample_count"], 2.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))


def _slice_replay(replay, index):
    return replace(
        replay,
        state_tokens=replay.state_tokens[index],
        state_valid=replay.state_valid[index],
        skill_tokens=replay.skill_tokens[index],
        skill_valid=replay.skill_valid[index],
        skill_kind_ids=replay.skill_kind_ids[index],
        epsilon=replay.epsilon[index],
    )


if __name__ == "__main__":
    unittest.main()
