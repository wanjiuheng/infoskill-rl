from __future__ import annotations

import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "requires torch")
class AuxiliaryBatchBuilderTests(unittest.TestCase):
    def test_build_broadcasts_fidelity_and_resolves_grounding(self) -> None:
        from infoskill.conditioning import (
            ConditionedPolicyInput,
            InfoSkillReplayTrace,
        )
        from infoskill.domain.actions import resolve_action
        from infoskill.domain.state import CanonicalAgentState
        from infoskill.episode import (
            EnvironmentTransition,
            TaskSpec,
            Trajectory,
            TrajectoryGroup,
            TrajectoryStep,
        )
        from infoskill.integrations.alfworld import GroundingSample
        from infoskill.rollout import GenerationResult
        from infoskill.semantic import FeatureBatch, SemanticFeatureCache
        from infoskill.skills import FixedSkillLibrary
        from infoskill.training import AuxiliaryBatchBuilder

        class _Encoder:
            device = torch.device("cpu")
            hidden_size = 6

            def encode_tokens(self, texts, *, max_length=512):
                del max_length
                lengths = [2 + index % 2 for index in range(len(texts))]
                tokens = torch.zeros(len(texts), max(lengths), self.hidden_size)
                valid = torch.zeros(len(texts), max(lengths), dtype=torch.bool)
                for index, length in enumerate(lengths):
                    tokens[index, :length] = index + 1
                    valid[index, :length] = True
                return FeatureBatch(tokens, valid)

        library = FixedSkillLibrary.load(
            Path(__file__).parents[1] / "fixtures" / "skills.json"
        )
        encoder = _Encoder()
        cache = SemanticFeatureCache(encoder)  # type: ignore[arg-type]
        task = TaskSpec("task", "train", "clean", "clean an apple")
        state = CanonicalAgentState(
            task_id=task.task_id,
            split=task.split,
            task_type=task.task_type,
            goal=task.goal,
            step_index=0,
            observation="Kitchen.",
            history=(),
            admissible_commands=("look", "open fridge 1"),
            candidate_skill_ids=("gen_a", "clean_a", "err_a"),
        )

        def step(seed: int, token_count: int):
            trace = InfoSkillReplayTrace(
                latent_seed=seed,
                state_summary=torch.zeros(8),
                state_tokens=torch.full((token_count, 6), float(seed)),
                posterior_mu=torch.zeros(4),
                posterior_logvar=torch.zeros(4),
                latent=torch.zeros(4),
                epsilon=torch.full((4,), float(seed)),
            )
            policy_input = ConditionedPolicyInput(
                user_message="prompt",
                candidate_skill_ids=state.candidate_skill_ids,
                soft_prefix=torch.zeros(5, 8),
                conditioning_trace=trace,
            )
            transition = EnvironmentTransition(
                next_state=state,
                raw_observation="Kitchen.",
                raw_reward=0.0,
                raw_done=False,
                raw_won=False,
                info={},
            )
            return TrajectoryStep(
                state_before=state,
                conditioned_input=policy_input,
                generation=GenerationResult("request", "<action>look</action>", "stop", (1,), (-0.1,), 4),
                action=resolve_action("<action>look</action>", state.admissible_commands),
                transition=transition,
            )

        trajectories = (
            Trajectory(task, 0, (step(1, 2),), False, True, False, 0, 0.0),
            Trajectory(
                task,
                1,
                (step(2, 3), step(3, 2)),
                True,
                True,
                False,
                0,
                1.0,
            ),
        )
        builder = AuxiliaryBatchBuilder(
            library=library,
            semantic_encoder=encoder,  # type: ignore[arg-type]
            feature_cache=cache,
            history_length=2,
            latent_dim=4,
        )

        batch = builder.build(
            groups=(TrajectoryGroup(task, trajectories),),
            fidelity_targets=((-0.5, 0.5),),
            grounding_samples=(GroundingSample(state, "look"),),
            grounding_epsilon_seeds=(19,),
        )

        self.assertEqual(tuple(batch.online.replay.state_tokens.shape), (3, 3, 6))
        self.assertEqual(batch.online.trajectory_index.tolist(), [0, 1, 1])
        self.assertEqual(batch.online.fidelity_target.tolist(), [-0.5, 0.5, 0.5])
        self.assertEqual(batch.offline.grounding_target.tolist(), [0])
        self.assertEqual(tuple(batch.online.replay.skill_tokens.shape[:2]), (3, 3))
        self.assertEqual(tuple(batch.offline.replay.epsilon.shape), (1, 4))

        repeated = builder.build(
            groups=(TrajectoryGroup(task, trajectories),),
            fidelity_targets=((-0.5, 0.5),),
            grounding_samples=(GroundingSample(state, "look"),),
            grounding_epsilon_seeds=(19,),
        )
        self.assertTrue(
            torch.equal(batch.offline.replay.epsilon, repeated.offline.replay.epsilon)
        )


if __name__ == "__main__":
    unittest.main()
