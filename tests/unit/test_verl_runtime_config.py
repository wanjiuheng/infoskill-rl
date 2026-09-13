from __future__ import annotations

import unittest

try:
    import ray  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - local docs-only environment
    ray = None


@unittest.skipIf(ray is None, "ray is not installed")
class VerlRuntimeConfigTests(unittest.TestCase):
    def test_infoskill_modules_require_complete_hybrid_configuration(self) -> None:
        from infoskill.integrations.verl.runtime import VerlRuntimeConfig

        common = {
            "skillrl_source": "/skillrl",
            "model_path": "/policy",
            "num_gpus": 4,
            "enable_infoskill_modules": True,
        }

        with self.assertRaisesRegex(ValueError, "hybrid-prefix"):
            VerlRuntimeConfig(**common)
        with self.assertRaisesRegex(ValueError, "requires INFO-SKILL modules"):
            VerlRuntimeConfig(
                skillrl_source="/skillrl",
                model_path="/policy",
                num_gpus=4,
                enable_infoskill_auxiliary=True,
            )
        with self.assertRaisesRegex(ValueError, "semantic model"):
            VerlRuntimeConfig(**common, require_hybrid_prefix=True)
        with self.assertRaisesRegex(ValueError, "skill bank"):
            VerlRuntimeConfig(
                **common,
                require_hybrid_prefix=True,
                semantic_model_path="/semantic",
            )

        settings = VerlRuntimeConfig(
            **common,
            require_hybrid_prefix=True,
            semantic_model_path="/semantic",
            skill_bank_path="/skills.json",
        )
        self.assertEqual(settings.soft_prefix_length, 5)
        self.assertEqual(settings.infoskill_latent_dim, 32)
        self.assertEqual(settings.infoskill_projector_learning_rate, 1e-4)
        self.assertEqual(settings.infoskill_projector_weight_decay, 0.01)
        self.assertEqual(settings.infoskill_policy_warmup_ratio, 0.03)
        self.assertFalse(settings.skip_unused_old_logprob_entropy)
        self.assertEqual(settings.rollout_max_batched_tokens, 16_384)

        with self.assertRaisesRegex(ValueError, "maximum-length sequence"):
            VerlRuntimeConfig(
                skillrl_source="/skillrl",
                model_path="/policy",
                num_gpus=4,
                rollout_max_batched_tokens=4_000,
            )
        with self.assertRaisesRegex(ValueError, "only for INFO-SKILL"):
            VerlRuntimeConfig(
                skillrl_source="/skillrl",
                model_path="/policy",
                num_gpus=4,
                skip_unused_old_logprob_entropy=True,
            )
        with self.assertRaisesRegex(ValueError, "only for INFO-SKILL"):
            VerlRuntimeConfig(
                skillrl_source="/skillrl",
                model_path="/policy",
                num_gpus=4,
                rollout_max_batched_tokens=32_768,
            )

        with self.assertRaisesRegex(ValueError, "grounding data"):
            VerlRuntimeConfig(
                **common,
                require_hybrid_prefix=True,
                semantic_model_path="/semantic",
                skill_bank_path="/skills.json",
                enable_infoskill_auxiliary=True,
            )

    def test_named_auxiliary_seeds_are_stable_and_namespaced(self) -> None:
        from infoskill.integrations.verl.runtime import (
            _effective_global_minibatch_size,
            _named_seed,
            _policy_micro_batch_size_per_gpu,
        )

        first = _named_seed(7, "sample", 3, "task")
        self.assertEqual(first, _named_seed(7, "sample", 3, "task"))
        self.assertNotEqual(first, _named_seed(7, "epsilon", 3, "task"))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**63 - 1)
        self.assertEqual(_policy_micro_batch_size_per_gpu(256, 2), 4)
        self.assertEqual(_policy_micro_batch_size_per_gpu(256, 4), 4)
        self.assertEqual(_policy_micro_batch_size_per_gpu(256, 3), 1)
        self.assertEqual(_policy_micro_batch_size_per_gpu(16, 3), 1)
        self.assertEqual(_effective_global_minibatch_size(256, 4), 256)
        self.assertEqual(_effective_global_minibatch_size(256, 3), 255)

    def test_conditioning_rpc_preserves_rows_across_world_size_padding(self) -> None:
        import numpy as np
        from verl import DataProto

        from infoskill.conditioning import ConditioningRequest
        from infoskill.domain.state import CanonicalAgentState, render_state_views
        from infoskill.integrations.verl.runtime import VerlRuntime, VerlRuntimeConfig

        worker_group = _WorkerGroup(DataProto, np)
        settings = VerlRuntimeConfig(
            skillrl_source="/skillrl",
            model_path="/policy",
            num_gpus=2,
            enable_infoskill_modules=True,
            require_hybrid_prefix=True,
            semantic_model_path="/semantic",
            skill_bank_path="/skills.json",
        )
        runtime = VerlRuntime(
            worker_group=worker_group,
            codec=object(),  # type: ignore[arg-type]
            config=settings,
        )
        state = CanonicalAgentState(
            task_id="task-1",
            split="train",
            task_type="pick_and_place_simple",
            goal="put an object somewhere",
            step_index=0,
            observation="room",
            history=(),
            admissible_commands=("look",),
        )
        request = ConditioningRequest(
            state=state,
            views=render_state_views(state),
            rollout_id=0,
            global_update=0,
            latent_seed=11,
        )

        outputs = runtime.condition_infoskill(
            (request,),
            ("skill-1",),
            latent_mode="sample",
        )

        self.assertEqual(worker_group.received_rows, 2)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].soft_prefix, "prefix-0")
        self.assertEqual(outputs[0].replay_trace.latent_seed, 11)

    def test_grouped_conditioning_replicates_rank_zero_sequence_exactly(self) -> None:
        import numpy as np
        from verl import DataProto

        from infoskill.conditioning import ConditioningRequest
        from infoskill.domain.state import CanonicalAgentState, render_state_views
        from infoskill.integrations.verl.runtime import VerlRuntime, VerlRuntimeConfig

        worker_group = _WorkerGroup(DataProto, np)
        runtime = VerlRuntime(
            worker_group=worker_group,
            codec=object(),  # type: ignore[arg-type]
            config=VerlRuntimeConfig(
                skillrl_source="/skillrl",
                model_path="/policy",
                num_gpus=2,
                enable_infoskill_modules=True,
                require_hybrid_prefix=True,
                semantic_model_path="/semantic",
                skill_bank_path="/skills.json",
            ),
        )
        state = CanonicalAgentState(
            task_id="task-1",
            split="train",
            task_type="pick_and_place_simple",
            goal="put an object somewhere",
            step_index=0,
            observation="room",
            history=(),
            admissible_commands=("look",),
        )
        requests = tuple(
            ConditioningRequest(
                state=state,
                views=render_state_views(state),
                rollout_id=index,
                global_update=0,
                latent_seed=11 + index,
            )
            for index in range(2)
        )

        outputs = runtime.condition_infoskill_grouped(
            requests,
            (("skill-1",), ("skill-2",)),
            latent_mode="mean",
        )

        self.assertEqual(worker_group.received_rows, 4)
        self.assertEqual(
            worker_group.received_candidate_groups,
            (("skill-1",), ("skill-2",), ("skill-1",), ("skill-2",)),
        )
        self.assertEqual(
            tuple(output.soft_prefix for output in outputs),
            ("prefix-0", "prefix-1"),
        )


class _WorkerGroup:
    world_size = 2

    def __init__(self, data_proto, numpy_module) -> None:
        self._data_proto = data_proto
        self._numpy = numpy_module
        self.received_rows = 0
        self.received_candidate_groups = ()

    def condition_infoskill(self, data):
        from infoskill.conditioning import (
            InfoSkillConditioningResult,
            InfoSkillReplayTrace,
        )

        self.received_rows = len(data)
        self.received_candidate_groups = tuple(
            item.candidate_skill_ids
            for item in data.non_tensor_batch["infoskill_work_item"]
        )
        results = []
        for row, item in enumerate(data.non_tensor_batch["infoskill_work_item"]):
            trace = InfoSkillReplayTrace(
                latent_seed=item.latent_seed,
                state_summary="summary",
                state_tokens="tokens",
                posterior_mu="mu",
                posterior_logvar="logvar",
                latent="latent",
                epsilon="epsilon",
            )
            results.append(InfoSkillConditioningResult(f"prefix-{row}", trace))
        return self._data_proto.from_dict(
            tensors={"infoskill_row_id": data.batch["infoskill_row_id"]},
            non_tensors={
                "infoskill_conditioning_result": self._numpy.asarray(
                    results,
                    dtype=object,
                )
            },
        )

    def condition_infoskill_serial(self, data):
        return self.condition_infoskill(data)


if __name__ == "__main__":
    unittest.main()
