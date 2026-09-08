from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from infoskill.app_config import AppConfig
from infoskill.builders import (
    audit_raw_skill_prompt_budget,
    build_raw_skill_setup,
    build_skillrl_grpo_prompt_setup,
    build_skillrl_sft_no_skills_prompt_setup,
    build_skillrl_sft_prompt_setup,
    build_verl_policy_evaluation,
)
from infoskill.conditioning import ConditioningRequest
from infoskill.domain.state import CanonicalAgentState, render_state_views
from infoskill.config import SkillMode
from infoskill.rollout import GenerationParameters


class _WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        del add_special_tokens
        return text.split()


class RawSkillTrainingSetupTests(unittest.TestCase):
    def test_skillrl_sft_no_skills_setup_has_no_retrieval_or_skill_budget(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")
        skill_bank = Path(__file__).parents[1] / "fixtures" / "skills.json"
        config = replace(
            config,
            paths=replace(
                config.paths,
                skill_bank=str(skill_bank),
                skill_bank_manifest=str(skill_bank.with_name("skills.manifest.json")),
            ),
        )

        setup = build_skillrl_sft_no_skills_prompt_setup(config)
        audit = audit_raw_skill_prompt_budget(
            setup,
            tokenizer=_WordTokenizer(),
            max_prompt_tokens=4096,
        )

        self.assertEqual(setup.provenance["retrieval_mode"], None)
        self.assertEqual(setup.provenance["prompt_format"], "skillrl_sft_no_skills")
        self.assertFalse(setup.provenance["skills_injected"])
        self.assertEqual(setup.skill_blocks, ())
        self.assertEqual(audit["raw_skill_block_count"], 0)
        self.assertEqual(audit["raw_skill_block_tokens_max"], 0)

    def test_verl_evaluation_sampling_override_is_explicit(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")
        parameters = GenerationParameters(
            do_sample=True,
            temperature=0.4,
            top_p=1.0,
            max_new_tokens=config.max_response_tokens,
        )

        with patch(
            "infoskill.builders.AlfworldEnvironmentFactory.from_paths",
            return_value=object(),
        ):
            collector = build_verl_policy_evaluation(
                config,
                mode=SkillMode.NO_SKILL,
                backend=object(),
                generation_parameters=parameters,
            )

        self.assertEqual(collector._generation_parameters, parameters)

    def test_skillrl_sft_setup_uses_dataset_observed_prompt_shape(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")
        skill_bank = Path(__file__).parents[1] / "fixtures" / "skills.json"
        config = replace(
            config,
            paths=replace(
                config.paths,
                skill_bank=str(skill_bank),
                skill_bank_manifest=str(skill_bank.with_name("skills.manifest.json")),
            ),
            retrieval_mode="embedding",
            history_length=2,
            task_top_k=0,
        )

        setup = build_skillrl_sft_prompt_setup(config)
        state = CanonicalAgentState(
            task_id="task",
            split="valid_seen",
            task_type="pick_clean_then_place_in_recep",
            goal="put a clean apple in a receptacle",
            step_index=0,
            observation="Kitchen.",
            history=(),
            admissible_commands=("look",),
        )
        context = setup.conditioner.prepare_group(state)

        self.assertEqual(
            context.candidate_skill_ids,
            ("gen_a", "gen_b", "clean_a", "err_a"),
        )
        self.assertEqual(setup.provenance["retrieval_mode"], "template")
        self.assertEqual(setup.provenance["prompt_format"], "skillrl_sft_exact")
        self.assertEqual(
            setup.provenance["task_category_classifier"],
            "skillrl-parquet-observed-keyword-v1",
        )
        self.assertEqual(
            setup.provenance["task_text_normalization"],
            "strip-terminal-period",
        )
        self.assertEqual(
            setup.provenance["observation_normalization"],
            "strip-textworld-welcome-banner",
        )
        self.assertEqual(setup.provenance["history_length"], 5)
        self.assertTrue(setup.provenance["step_zero_skill_injection"])
        self.assertEqual(setup.provenance["sft_dataset_row_count"], 7_486)
        self.assertEqual(setup.provenance["sft_trajectory_count"], 500)
        self.assertEqual(
            setup.provenance["sft_dataset_sha256"],
            "dfbbf265e19ac8087a54ec474727fcb400483a9a243eea6e977a02ae6ca85b94",
        )

    def test_skillrl_grpo_setup_is_template_only_and_episode_goal_driven(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")
        skill_bank = Path(__file__).parents[1] / "fixtures" / "skills.json"
        config = replace(
            config,
            paths=replace(
                config.paths,
                skill_bank=str(skill_bank),
                skill_bank_manifest=str(skill_bank.with_name("skills.manifest.json")),
            ),
            retrieval_mode="embedding",
            task_top_k=0,
        )

        setup = build_skillrl_grpo_prompt_setup(config)
        state = CanonicalAgentState(
            task_id="task",
            split="valid_seen",
            task_type="pick_clean_then_place_in_recep",
            goal="put a clean apple in a receptacle",
            step_index=0,
            observation="Kitchen.",
            history=(),
            admissible_commands=("look",),
        )
        context = setup.conditioner.prepare_group(state)

        self.assertEqual(
            context.candidate_skill_ids,
            ("gen_a", "gen_b", "clean_a", "err_a"),
        )
        self.assertEqual(setup.provenance["retrieval_mode"], "template")
        self.assertEqual(
            setup.provenance["retrieval_query_source"],
            "canonical_environment_goal_at_reset",
        )
        self.assertEqual(setup.provenance["prompt_format"], "skillrl_rl_exact")
        self.assertIsNone(setup.provenance["task_top_k"])
        self.assertFalse(setup.provenance["step_zero_skill_injection"])
        self.assertEqual(
            setup.provenance["reference_scope"],
            "prompt-and-initial-static-retrieval-only",
        )

    def test_skillrl_grpo_setup_rejects_non_reference_prompt_shape(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")

        with self.assertRaisesRegex(ValueError, "general_top_k=6"):
            build_skillrl_grpo_prompt_setup(
                replace(config, history_length=3)
            )

    def test_diagnostic_setup_can_override_retrieval_and_prompt_format(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")
        skill_bank = Path(__file__).parents[1] / "fixtures" / "skills.json"
        config = replace(
            config,
            paths=replace(
                config.paths,
                skill_bank=str(skill_bank),
                skill_bank_manifest=str(skill_bank.with_name("skills.manifest.json")),
            ),
            retrieval_mode="embedding",
            general_top_k=1,
            task_top_k=1,
            mistake_count=1,
        )

        setup = build_raw_skill_setup(
            config,
            retrieval_queries=("clean an apple",),
            retrieval_mode="template",
            prompt_format="skillrl",
        )

        self.assertEqual(setup.provenance["retrieval_mode"], "template")
        self.assertEqual(setup.provenance["prompt_format"], "skillrl")
        self.assertTrue(setup.skill_blocks[0].startswith("### General Principles"))
        self.assertEqual(config.retrieval_mode, "embedding")

    def test_compact_and_full_formats_preserve_the_same_retrieval_plan(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")
        skill_bank = Path(__file__).parents[1] / "fixtures" / "skills.json"
        config = replace(
            config,
            paths=replace(
                config.paths,
                skill_bank=str(skill_bank),
                skill_bank_manifest=str(skill_bank.with_name("skills.manifest.json")),
            ),
            retrieval_mode="template",
            general_top_k=1,
            task_top_k=1,
            mistake_count=1,
        )
        queries = {"task": "clean an apple and put it away"}

        compact = build_raw_skill_setup(
            config,
            retrieval_queries=queries,
            prompt_format="compact",
        )
        full = build_raw_skill_setup(
            config,
            retrieval_queries=queries,
            prompt_format="full",
        )

        self.assertEqual(
            compact.provenance["retrieval_plan"],
            full.provenance["retrieval_plan"],
        )
        self.assertEqual(
            compact.provenance["retrieval_plan_sha256"],
            full.provenance["retrieval_plan_sha256"],
        )
        self.assertNotEqual(compact.skill_blocks, full.skill_blocks)
        self.assertEqual(compact.provenance["prompt_format"], "compact")
        self.assertEqual(full.provenance["prompt_format"], "full")

    def test_template_setup_builds_conditioner_and_auditable_provenance(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")
        skill_bank = Path(__file__).parents[1] / "fixtures" / "skills.json"
        skill_manifest = skill_bank.with_name("skills.manifest.json")
        config = replace(
            config,
            paths=replace(
                config.paths,
                skill_bank=str(skill_bank),
                skill_bank_manifest=str(skill_manifest),
            ),
            retrieval_mode="template",
            general_top_k=1,
            task_top_k=1,
            mistake_count=1,
        )
        goal = "clean an apple and put it away"

        setup = build_raw_skill_setup(config, retrieval_queries=(goal,))
        state = CanonicalAgentState(
            task_id="task",
            split="train",
            task_type="clean",
            goal=goal,
            step_index=0,
            observation="Kitchen.",
            history=(),
            admissible_commands=("look",),
        )
        context = setup.conditioner.prepare_group(state)
        conditioned = setup.conditioner.condition_batch(
            (
                ConditioningRequest(
                    state=state,
                    views=render_state_views(state),
                    rollout_id=0,
                    global_update=0,
                    latent_seed=1,
                ),
            ),
            context,
        )[0]

        self.assertIn("## Retrieved Relevant Skills", conditioned.user_message)
        self.assertEqual(conditioned.candidate_skill_ids, ("gen_a", "clean_a", "err_a"))
        self.assertEqual(setup.provenance["retrieval_mode"], "template")
        self.assertEqual(setup.provenance["prompt_format"], "compact")
        self.assertNotIn("gen_a", conditioned.user_message)
        self.assertNotIn("why_it_happens", conditioned.user_message)
        self.assertEqual(setup.provenance["retrieval_query_count"], 1)
        self.assertEqual(len(setup.provenance["retrieval_plan_sha256"]), 64)
        self.assertEqual(setup.provenance["skill_bank_sha256"], setup.library.source_sha256)
        provenance = setup.provenance["skill_library_provenance"]
        self.assertEqual(provenance["source_split"], "train")
        self.assertEqual(provenance["trajectory_count"], 223)
        self.assertEqual(
            provenance["skill_bank_sha256_algorithm"],
            "canonical-json-sha256-v1",
        )
        self.assertEqual(len(setup.provenance["skill_library_provenance_id"]), 64)

    def test_skill_block_budget_is_audited_without_a_separate_raw_limit(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")
        skill_bank = Path(__file__).parents[1] / "fixtures" / "skills.json"
        skill_manifest = skill_bank.with_name("skills.manifest.json")
        config = replace(
            config,
            paths=replace(
                config.paths,
                skill_bank=str(skill_bank),
                skill_bank_manifest=str(skill_manifest),
            ),
            retrieval_mode="template",
            general_top_k=1,
            task_top_k=1,
            mistake_count=1,
        )
        setup = build_raw_skill_setup(
            config,
            retrieval_queries=(
                "clean an apple and put it away",
                "clean an apple and put it away",
            ),
        )

        class WordTokenizer:
            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                self.assert_no_special_tokens(add_special_tokens)
                return list(range(len(text.split())))

            @staticmethod
            def assert_no_special_tokens(value: bool) -> None:
                if value:
                    raise AssertionError("skill-block audit must exclude chat framing")

        audit = audit_raw_skill_prompt_budget(
            setup,
            tokenizer=WordTokenizer(),
            max_prompt_tokens=4096,
        )

        self.assertEqual(audit["raw_skill_block_count"], 1)
        self.assertGreater(audit["raw_skill_block_tokens_max"], 0)
        self.assertEqual(audit["raw_skill_prompt_token_limit"], 4096)
        self.assertNotIn("max_skill_prompt_tokens", audit)

        oversized = audit_raw_skill_prompt_budget(
            setup,
            tokenizer=WordTokenizer(),
            max_prompt_tokens=1,
        )
        self.assertFalse(oversized["raw_skill_block_fits_prompt_limit"])

    def test_checkpoint_must_use_the_same_raw_skill_conditioning(self) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")
        skill_bank = Path(__file__).parents[1] / "fixtures" / "skills.json"
        skill_manifest = skill_bank.with_name("skills.manifest.json")
        config = replace(
            config,
            paths=replace(
                config.paths,
                skill_bank=str(skill_bank),
                skill_bank_manifest=str(skill_manifest),
            ),
            retrieval_mode="template",
            general_top_k=1,
            task_top_k=1,
            mistake_count=1,
        )
        setup = build_raw_skill_setup(
            config,
            retrieval_queries={"task": "clean an apple and put it away"},
        )

        setup.require_checkpoint_compatibility(
            {"skill_conditioning": dict(setup.provenance)}
        )
        different_corpus = dict(setup.provenance)
        different_corpus["retrieval_query_count"] = 3_553
        different_corpus["retrieval_plan_sha256"] = "0" * 64
        setup.require_checkpoint_compatibility(
            {"skill_conditioning": different_corpus}
        )
        different_plan = dict(setup.provenance)
        different_plan["retrieval_plan"] = {
            "task": {
                "query_sha256": "0" * 64,
                "skill_ids": ["gen_b"],
            }
        }
        with self.assertRaisesRegex(RuntimeError, "retrieval plan differs"):
            setup.require_checkpoint_compatibility(
                {"skill_conditioning": different_plan}
            )
        incompatible = dict(setup.provenance)
        incompatible["general_top_k"] = 2
        with self.assertRaisesRegex(RuntimeError, "skill conditioning differs"):
            setup.require_checkpoint_compatibility(
                {"skill_conditioning": incompatible}
            )
        different_prompt_format = dict(setup.provenance)
        different_prompt_format["prompt_format"] = "skillrl"
        with self.assertRaisesRegex(RuntimeError, "skill conditioning differs"):
            setup.require_checkpoint_compatibility(
                {"skill_conditioning": different_prompt_format}
            )

    def test_checkpoint_semantic_identity_allows_relocation_but_not_new_weights(
        self,
    ) -> None:
        config = AppConfig.load("configs/alfworld_qwen25_7b.yaml")
        skill_bank = Path(__file__).parents[1] / "fixtures" / "skills.json"
        config = replace(
            config,
            paths=replace(
                config.paths,
                skill_bank=str(skill_bank),
                skill_bank_manifest=str(skill_bank.with_name("skills.manifest.json")),
            ),
            retrieval_mode="template",
            general_top_k=1,
            task_top_k=1,
            mistake_count=1,
        )
        setup = build_raw_skill_setup(
            config,
            retrieval_queries={"task": "clean an apple and put it away"},
        )
        current = {
            **setup.provenance,
            "semantic_model_identity": {
                "path": "/new/location/embedding-model",
                "algorithm": "model-files-sha256-v1",
                "sha256": "a" * 64,
            },
        }
        setup = replace(setup, provenance=current)
        relocated = {
            **current,
            "semantic_model_identity": {
                "path": "/old/location/embedding-model",
                "algorithm": "model-files-sha256-v1",
                "sha256": "a" * 64,
            },
        }
        setup.require_checkpoint_compatibility(
            {"skill_conditioning": relocated}
        )

        changed = dict(relocated)
        changed["semantic_model_identity"] = {
            "path": "/old/location/embedding-model",
            "algorithm": "model-files-sha256-v1",
            "sha256": "b" * 64,
        }
        with self.assertRaisesRegex(RuntimeError, "semantic model differs"):
            setup.require_checkpoint_compatibility(
                {"skill_conditioning": changed}
            )


if __name__ == "__main__":
    unittest.main()
