from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from infoskill.app_config import AppConfig
from infoskill.builders import (
    audit_raw_skill_prompt_budget,
    build_raw_skill_setup,
)
from infoskill.conditioning import ConditioningRequest
from infoskill.domain.state import CanonicalAgentState, render_state_views


class RawSkillTrainingSetupTests(unittest.TestCase):
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
