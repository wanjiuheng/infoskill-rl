from __future__ import annotations

import unittest
from pathlib import Path

from infoskill.conditioning import (
    ConditioningRequest,
    RawSkillPromptConditioner,
    format_raw_skill_block,
)
from infoskill.domain import CanonicalAgentState, render_state_views
from infoskill.skills import FixedSkillLibrary, TemplateRetriever


class RawSkillConditionerTests(unittest.TestCase):
    def test_skillrl_prompt_format_matches_the_pinned_reference_shape(self) -> None:
        library = FixedSkillLibrary.load(
            Path(__file__).parents[1] / "fixtures" / "skills.json"
        )
        retrieval = TemplateRetriever(
            library,
            general_count=1,
            task_count=1,
            mistake_count=1,
        ).retrieve("clean an apple")

        block = format_raw_skill_block(retrieval, style="skillrl")

        self.assertEqual(
            block,
            "### General Principles\n"
            "- **General A**: alpha\n\n"
            "### Clean Skills\n"
            "- **Clean A**: wash\n"
            "  _Apply when: dirty_\n\n"
            "### Mistakes to Avoid\n"
            "- **Don't**: Loop\n"
            "  **Instead**: remember",
        )
        self.assertNotIn("skill_id", block)
        self.assertNotIn("why_it_happens", block)

    def test_skill_block_is_inserted_after_goal_and_retrieved_once_per_group(self) -> None:
        library = FixedSkillLibrary.load(Path(__file__).parents[1] / "fixtures" / "skills.json")
        conditioner = RawSkillPromptConditioner(TemplateRetriever(library), history_length=2)
        state = CanonicalAgentState(
            task_id="task",
            split="train",
            task_type="pick_clean_then_place_in_recep",
            goal="clean an apple",
            step_index=0,
            observation="Kitchen.",
            history=(),
            admissible_commands=("look",),
        )
        context = conditioner.prepare_group(state)

        conditioned = conditioner.condition_batch(
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

        self.assertEqual(conditioned.candidate_skill_ids, context.candidate_skill_ids)
        self.assertLess(conditioned.user_message.index("clean an apple"), conditioned.user_message.index("## Retrieved"))
        self.assertLess(conditioned.user_message.index("## Retrieved"), conditioned.user_message.index("Prior to this step"))
        self.assertIn("[clean_a] type=task_specific category=clean", conditioned.user_message)

    def test_registered_task_goal_is_the_retrieval_key_not_environment_wording(self) -> None:
        library = FixedSkillLibrary.load(
            Path(__file__).parents[1] / "fixtures" / "skills.json"
        )

        class RecordingRetriever(TemplateRetriever):
            def __init__(self) -> None:
                super().__init__(library)
                self.queries: list[str] = []

            def retrieve(self, query: str):
                self.queries.append(query)
                return super().retrieve(query)

        retriever = RecordingRetriever()
        conditioner = RawSkillPromptConditioner(
            retriever,
            history_length=2,
            query_by_task_id={"game-1": "clean the human-described apple"},
        )
        environment_state = CanonicalAgentState(
            task_id="game-1",
            split="train",
            task_type="pick_clean_then_place_in_recep",
            goal="put a clean apple in a receptacle",
            step_index=0,
            observation="Kitchen.",
            history=(),
            admissible_commands=("look",),
        )

        conditioner.prepare_group(environment_state)

        self.assertEqual(retriever.queries, ["clean the human-described apple"])


if __name__ == "__main__":
    unittest.main()
