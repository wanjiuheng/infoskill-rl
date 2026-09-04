from __future__ import annotations

import unittest
from pathlib import Path

from infoskill.conditioning import RawSkillPromptConditioner
from infoskill.domain import CanonicalAgentState, render_state_views
from infoskill.skills import FixedSkillLibrary, TemplateRetriever


class RawSkillConditionerTests(unittest.TestCase):
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

        conditioned = conditioner.condition_batch((state,), (render_state_views(state),), context)[0]

        self.assertEqual(conditioned.candidate_skill_ids, context.candidate_skill_ids)
        self.assertLess(conditioned.user_message.index("clean an apple"), conditioned.user_message.index("## Retrieved"))
        self.assertLess(conditioned.user_message.index("## Retrieved"), conditioned.user_message.index("Prior to this step"))
        self.assertIn("[clean_a] type=task_specific category=clean", conditioned.user_message)


if __name__ == "__main__":
    unittest.main()
