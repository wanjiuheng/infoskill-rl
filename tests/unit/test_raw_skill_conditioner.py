from __future__ import annotations

import unittest
from pathlib import Path

from infoskill.conditioning import (
    ConditioningRequest,
    RawSkillPromptConditioner,
    SkillRlGrpoPromptConditioner,
    SkillRlSftPromptConditioner,
    SkillRlSftNoSkillsPromptConditioner,
    format_raw_skill_block,
)
from infoskill.domain import (
    AgentHistoryEntry,
    CanonicalAgentState,
    render_state_views,
)
from infoskill.skills import FixedSkillLibrary, TemplateRetriever


class RawSkillConditionerTests(unittest.TestCase):
    def test_skillrl_sft_no_skills_keeps_shell_without_skill_section(self) -> None:
        conditioner = SkillRlSftNoSkillsPromptConditioner(history_length=5)
        state = CanonicalAgentState(
            task_id="task",
            split="valid_seen",
            task_type="pick_and_place_simple",
            goal="put a book in sofa.",
            step_index=0,
            observation=(
                "-= Welcome to TextWorld, ALFRED! =-\n\n"
                "You are in the middle of a room."
            ),
            history=(),
            admissible_commands=("go to sofa 1", "look"),
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

        self.assertEqual(context.candidate_skill_ids, ())
        self.assertEqual(conditioned.candidate_skill_ids, ())
        self.assertNotIn("Retrieved Relevant Experience", conditioned.user_message)
        self.assertNotIn("Welcome to TextWorld", conditioned.user_message)
        self.assertIn("Your task is to: put a book in sofa\n", conditioned.user_message)
        self.assertIn("## Current Progress\n", conditioned.user_message)
        self.assertFalse(conditioned.conditioning_trace["skills_injected"])

    def test_skillrl_sft_prompt_injects_skills_on_initial_step(self) -> None:
        library = FixedSkillLibrary.load(
            Path(__file__).parents[1] / "fixtures" / "skills.json"
        )
        conditioner = SkillRlSftPromptConditioner(
            TemplateRetriever(
                library,
                general_count=1,
                task_count=1,
                mistake_count=1,
            ),
            history_length=5,
        )
        state = CanonicalAgentState(
            task_id="task",
            split="valid_seen",
            task_type="pick_clean_then_place_in_recep",
            goal="put a clean apple in a receptacle.",
            step_index=0,
            observation=(
                "-= Welcome to TextWorld, ALFRED! =-\n\n"
                "Kitchen."
            ),
            history=(),
            admissible_commands=("help", "look"),
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
        self.assertEqual(
            conditioned.user_message,
            "You are an expert agent operating in the ALFRED Embodied Environment.\n"
            "Your task is to: put a clean apple in a receptacle\n\n"
            "## Retrieved Relevant Experience\n\n"
            "### General Principles\n"
            "- **General A**: alpha\n\n"
            "### Clean Skills\n"
            "- **Clean A**: wash\n"
            "  _Apply when: dirty_\n\n"
            "### Mistakes to Avoid\n"
            "- **Don't**: Loop\n"
            "  **Instead**: remember\n\n"
            "## Current Progress\n"
            "Your current observation is: Kitchen.\n"
            "Your admissible actions of the current situation are: [help, look].\n\n"
            "Now it's your turn to take an action.\n"
            "You should first reason step-by-step about the current situation. "
            "This reasoning process MUST be enclosed within <think> </think> tags.\n"
            "Once you've finished your reasoning, you should choose an admissible "
            "action for current step and present it within <action> </action> tags.",
        )
        self.assertEqual(
            conditioned.conditioning_trace,
            {
                "prompt_protocol": "skillrl-sft-alfworld-v1",
                "skills_injected": True,
            },
        )

    def test_skillrl_sft_prompt_uses_five_recent_reindexed_history_entries(self) -> None:
        library = FixedSkillLibrary.load(
            Path(__file__).parents[1] / "fixtures" / "skills.json"
        )
        conditioner = SkillRlSftPromptConditioner(
            TemplateRetriever(
                library,
                general_count=1,
                task_count=1,
                mistake_count=1,
            ),
            history_length=5,
        )
        initial = CanonicalAgentState(
            task_id="task",
            split="valid_seen",
            task_type="pick_clean_then_place_in_recep",
            goal="put a clean apple in a receptacle.",
            step_index=0,
            observation="Kitchen.",
            history=(),
            admissible_commands=("look",),
        )
        context = conditioner.prepare_group(initial)
        state = CanonicalAgentState(
            task_id=initial.task_id,
            split=initial.split,
            task_type=initial.task_type,
            goal=initial.goal,
            step_index=6,
            observation="At the counter.",
            history=tuple(
                AgentHistoryEntry(
                    index,
                    (
                        "-= Welcome to TextWorld, ALFRED! =-\n\n"
                        if index == 1
                        else ""
                    )
                    + f"Observation {index}.",
                    f"action {index}",
                )
                for index in range(6)
            ),
            admissible_commands=("help", "take apple 1 from countertop 1", "look"),
        )

        conditioned = conditioner.condition_batch(
            (
                ConditioningRequest(
                    state=state,
                    views=render_state_views(state),
                    rollout_id=0,
                    global_update=0,
                    latent_seed=2,
                ),
            ),
            context,
        )[0]

        self.assertNotIn("Observation 0.", conditioned.user_message)
        self.assertNotIn("Welcome to TextWorld", conditioned.user_message)
        self.assertIn(
            "most recent 5 observations and the corresponding actions you took: "
            "[Observation 1: 'Observation 1.', Action 1: 'action 1']",
            conditioned.user_message,
        )
        self.assertIn(
            "[Observation 5: 'Observation 5.', Action 5: 'action 5']",
            conditioned.user_message,
        )
        self.assertIn("You are now at step 7", conditioned.user_message)
        self.assertIn(
            "[help, take apple 1 from countertop 1, look]",
            conditioned.user_message,
        )
        self.assertEqual(conditioned.history_entries_omitted, 1)

    def test_skillrl_grpo_prompt_omits_skills_on_initial_step(self) -> None:
        library = FixedSkillLibrary.load(
            Path(__file__).parents[1] / "fixtures" / "skills.json"
        )
        conditioner = SkillRlGrpoPromptConditioner(
            TemplateRetriever(
                library,
                general_count=1,
                task_count=1,
                mistake_count=1,
            ),
            history_length=2,
        )
        state = CanonicalAgentState(
            task_id="task",
            split="valid_seen",
            task_type="pick_clean_then_place_in_recep",
            goal="put a clean apple in a receptacle.",
            step_index=0,
            observation="Kitchen.",
            history=(),
            admissible_commands=("help", "look"),
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
        self.assertEqual(
            conditioned.user_message,
            "\nYou are an expert agent operating in the ALFRED Embodied Environment.\n"
            "Your current observation is: Kitchen.\n\n"
            "Your task is to: put a clean apple in a receptacle.\n"
            "Your admissible actions of the current situation are: ['look'].\n\n"
            "Now it's your turn to take an action.\n"
            "You should first reason step-by-step about the current situation. "
            "This reasoning process MUST be enclosed within <think> </think> tags. \n"
            "Once you've finished your reasoning, you should choose an admissible "
            "action for current step and present it within <action> </action> tags.\n",
        )

    def test_skillrl_grpo_prompt_injects_native_memory_shape_after_step_zero(self) -> None:
        library = FixedSkillLibrary.load(
            Path(__file__).parents[1] / "fixtures" / "skills.json"
        )
        conditioner = SkillRlGrpoPromptConditioner(
            TemplateRetriever(
                library,
                general_count=1,
                task_count=1,
                mistake_count=1,
            ),
            history_length=2,
        )
        initial = CanonicalAgentState(
            task_id="task",
            split="valid_seen",
            task_type="pick_clean_then_place_in_recep",
            goal="put a clean apple in a receptacle.",
            step_index=0,
            observation="Kitchen.",
            history=(),
            admissible_commands=("look",),
        )
        context = conditioner.prepare_group(initial)
        state = CanonicalAgentState(
            task_id=initial.task_id,
            split=initial.split,
            task_type=initial.task_type,
            goal=initial.goal,
            step_index=1,
            observation="At the counter.",
            history=(AgentHistoryEntry(0, "Kitchen.", "go to countertop 1"),),
            admissible_commands=("help", "take apple 1 from countertop 1", "look"),
        )

        conditioned = conditioner.condition_batch(
            (
                ConditioningRequest(
                    state=state,
                    views=render_state_views(state),
                    rollout_id=0,
                    global_update=0,
                    latent_seed=2,
                ),
            ),
            context,
        )[0]

        self.assertEqual(
            conditioned.user_message,
            "\nYou are an expert agent operating in the ALFRED Embodied Environment. "
            "Your task is to: put a clean apple in a receptacle.\n\n"
            "## Retrieved Relevant Experience\n\n"
            "### General Principles\n"
            "- **General A**: alpha\n\n"
            "### Clean Skills\n"
            "- **Clean A**: wash\n"
            "  _Apply when: dirty_\n\n"
            "### Mistakes to Avoid\n"
            "- **Don't**: Loop\n"
            "  **Instead**: remember\n\n"
            "## Current Progress\n\n"
            "Prior to this step, you have already taken 1 step(s). Below are the "
            "most recent 1 observations and the corresponding actions you took: "
            "[Observation 1: 'Kitchen.', Action 1: 'go to countertop 1']\n"
            "You are now at step 2 and your current observation is: At the counter.\n"
            "Your admissible actions of the current situation are: "
            "['take apple 1 from countertop 1'\n 'look'].\n\n"
            "Now it's your turn to take an action.\n"
            "You should first reason step-by-step about the current situation. "
            "This reasoning process MUST be enclosed within <think> </think> tags.\n"
            "Once you've finished your reasoning, you should choose an admissible "
            "action for current step and present it within <action> </action> tags.\n",
        )

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
