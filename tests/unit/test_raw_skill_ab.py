from __future__ import annotations

import unittest
from collections import Counter

from infoskill.diagnostics.raw_skill_ab import (
    RAW_SKILL_AB_VARIANTS,
    SKILLRL_RL_EXACT_VARIANT,
    SKILLRL_SFT_CAUSAL_VARIANTS,
    SKILLRL_SFT_EXACT_VARIANT,
    resolve_raw_skill_diagnostic_variants,
    select_stratified_tasks,
    summarize_probe_groups,
)
from infoskill.episode import TaskSpec, Trajectory, TrajectoryGroup
from infoskill.integrations.alfworld import ALFWORLD_TASK_TYPES


class RawSkillAbTests(unittest.TestCase):
    def test_probe_matrix_is_the_three_unmeasured_cells(self) -> None:
        self.assertEqual(
            tuple(
                (item.name, item.retrieval_mode, item.prompt_format)
                for item in RAW_SKILL_AB_VARIANTS
            ),
            (
                ("embedding-skillrl", "embedding", "skillrl"),
                ("template-full", "template", "full"),
                ("template-skillrl", "template", "skillrl"),
            ),
        )

    def test_skillrl_rl_exact_is_opt_in_and_resolves_alone(self) -> None:
        self.assertNotIn(SKILLRL_RL_EXACT_VARIANT, RAW_SKILL_AB_VARIANTS)

        selected = resolve_raw_skill_diagnostic_variants(("skillrl-rl-exact",))

        self.assertEqual(selected, (SKILLRL_RL_EXACT_VARIANT,))
        self.assertEqual(
            SKILLRL_RL_EXACT_VARIANT.prompt_format,
            "skillrl_rl_exact",
        )

    def test_skillrl_sft_exact_is_opt_in_and_resolves_alone(self) -> None:
        self.assertNotIn(SKILLRL_SFT_EXACT_VARIANT, RAW_SKILL_AB_VARIANTS)

        selected = resolve_raw_skill_diagnostic_variants(("skillrl-sft-exact",))

        self.assertEqual(selected, (SKILLRL_SFT_EXACT_VARIANT,))
        self.assertEqual(
            SKILLRL_SFT_EXACT_VARIANT.prompt_format,
            "skillrl_sft_exact",
        )

    def test_skillrl_sft_causal_matrix_changes_one_axis_per_cell(self) -> None:
        self.assertEqual(
            tuple(
                (
                    item.name,
                    item.policy_mode,
                    item.prompt_format,
                    item.do_sample,
                    item.temperature,
                )
                for item in SKILLRL_SFT_CAUSAL_VARIANTS
            ),
            (
                (
                    "skillrl-sft-shell-no-skills-deterministic",
                    "raw_skill_prompt",
                    "skillrl_sft_no_skills",
                    False,
                    0.0,
                ),
                (
                    "no-skill-sampled-t0.4",
                    "no_skill",
                    "unified_no_skill",
                    True,
                    0.4,
                ),
                (
                    "skillrl-sft-exact-sampled-t0.4",
                    "raw_skill_prompt",
                    "skillrl_sft_exact",
                    True,
                    0.4,
                ),
            ),
        )
        selected = resolve_raw_skill_diagnostic_variants(
            tuple(item.name for item in SKILLRL_SFT_CAUSAL_VARIANTS)
        )
        self.assertEqual(selected, SKILLRL_SFT_CAUSAL_VARIANTS)
        self.assertTrue(all(item.top_p == 1.0 for item in selected))

    def test_decimal_sampling_name_has_a_trace_safe_slug(self) -> None:
        sampled = next(
            item
            for item in SKILLRL_SFT_CAUSAL_VARIANTS
            if item.name == "no-skill-sampled-t0.4"
        )

        self.assertEqual(sampled.trace_slug, "no-skill-sampled-t0-4")

    def test_unknown_diagnostic_variant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown raw-skill diagnostic"):
            resolve_raw_skill_diagnostic_variants(("missing",))

    def test_stratified_probe_selects_two_stable_tasks_per_type(self) -> None:
        tasks = tuple(
            TaskSpec(
                task_id=f"{task_type}/{index}",
                split="valid_seen",
                task_type=task_type,
                goal=f"goal {index}",
            )
            for task_type in reversed(ALFWORLD_TASK_TYPES)
            for index in (2, 0, 1)
        )

        selected = select_stratified_tasks(tasks, tasks_per_type=2)

        self.assertEqual(len(selected), 12)
        self.assertEqual(Counter(item.task_type for item in selected), {
            task_type: 2 for task_type in ALFWORLD_TASK_TYPES
        })
        self.assertEqual(
            tuple(item.task_type for item in selected[::2]),
            ALFWORLD_TASK_TYPES,
        )
        self.assertTrue(all(item.task_id.endswith(("/0", "/1")) for item in selected))

    def test_stratified_probe_rejects_an_incomplete_category(self) -> None:
        tasks = (
            TaskSpec("only-one", "valid_seen", ALFWORLD_TASK_TYPES[0], "goal"),
        )

        with self.assertRaisesRegex(ValueError, "requires 2 tasks"):
            select_stratified_tasks(tasks, tasks_per_type=2)

    def test_probe_summary_is_explicitly_non_reportable(self) -> None:
        groups = tuple(
            TrajectoryGroup(
                task=task,
                trajectories=(
                    Trajectory(
                        task=task,
                        rollout_id=0,
                        steps=(),
                        won=index == 0,
                        environment_done=True,
                        horizon_exhausted=False,
                        invalid_action_count=0,
                        reward=float(index == 0),
                    ),
                ),
            )
            for task_type in ALFWORLD_TASK_TYPES
            for index, task in enumerate(
                (
                    TaskSpec(
                        task_id=f"{task_type}/won",
                        split="valid_seen",
                        task_type=task_type,
                        goal="goal",
                    ),
                    TaskSpec(
                        task_id=f"{task_type}/lost",
                        split="valid_seen",
                        task_type=task_type,
                        goal="goal",
                    ),
                )
            )
        )

        summary = summarize_probe_groups(groups)

        self.assertTrue(summary["diagnostic_only"])
        self.assertFalse(summary["reportable_as_valid_seen"])
        self.assertEqual(summary["task_count"], 12)
        self.assertEqual(summary["success_count"], 6)
        self.assertEqual(summary["macro_success"], 0.5)
        self.assertEqual(summary["overall_success"], 0.5)


if __name__ == "__main__":
    unittest.main()
