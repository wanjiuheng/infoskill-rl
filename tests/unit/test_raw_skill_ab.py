from __future__ import annotations

import unittest
from collections import Counter

from infoskill.diagnostics.raw_skill_ab import (
    RAW_SKILL_AB_VARIANTS,
    SKILLRL_RL_EXACT_VARIANT,
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
