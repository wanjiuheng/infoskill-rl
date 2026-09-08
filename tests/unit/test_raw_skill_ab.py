from __future__ import annotations

import unittest
from collections import Counter
from types import SimpleNamespace

from infoskill.diagnostics.raw_skill_ab import (
    RAW_SKILL_AB_VARIANTS,
    SKILLRL_RL_EXACT_VARIANT,
    SKILLRL_SFT_CAUSAL_VARIANTS,
    SKILLRL_SFT_EXACT_VARIANT,
    UNIFIED_SKILL_CAUSAL_VARIANTS,
    compare_unified_prompt_controls,
    is_unified_skill_causal_matrix,
    resolve_raw_skill_diagnostic_variants,
    select_stratified_tasks,
    summarize_probe_groups,
    validate_unified_skill_causal_gate,
)
from infoskill.episode import TaskSpec, Trajectory, TrajectoryGroup
from infoskill.integrations.alfworld import ALFWORLD_TASK_TYPES


class RawSkillAbTests(unittest.TestCase):
    def test_unified_skill_causal_matrix_changes_only_skill_conditioning(self) -> None:
        self.assertEqual(
            tuple(
                (
                    item.name,
                    item.policy_mode,
                    item.retrieval_mode,
                    item.prompt_format,
                    item.do_sample,
                    item.temperature,
                )
                for item in UNIFIED_SKILL_CAUSAL_VARIANTS
            ),
            (
                (
                    "unified-no-skill-deterministic",
                    "no_skill",
                    None,
                    "unified_no_skill",
                    False,
                    0.0,
                ),
                (
                    "unified-empty-skills-deterministic",
                    "raw_skill_prompt",
                    None,
                    "unified_empty_skills",
                    False,
                    0.0,
                ),
                (
                    "unified-template-skills-deterministic",
                    "raw_skill_prompt",
                    "template",
                    "full",
                    False,
                    0.0,
                ),
                (
                    "unified-embedding-skills-deterministic",
                    "raw_skill_prompt",
                    "embedding",
                    "full",
                    False,
                    0.0,
                ),
            ),
        )
        self.assertEqual(
            resolve_raw_skill_diagnostic_variants(
                tuple(item.name for item in UNIFIED_SKILL_CAUSAL_VARIANTS)
            ),
            UNIFIED_SKILL_CAUSAL_VARIANTS,
        )
        self.assertTrue(
            is_unified_skill_causal_matrix(UNIFIED_SKILL_CAUSAL_VARIANTS)
        )

    def test_unified_skill_causal_gate_locks_order_and_config(self) -> None:
        kwargs = {
            "tasks_per_type": 2,
            "policy_model_id": "alfworld-7b-sft-checkpoint-140",
            "master_seed": 0,
            "history_length": 2,
            "max_steps": 30,
        }

        self.assertTrue(
            validate_unified_skill_causal_gate(
                UNIFIED_SKILL_CAUSAL_VARIANTS,
                **kwargs,
            )
        )
        self.assertFalse(
            is_unified_skill_causal_matrix(
                tuple(reversed(UNIFIED_SKILL_CAUSAL_VARIANTS))
            )
        )
        with self.assertRaisesRegex(ValueError, "registered variant order"):
            validate_unified_skill_causal_gate(
                tuple(reversed(UNIFIED_SKILL_CAUSAL_VARIANTS)),
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "master_seed=0"):
            validate_unified_skill_causal_gate(
                UNIFIED_SKILL_CAUSAL_VARIANTS,
                **{**kwargs, "master_seed": 1},
            )
        with self.assertRaisesRegex(ValueError, "policy_model_id"):
            validate_unified_skill_causal_gate(
                UNIFIED_SKILL_CAUSAL_VARIANTS,
                **{**kwargs, "policy_model_id": "another-model"},
            )

    def test_unified_prompt_control_parity_is_a_runtime_gate(self) -> None:
        task = TaskSpec(
            task_id="task/one",
            split="valid_seen",
            task_type=ALFWORLD_TASK_TYPES[0],
            goal="goal",
        )

        def groups(message: str) -> tuple[TrajectoryGroup, ...]:
            step = SimpleNamespace(
                conditioned_input=SimpleNamespace(
                    user_message=message,
                    candidate_skill_ids=(),
                ),
                generation=SimpleNamespace(
                    prompt_token_count=42,
                    token_ids=(1, 2, 3),
                ),
                action=SimpleNamespace(executed_action="look"),
            )
            trajectory = SimpleNamespace(
                task=task,
                rollout_id=0,
                steps=(step,),
            )
            return (
                SimpleNamespace(task=task, trajectories=(trajectory,)),
            )

        exact = compare_unified_prompt_controls(
            groups("same prompt"),
            groups("same prompt"),
        )
        changed = compare_unified_prompt_controls(
            groups("same prompt"),
            groups("changed prompt"),
        )

        self.assertTrue(exact["passed"])
        self.assertTrue(exact["policy_user_messages_exact"])
        self.assertEqual(exact["compared_step_count"], 1)
        self.assertFalse(changed["passed"])
        self.assertFalse(changed["policy_user_messages_exact"])
        self.assertTrue(changed["mismatches"])

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
