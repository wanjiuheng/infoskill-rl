from __future__ import annotations

import unittest
from pathlib import Path


class RunScriptTests(unittest.TestCase):
    def test_safe_policy_token_budget_is_the_shell_default(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'POLICY_MAX_TOKENS_PER_GPU="${POLICY_MAX_TOKENS_PER_GPU:-12288}"',
            script,
        )

    def test_m0_pair_explicitly_clears_checkpoint_for_update_zero(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_m0_valid_seen_pair.sh").read_text(
            encoding="utf-8"
        )
        base_invocation, checkpoint_invocation = script.split(
            'echo "[INFO-SKILL] paired valid_seen evaluation: portable checkpoint"'
        )

        self.assertIn('POLICY_CHECKPOINT=""', base_invocation)
        self.assertIn(
            'POLICY_CHECKPOINT="${POLICY_CHECKPOINT}"',
            checkpoint_invocation,
        )

    def test_checkpoint_effect_action_requires_checkpoint_and_forwards_probe_cap(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("checkpoint-effect)", script)
        self.assertIn('if [[ -z "${POLICY_CHECKPOINT}" ]]', script)
        self.assertIn(
            '--max-new-tokens "${CHECKPOINT_EFFECT_MAX_NEW_TOKENS}"',
            script,
        )

    def test_retrieval_mode_override_reaches_training_and_evaluation(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('RETRIEVAL_MODE="${RETRIEVAL_MODE:-}"', script)
        self.assertIn(
            'RETRIEVAL_ARGS+=(--retrieval-mode "${RETRIEVAL_MODE}")',
            script,
        )
        self.assertEqual(script.count('"${RETRIEVAL_ARGS[@]}"'), 3)

    def test_full_raw_skill_format_is_default_with_compact_ablation(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'RAW_SKILL_PROMPT_FORMAT="${RAW_SKILL_PROMPT_FORMAT:-full}"',
            script,
        )
        self.assertEqual(
            script.count(
                '--raw-skill-prompt-format "${RAW_SKILL_PROMPT_FORMAT}"'
            ),
            2,
        )
        self.assertIn(
            "RAW_SKILL_PROMPT_FORMAT must be compact or full",
            script,
        )

    def test_raw_skill_ab_action_forwards_the_stratified_probe_size(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "raw-skill-ab|unified-skill-causal|skillrl-rl-exact|"
            "skillrl-sft-exact|skillrl-sft-causal)",
            script,
        )
        self.assertIn(
            '--tasks-per-type "${RAW_SKILL_AB_TASKS_PER_TYPE}"',
            script,
        )

    def test_skillrl_rl_exact_action_selects_only_its_diagnostic_variant(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "raw-skill-ab|unified-skill-causal|skillrl-rl-exact|"
            "skillrl-sft-exact|skillrl-sft-causal)",
            script,
        )
        self.assertIn(
            'RAW_SKILL_AB_ARGS+=(--variants skillrl-rl-exact)',
            script,
        )

    def test_skillrl_sft_exact_action_selects_only_its_diagnostic_variant(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "raw-skill-ab|unified-skill-causal|skillrl-rl-exact|"
            "skillrl-sft-exact|skillrl-sft-causal)",
            script,
        )
        self.assertIn(
            'RAW_SKILL_AB_ARGS+=(--variants skillrl-sft-exact)',
            script,
        )

    def test_skillrl_sft_causal_action_selects_three_causal_variants(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("skillrl-sft-causal", script)
        for variant in (
            "skillrl-sft-shell-no-skills-deterministic",
            "no-skill-sampled-t0.4",
            "skillrl-sft-exact-sampled-t0.4",
        ):
            self.assertIn(variant, script)

    def test_unified_skill_causal_action_selects_four_controlled_variants(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("unified-skill-causal", script)
        for variant in (
            "unified-no-skill-deterministic",
            "unified-empty-skills-deterministic",
            "unified-template-skills-deterministic",
            "unified-embedding-skills-deterministic",
        ):
            self.assertIn(variant, script)
        self.assertIn(
            '"${ACTION}" == "unified-skill-causal" '
            '&& "${RAW_SKILL_AB_TASKS_PER_TYPE}" != "2"',
            script,
        )
        self.assertIn(
            "unified-skill-causal requires exactly 2 tasks per task type",
            script,
        )


if __name__ == "__main__":
    unittest.main()
