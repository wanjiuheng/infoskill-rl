from __future__ import annotations

import unittest
from pathlib import Path


class RunScriptTests(unittest.TestCase):
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

    def test_raw_skill_ab_action_forwards_the_stratified_probe_size(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "raw-skill-ab|skillrl-rl-exact|skillrl-sft-exact|skillrl-sft-causal)",
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
            "raw-skill-ab|skillrl-rl-exact|skillrl-sft-exact|skillrl-sft-causal)",
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
            "raw-skill-ab|skillrl-rl-exact|skillrl-sft-exact|skillrl-sft-causal)",
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


if __name__ == "__main__":
    unittest.main()
