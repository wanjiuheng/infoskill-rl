from __future__ import annotations

import unittest
from pathlib import Path


class RunScriptTests(unittest.TestCase):
    def test_step50_formal_fork_recipe_is_complete(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        bringup = (project_root / "docs" / "SERVER_BRINGUP.md").read_text(
            encoding="utf-8"
        )
        marker = "## M1 step-50 正式分叉：CUDA Graph、batch 64 和有界 checkpoint"
        self.assertIn(marker, bringup)
        recipe = bringup.split(marker, 1)[1].split("\n## ", 1)[0]

        self.assertIn("checkpoints/step-000050", recipe)
        self.assertIn('RESUME="$SOURCE"', recipe)
        self.assertIn("EVAL_BATCH_SIZE=64", recipe)
        self.assertIn("HYBRID_PREFIX_CUDA_GRAPH=1", recipe)
        self.assertIn("CHECKPOINT_KEEP_RECENT=5", recipe)
        self.assertIn("CHECKPOINT_KEEP_BEST_VALID=1", recipe)
        self.assertIn(
            "RUN_NAME=m1-infoskill-formal-s50-cudagraph-b64-retained",
            recipe,
        )

    def test_training_executes_python_so_background_pid_receives_pause_signal(
        self,
    ) -> None:
        script = Path("scripts/run_alfworld.sh").read_text(encoding="utf-8")

        self.assertIn(
            'exec python -m infoskill.cli train "${TRAIN_ARGS[@]}"',
            script,
        )

    def test_eval_batch_gate_is_fixed_non_reportable_and_fail_closed(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        runner = (project_root / "scripts" / "run_infoskill_eval_batch_gate.sh").read_text(
            encoding="utf-8"
        )
        entrypoint = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("m1_eval_batch_pressure_valid_seen.json", runner)
        self.assertIn('BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-8}"', runner)
        self.assertIn('CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-12}"', runner)
        self.assertIn('MINIMUM_ROLLOUT_SPEEDUP="${MINIMUM_ROLLOUT_SPEEDUP:-1.10}"', runner)
        self.assertIn('MINIMUM_PHYSICAL_FREE_GB="${MINIMUM_PHYSICAL_FREE_GB:-8.0}"', runner)
        self.assertIn("if (( GATE_RC != 0 )); then", runner)
        self.assertIn('RUN_FULL_EVAL_ON_PASS="${RUN_FULL_EVAL_ON_PASS:-1}"', runner)
        self.assertIn("--diagnostic-task-manifest", entrypoint)
        self.assertIn("--eval-batch-size", entrypoint)
        self.assertIn("--cuda-memory-poll-interval-ms", entrypoint)

    def test_training_forwards_periodic_evaluation_batch_override(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )
        train_case = script.split("  train)\n", 1)[1].split("  *)\n", 1)[0]

        self.assertIn('if [[ -n "${EVAL_BATCH_SIZE}" ]]', train_case)
        self.assertIn(
            'TRAIN_ARGS+=(--eval-batch-size "${EVAL_BATCH_SIZE}")',
            train_case,
        )

    def test_training_forwards_checkpoint_retention_policy(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )
        train_case = script.split("  train)\n", 1)[1].split("  *)\n", 1)[0]

        self.assertIn(
            'CHECKPOINT_KEEP_RECENT="${CHECKPOINT_KEEP_RECENT:-2}"',
            script,
        )
        self.assertIn(
            'CHECKPOINT_KEEP_BEST_VALID="${CHECKPOINT_KEEP_BEST_VALID:-0}"',
            script,
        )
        self.assertIn(
            '--checkpoint-keep-recent "${CHECKPOINT_KEEP_RECENT}"',
            train_case,
        )
        self.assertIn("--checkpoint-keep-best-valid", train_case)

    def test_deep_m1_candidates_are_default_off_and_forwarded(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )
        train_case = script.split("  train)\n", 1)[1].split("  *)\n", 1)[0]

        self.assertIn(
            'HYBRID_PREFIX_CUDA_GRAPH="${HYBRID_PREFIX_CUDA_GRAPH:-0}"',
            script,
        )
        self.assertIn(
            'FUSE_KL_PPO_FORWARD="${FUSE_KL_PPO_FORWARD:-0}"',
            script,
        )
        self.assertIn(
            'POLICY_GRADIENT_CLIP_MODE="${POLICY_GRADIENT_CLIP_MODE:-joint}"',
            script,
        )
        self.assertIn("--hybrid-prefix-cuda-graph", train_case)
        self.assertIn("--no-hybrid-prefix-cuda-graph", train_case)
        self.assertIn("--fuse-kl-ppo-forward", train_case)
        self.assertIn("--no-fuse-kl-ppo-forward", train_case)
        self.assertIn(
            '--policy-gradient-clip-mode "${POLICY_GRADIENT_CLIP_MODE}"',
            train_case,
        )

    def test_grouped_infoskill_conditioning_is_eval_only_and_opt_in(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'GROUPED_INFOSKILL_CONDITIONING="${GROUPED_INFOSKILL_CONDITIONING:-0}"',
            script,
        )
        eval_case = script.split("  eval)\n", 1)[1].split("  raw-skill-ab|", 1)[0]
        train_case = script.split("  train)\n", 1)[1].split("  *)\n", 1)[0]
        self.assertIn("--grouped-infoskill-conditioning", eval_case)
        self.assertNotIn("grouped-infoskill-conditioning", train_case)

    def test_eval_forwards_hybrid_prefix_cuda_graph(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )
        eval_case = script.split("  eval)\n", 1)[1].split("  raw-skill-ab|", 1)[0]

        self.assertIn("--hybrid-prefix-cuda-graph", eval_case)
        self.assertIn("--no-hybrid-prefix-cuda-graph", eval_case)

    def test_timeout_rescue_is_targeted_resumable_and_longer_bounded(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'GROUNDING_RESCUE_WORKER_PROCESSES="${GROUNDING_RESCUE_WORKER_PROCESSES:-4}"',
            script,
        )
        self.assertIn(
            'GROUNDING_RESCUE_TIMEOUT_SECONDS="${GROUNDING_RESCUE_TIMEOUT_SECONDS:-600}"',
            script,
        )
        self.assertIn("grounding-timeout-rescue)", script)
        self.assertIn(
            '--source-grounding-run "${GROUNDING_SOURCE_RUN}"',
            script,
        )
        self.assertIn(
            'GROUNDING_RESCUE_ARGS+=(--resume-run "${GROUNDING_RESUME_RUN}")',
            script,
        )
        self.assertIn(
            'GROUNDING_RESCUE_ARGS+=(--finalize-committed-rescue-run "${GROUNDING_RESCUE_FINALIZE_RUN}")',
            script,
        )
        grounding_case = script.split("  grounding)\n", 1)[1].split("    ;;", 1)[0]
        rescue_case = script.split("  grounding-timeout-rescue)\n", 1)[1].split(
            "    ;;", 1
        )[0]
        self.assertNotIn("GROUNDING_RESCUE_FINALIZE_RUN", grounding_case)
        self.assertIn("GROUNDING_RESCUE_FINALIZE_RUN", rescue_case)
        self.assertIn("--finalize-committed-rescue-run", rescue_case)

    def test_planner_loop_diagnostic_is_cpu_only_and_uses_source_pilot(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'PLANNER_LOOP_SUCCESS_CONTROLS="${PLANNER_LOOP_SUCCESS_CONTROLS:-6}"',
            script,
        )
        self.assertIn(
            'PLANNER_LOOP_MAX_REPLAY_STEPS="${PLANNER_LOOP_MAX_REPLAY_STEPS:-300}"',
            script,
        )
        self.assertIn("grounding-planner-loop-diagnostic)", script)
        self.assertIn(
            'CUDA_VISIBLE_DEVICES="" python -m infoskill.cli grounding-planner-loop-diagnostic',
            script,
        )
        self.assertIn('--source-pilot-run "${GROUNDING_SOURCE_RUN}"', script)

    def test_planner_pilot_is_cpu_only_balanced_and_bounded(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'PLANNER_PILOT_TASKS_PER_TYPE="${PLANNER_PILOT_TASKS_PER_TYPE:-50}"',
            script,
        )
        self.assertIn(
            'PLANNER_PILOT_MAX_REPLAY_STEPS="${PLANNER_PILOT_MAX_REPLAY_STEPS:-150}"',
            script,
        )
        self.assertIn("grounding-planner-pilot)", script)
        self.assertIn(
            'CUDA_VISIBLE_DEVICES="" python -m infoskill.cli grounding-planner-pilot',
            script,
        )
        self.assertIn(
            '--tasks-per-type "${PLANNER_PILOT_TASKS_PER_TYPE}"',
            script,
        )
        self.assertIn(
            '--worker-batch-size "${GROUNDING_WORKER_BATCH_SIZE}"',
            script,
        )
        self.assertIn(
            '--worker-processes "${GROUNDING_WORKER_PROCESSES}"',
            script,
        )

    def test_planner_parallelism_has_an_exact_serial_parity_gate(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'GROUNDING_WORKER_PROCESSES="${GROUNDING_WORKER_PROCESSES:-1}"',
            script,
        )
        self.assertIn(
            'GROUNDING_PARITY_PARALLEL_WORKERS="${GROUNDING_PARITY_PARALLEL_WORKERS:-2}"',
            script,
        )
        self.assertIn("grounding-planner-parity)", script)
        self.assertIn(
            'CUDA_VISIBLE_DEVICES="" python -m infoskill.cli grounding-planner-parity',
            script,
        )
        self.assertIn(
            '--parallel-workers "${GROUNDING_PARITY_PARALLEL_WORKERS}"',
            script,
        )
        self.assertIn(
            '--candidate-backend "${GROUNDING_PARITY_CANDIDATE_BACKEND}"',
            script,
        )
        self.assertIn(
            '--native-batch-size "${GROUNDING_NATIVE_BATCH_SIZE}"',
            script,
        )
        self.assertIn(
            '--minimum-speedup "${GROUNDING_PARITY_MINIMUM_SPEEDUP}"',
            script,
        )
        self.assertIn("native_batch_parallel", script)

    def test_grounding_uses_bounded_worker_batches_by_default(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'GROUNDING_WORKER_BATCH_SIZE="${GROUNDING_WORKER_BATCH_SIZE:-64}"',
            script,
        )
        self.assertIn(
            '--worker-batch-size "${GROUNDING_WORKER_BATCH_SIZE}"',
            script,
        )
        self.assertIn(
            'GROUNDING_WORKER_INACTIVITY_TIMEOUT_SECONDS="${GROUNDING_WORKER_INACTIVITY_TIMEOUT_SECONDS:-300}"',
            script,
        )
        self.assertIn(
            '--worker-inactivity-timeout-seconds "${GROUNDING_WORKER_INACTIVITY_TIMEOUT_SECONDS}"',
            script,
        )
        self.assertIn(
            'GROUNDING_ARGS+=(--resume-run "${GROUNDING_RESUME_RUN}")',
            script,
        )

    def test_grounding_expert_diagnostic_is_cpu_only_and_stratified(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'GROUNDING_DIAGNOSTIC_TASKS_PER_TYPE="${GROUNDING_DIAGNOSTIC_TASKS_PER_TYPE:-3}"',
            script,
        )
        self.assertIn(
            'GROUNDING_DIAGNOSTIC_MAX_REPLAY_STEPS="${GROUNDING_DIAGNOSTIC_MAX_REPLAY_STEPS:-150}"',
            script,
        )
        self.assertIn("grounding-expert-diagnostic)", script)
        self.assertIn(
            'CUDA_VISIBLE_DEVICES="" python -m infoskill.cli grounding-expert-diagnostic',
            script,
        )
        self.assertIn('--source-grounding-run "${GROUNDING_SOURCE_RUN}"', script)
        self.assertIn(
            '--max-replay-steps "${GROUNDING_DIAGNOSTIC_MAX_REPLAY_STEPS}"',
            script,
        )

    def test_safe_policy_token_budget_is_the_shell_default(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'POLICY_MAX_TOKENS_PER_GPU="${POLICY_MAX_TOKENS_PER_GPU:-12288}"',
            script,
        )
        self.assertIn(
            'SKIP_UNUSED_OLD_LOGPROB_ENTROPY="${SKIP_UNUSED_OLD_LOGPROB_ENTROPY:-0}"',
            script,
        )
        self.assertIn(
            'ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-16384}"',
            script,
        )
        self.assertIn('SEGMENT_END_UPDATE="${SEGMENT_END_UPDATE:-}"', script)
        self.assertIn('--segment-end-update "${SEGMENT_END_UPDATE}"', script)

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

    def test_m1_gradient_clip_gate_runs_control_and_candidate_sequentially(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_m1_gradient_clip_gate.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('run_case "${CONTROL_NAME}" joint', script)
        self.assertIn('run_case "${CANDIDATE_NAME}" separate', script)
        self.assertIn('HYBRID_PREFIX_CUDA_GRAPH=1', script)
        self.assertIn('EVAL_BATCH_SIZE=64', script)
        self.assertIn('--candidate-mode separate-grad-clip', script)
        self.assertIn('"infrastructure_gate_passed": all(checks.values())', script)


if __name__ == "__main__":
    unittest.main()
