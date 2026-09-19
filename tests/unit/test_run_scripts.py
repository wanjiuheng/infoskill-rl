from __future__ import annotations

import unittest
from pathlib import Path


class RunScriptTests(unittest.TestCase):
    def test_warmstart_handoff_is_forwarded_to_train_and_eval(self) -> None:
        script = Path("scripts/run_alfworld.sh").read_text(encoding="utf-8")
        eval_case = script.split("  eval)\n", 1)[1].split("  raw-skill-ab|", 1)[0]
        train_case = script.split("  train)\n", 1)[1].split("  *)\n", 1)[0]
        self.assertIn('--warmstart-handoff "${WARMSTART_HANDOFF}"', eval_case)
        self.assertIn('--warmstart-handoff "${WARMSTART_HANDOFF}"', train_case)
        self.assertIn('--skill-bank "${SKILL_BANK}"', eval_case)
        self.assertIn('--skill-bank "${SKILL_BANK}"', train_case)
        self.assertIn(
            '--skill-bank-manifest "${SKILL_BANK_MANIFEST}"', eval_case
        )
        self.assertIn(
            '--skill-bank-manifest "${SKILL_BANK_MANIFEST}"', train_case
        )

        imitation = Path("scripts/run_actor_imitation.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"${PYTHON_BIN}" -m torch.distributed.run', imitation)
        self.assertIn('--base-model-id "${BASE_MODEL_ID}"', imitation)
        self.assertIn('--resume-from-checkpoint "${IMITATION_RESUME}"', imitation)

    def test_entrypoint_honors_explicit_python_for_eval_and_train(self) -> None:
        script = Path("scripts/run_alfworld.sh").read_text(encoding="utf-8")

        self.assertIn('PYTHON_BIN="${PYTHON:-python}"', script)
        self.assertIn(
            '"${PYTHON_BIN}" -m infoskill.cli eval "${EVAL_ARGS[@]}"',
            script,
        )
        self.assertIn(
            'exec "${PYTHON_BIN}" -m infoskill.cli train "${TRAIN_ARGS[@]}"',
            script,
        )

    def test_m1_isolation_entry_is_three_gpu_and_audit_only(self) -> None:
        script = Path("scripts/run_alfworld.sh").read_text(encoding="utf-8")
        isolation = script.split("  m1-lora-isolation|m1-lora-boundary)\n", 1)[1].split("    ;;", 1)[0]
        self.assertIn("INFOSKILL_VLLM_INPUT_AUDIT=1", isolation)
        self.assertIn('"${#GPU_IDS[@]}" -ne 3', isolation)
        self.assertIn('"${POLICY_CHECKPOINT}"', isolation)
        self.assertIn('"${PYTHON_BIN}" -m infoskill.cli "${ACTION}"', isolation)
        self.assertIn("INFOSKILL_VLLM_BOUNDARY_AUDIT=1", isolation)

    def test_m1_layer_localization_is_scoped_and_three_gpu_only(self) -> None:
        script = Path("scripts/run_alfworld.sh").read_text(encoding="utf-8")
        block = script.split("  m1-lora-layer-localization)\n", 1)[1].split(
            "    ;;", 1
        )[0]
        self.assertIn("INFOSKILL_VLLM_INPUT_AUDIT=1", block)
        self.assertIn("INFOSKILL_VLLM_BOUNDARY_AUDIT=1", block)
        self.assertIn("INFOSKILL_VLLM_LAYER_AUDIT=1", block)
        self.assertIn('"${#GPU_IDS[@]}" -ne 3', block)
        self.assertIn('"${POLICY_CHECKPOINT}"', block)
        self.assertIn("--detailed-rounds", block)
        self.assertIn("--control-rounds", block)

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
            'exec "${PYTHON_BIN}" -m infoskill.cli train "${TRAIN_ARGS[@]}"',
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

    def test_split_k_one_is_default_off_and_forwarded_to_train_and_eval(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )
        eval_case = script.split("  eval)\n", 1)[1].split("  raw-skill-ab|", 1)[0]
        train_case = script.split("  train)\n", 1)[1].split("  *)\n", 1)[0]

        self.assertIn(
            'LORA_SHRINK_SPLIT_K_ONE="${LORA_SHRINK_SPLIT_K_ONE:-0}"',
            script,
        )
        for action_case in (eval_case, train_case):
            self.assertIn("--lora-shrink-split-k-one", action_case)
            self.assertIn("--no-lora-shrink-split-k-one", action_case)

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
            'CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -m infoskill.cli grounding-planner-loop-diagnostic',
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
            'CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -m infoskill.cli grounding-planner-pilot',
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
            'CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -m infoskill.cli grounding-planner-parity',
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
            'CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -m infoskill.cli grounding-expert-diagnostic',
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
        self.assertIn(
            'ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE:-1e-6}"',
            script,
        )
        self.assertIn(
            '--actor-learning-rate "${ACTOR_LEARNING_RATE}"',
            script,
        )

    def test_m0_lora_lr_sweep_is_one_resumable_paired_job(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (
            project_root / "scripts" / "run_m0_lora_lr_sweep.sh"
        ).read_text(encoding="utf-8")

        for learning_rate in ("1e-6", "3e-6", "1e-5"):
            self.assertIn(learning_rate, script)
        self.assertIn('TARGET_DELTA_UPDATES="${TARGET_DELTA_UPDATES:-5}"', script)
        self.assertIn(
            'LEARNING_RATES_CSV="${LEARNING_RATES_CSV:-1e-6,3e-6,1e-5}"',
            script,
        )
        self.assertIn('ACTOR_LEARNING_RATE="${LEARNING_RATE}"', script)
        self.assertIn('--expected-learning-rates "${LEARNING_RATES[@]}"', script)
        self.assertIn('EVAL_BATCH_SIZE=64', script)
        self.assertIn('--expected-task-count 140', script)
        self.assertIn('trap archive_diagnostics EXIT', script)
        self.assertIn('compare_m0_lora_lr_sweep.py', script)

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

    def test_m1_lora_reproducibility_forwards_bounded_probe_controls(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (project_root / "scripts" / "run_alfworld.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("m1-lora-reproducibility)", script)
        self.assertIn('M1_REPRO_CASE_COUNT="${M1_REPRO_CASE_COUNT:-3}"', script)
        self.assertIn(
            '--max-new-tokens "${M1_REPRO_MAX_NEW_TOKENS}"',
            script,
        )
        self.assertIn(
            '"${PYTHON_BIN}" -m infoskill.cli m1-lora-reproducibility',
            script,
        )
        self.assertIn("export INFOSKILL_VLLM_INPUT_AUDIT=1", script)
        self.assertIn(
            'export INFOSKILL_VLLM_LAYER_AUDIT=1',
            script,
        )
        self.assertIn(
            '--lora-kernel-intervention "${M1_REPRO_LORA_KERNEL_INTERVENTION}"',
            script,
        )
        self.assertIn(
            '1) M1_REPRO_ARGS+=(--lora-shrink-split-k-one)',
            script,
        )

    def test_split_k_fresh_runtime_matrix_is_one_unattended_job(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (
            project_root / "scripts" / "run_m1_splitk1_fresh_runtime_matrix.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("run_cell graph-splitk1 1 1 none", script)
        self.assertIn("run_cell eager-splitk1 0 1 none", script)
        self.assertIn("run_cell eager-reference-full 0 0 reference_full", script)
        self.assertIn("compare_m1_fresh_runtime_matrix.py", script)

    def test_graph_precapture_gate_reuses_stable_eager_controls(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (
            project_root
            / "scripts"
            / "run_m1_splitk1_graph_precapture_gate.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("EAGER_RUN=", script)
        self.assertIn("REFERENCE_RUN=", script)
        self.assertIn("HYBRID_PREFIX_CUDA_GRAPH=1", script)
        self.assertIn("LORA_SHRINK_SPLIT_K_ONE=1", script)
        self.assertIn("compare_m1_fresh_runtime_matrix.py", script)
        self.assertNotIn("eager-splitk1", script)
        self.assertIn("--query-compute-apps=pid,used_memory", script)
        self.assertIn("already has compute processes", script)
        self.assertIn("ARCHIVE=", script)

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
        self.assertIn('CONTROL_RUN="${CONTROL_RUN:-}"', script)
        self.assertIn('reusing gradient-clip control', script)
        self.assertIn('HYBRID_PREFIX_CUDA_GRAPH=1', script)
        self.assertIn('EVAL_BATCH_SIZE=64', script)
        self.assertIn('--candidate-mode separate-grad-clip', script)
        self.assertIn('"infrastructure_gate_passed": all(checks.values())', script)
        self.assertIn('"same_training_workload"', script)
        self.assertIn('exit "${INFRASTRUCTURE_RC}"', script)

    def test_m1_gradient_clip_efficacy_runner_locks_python_and_archives(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (
            project_root / "scripts" / "run_m1_gradient_clip_efficacy.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('export PATH="$(dirname -- "${PYTHON}"):${PATH}"', script)
        self.assertIn('[[ "$(command -v python)" == "${PYTHON}" ]]', script)
        self.assertIn('run_branch "${CONTROL_RUN}" "${CONTROL_UPDATE}" joint', script)
        self.assertIn(
            'run_branch "${CANDIDATE_RUN}" "${CANDIDATE_UPDATE}" separate',
            script,
        )
        self.assertIn('SEGMENT_END_UPDATE="${TARGET_UPDATE}"', script)
        self.assertIn("already at update ${TARGET_UPDATE}", script)
        self.assertIn('run_evaluation "${CONTROL_RUN}" joint', script)
        self.assertIn('run_evaluation "${CANDIDATE_RUN}" separate', script)
        self.assertIn("--expected-task-count 140", script)
        self.assertIn('classification = "candidate_selected"', script)
        self.assertIn('classification = "repeat_required"', script)
        self.assertIn("trap archive_diagnostics EXIT", script)
        self.assertIn("--exclude='*/checkpoints/*/runtime/*'", script)

    def test_m1_gradient_clip_repeat_runner_is_eval_only_and_resumable(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (
            project_root / "scripts" / "run_m1_gradient_clip_efficacy_repeat.sh"
        ).read_text(encoding="utf-8")

        self.assertNotIn("run_alfworld.sh train", script)
        self.assertEqual(script.count("run_alfworld.sh eval infoskill"), 1)
        self.assertIn('run_evaluation "${CONTROL_RUN}" joint', script)
        self.assertIn('run_evaluation "${CANDIDATE_RUN}" separate', script)
        self.assertIn("evaluation_is_reusable", script)
        self.assertIn("compare_infoskill_efficacy_repeats.py", script)
        self.assertIn("--minimum-macro-delta 0.03", script)
        self.assertIn("--minimum-overall-delta 0.0", script)
        self.assertIn("trap archive_diagnostics EXIT", script)

    def test_m1_precapture_learning_gate_is_a_bounded_named_fork(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (
            project_root / "scripts" / "run_m1_precapture_learning_gate.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            '[[ "$(basename -- "${SOURCE_CHECKPOINT}")" == "step-000205" ]]',
            script,
        )
        self.assertIn('SEGMENT_END_UPDATE=215', script)
        self.assertIn('RUN_NAME="${TRAIN_NAME}"', script)
        self.assertIn('LORA_SHRINK_SPLIT_K_ONE=1', script)
        self.assertIn('HYBRID_PREFIX_CUDA_GRAPH=1', script)
        self.assertIn('CHECKPOINT_KEEP_RECENT=5', script)
        self.assertIn('CHECKPOINT_KEEP_BEST_VALID=1', script)
        self.assertIn('for step in 210 215; do', script)
        self.assertIn('EVAL_BATCH_SIZE=64', script)
        self.assertIn('trap archive_diagnostics EXIT', script)
        self.assertIn(
            '"decision_scope": "short_learning_signal_only_not_formal_improvement_proof"',
            script,
        )

    def test_m1_precapture_learning_followup_preserves_protocol(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (
            project_root / "scripts" / "run_m1_precapture_learning_followup.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('"step-000215"', script)
        self.assertIn('SEGMENT_END_UPDATE=225', script)
        self.assertIn('RUN_NAME="${TRAIN_NAME}"', script)
        self.assertIn('HYBRID_PREFIX_CUDA_GRAPH=1', script)
        self.assertIn('LORA_SHRINK_SPLIT_K_ONE=1', script)
        self.assertIn('POLICY_GRADIENT_CLIP_MODE=joint', script)
        self.assertIn('CHECKPOINT_KEEP_RECENT=5', script)
        self.assertIn('CHECKPOINT_KEEP_BEST_VALID=1', script)
        self.assertIn('EVAL_BATCH_SIZE=64', script)
        self.assertIn('CHECKPOINT_STEP=225', script)
        self.assertIn('trap archive_diagnostics EXIT', script)
        self.assertIn('"checkpoint_225_loaded_on_three_ranks"', script)
        self.assertIn('"all_complete_same_140_tasks"', script)

    def test_m1_precapture_clip_ab_reuses_joint_control_and_forks_candidate(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = (
            project_root / "scripts" / "run_m1_precapture_clip_ab.sh"
        ).read_text(encoding="utf-8")

        self.assertEqual(script.count("run_alfworld.sh train infoskill"), 1)
        self.assertEqual(script.count("run_alfworld.sh eval infoskill"), 1)
        self.assertIn('SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"', script)
        self.assertIn('CONTROL_TRAIN="${CONTROL_TRAIN:?CONTROL_TRAIN is required}"', script)
        self.assertIn('POLICY_GRADIENT_CLIP_MODE=separate', script)
        self.assertIn('SEGMENT_END_UPDATE=225', script)
        self.assertIn('HYBRID_PREFIX_CUDA_GRAPH=1', script)
        self.assertIn('LORA_SHRINK_SPLIT_K_ONE=1', script)
        self.assertIn('EVAL_BATCH_SIZE=64', script)
        self.assertIn('CHECKPOINT_KEEP_RECENT=5', script)
        self.assertIn('CHECKPOINT_KEEP_BEST_VALID=1', script)
        self.assertIn('trap archive_diagnostics EXIT', script)
        self.assertIn('scripts/compare_m1_precapture_clip_ab.py', script)


if __name__ == "__main__":
    unittest.main()
