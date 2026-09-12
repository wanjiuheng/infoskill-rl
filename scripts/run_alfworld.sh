#!/usr/bin/env bash
set -euo pipefail

# Central parameter panel. Every value can also be overridden as an environment variable.
ACTION="${ACTION:-${1:-eval}}"                 # validate | eval | diagnostics | grounding | train
MODE="${MODE:-${2:-no_skill}}"                # no_skill | raw_skill_prompt | infoskill
CONFIG="${CONFIG:-${3:-configs/alfworld_qwen25_7b.yaml}}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-}"          # empty=YAML default; embedding | template
# Registered raw control uses full; compact remains available as a measured ablation.
RAW_SKILL_PROMPT_FORMAT="${RAW_SKILL_PROMPT_FORMAT:-full}" # full | compact
GPUS="${GPUS:-${4:-0}}"                       # examples: 0 or 0,1 or 0,1,2,3
RUN_NAME="${RUN_NAME:-}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-0}"
EVAL_BACKEND="${EVAL_BACKEND:-transformers}" # transformers | verl
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-}"
CHECKPOINT_EFFECT_MAX_NEW_TOKENS="${CHECKPOINT_EFFECT_MAX_NEW_TOKENS:-64}"
PROFILE="${PROFILE:-smoke}"                   # smoke | integration | benchmark | pilot | formal
MAX_UPDATES="${MAX_UPDATES:-}"
RESUME="${RESUME:-}"
GROUNDING_DATA="${GROUNDING_DATA:-}"          # M1: completed train-only grounding run
# Short-lived process boundary for TextWorld/Fast Downward resource cleanup.
GROUNDING_WORKER_BATCH_SIZE="${GROUNDING_WORKER_BATCH_SIZE:-64}"
# Number of independent bounded grounding subprocesses allowed concurrently.
# Default 1 preserves historical serial behavior until parity is verified.
GROUNDING_WORKER_PROCESSES="${GROUNDING_WORKER_PROCESSES:-1}"
# Candidate replay engine. Formal default remains individual until exact parity
# and the 60-task lifecycle/performance gate pass on the target server.
GROUNDING_REPLAY_BACKEND="${GROUNDING_REPLAY_BACKEND:-individual}" # individual | native_batch
GROUNDING_NATIVE_BATCH_SIZE="${GROUNDING_NATIVE_BATCH_SIZE:-4}"
# A completed task is the heartbeat. A stuck planner is killed and retried alone.
GROUNDING_WORKER_INACTIVITY_TIMEOUT_SECONDS="${GROUNDING_WORKER_INACTIVITY_TIMEOUT_SECONDS:-300}"
# Existing post-fix run directory whose committed shards should be reused.
GROUNDING_RESUME_RUN="${GROUNDING_RESUME_RUN:-}"
# CPU-only strict handcoded/planner comparison against an existing grounding run.
GROUNDING_SOURCE_RUN="${GROUNDING_SOURCE_RUN:-}"
GROUNDING_DIAGNOSTIC_TASKS_PER_TYPE="${GROUNDING_DIAGNOSTIC_TASKS_PER_TYPE:-3}"
GROUNDING_DIAGNOSTIC_MAX_REPLAY_STEPS="${GROUNDING_DIAGNOSTIC_MAX_REPLAY_STEPS:-150}"
# Balanced CPU-only planner candidate pilot; never produces formal M1 labels.
PLANNER_PILOT_TASKS_PER_TYPE="${PLANNER_PILOT_TASKS_PER_TYPE:-50}"
PLANNER_PILOT_MAX_REPLAY_STEPS="${PLANNER_PILOT_MAX_REPLAY_STEPS:-150}"
# Fixed serial/parallel differential gate before enabling pilot concurrency.
GROUNDING_PARITY_TASKS_PER_TYPE="${GROUNDING_PARITY_TASKS_PER_TYPE:-2}"
GROUNDING_PARITY_WORKER_BATCH_SIZE="${GROUNDING_PARITY_WORKER_BATCH_SIZE:-6}"
GROUNDING_PARITY_PARALLEL_WORKERS="${GROUNDING_PARITY_PARALLEL_WORKERS:-2}"
GROUNDING_PARITY_CANDIDATE_BACKEND="${GROUNDING_PARITY_CANDIDATE_BACKEND:-process_parallel}"
GROUNDING_PARITY_MINIMUM_SPEEDUP="${GROUNDING_PARITY_MINIMUM_SPEEDUP:-0.0}"
# CPU-only follow-up over planner-pilot failures and long successful controls.
PLANNER_LOOP_SUCCESS_CONTROLS="${PLANNER_LOOP_SUCCESS_CONTROLS:-6}"
PLANNER_LOOP_MAX_REPLAY_STEPS="${PLANNER_LOOP_MAX_REPLAY_STEPS:-300}"
DRY_RUN="${DRY_RUN:-0}"
# Validated by exact semantic/token/logprob A/B parity; set to 0 for rollback.
PERSISTENT_ROLLOUT_SESSION="${PERSISTENT_ROLLOUT_SESSION:-1}"
# Used only by the individual backend; native_batch owns one process per slot.
ENVIRONMENT_WORKERS="${ENVIRONMENT_WORKERS:-1}"
# Validated by CPU differential, 64-trajectory A/B and two-update longevity gates.
ENVIRONMENT_BACKEND="${ENVIRONMENT_BACKEND:-native_batch}" # native_batch | individual (rollback)
# Keep BLAS/OpenMP from multiplying threads inside each environment worker.
INFO_SKILL_CPU_THREADS="${INFO_SKILL_CPU_THREADS:-1}"
# Default keeps only INFO-SKILL milestones, errors and progress bars in terminal.
VERBOSE_RUNTIME_LOGS="${VERBOSE_RUNTIME_LOGS:-0}"
# Diagnostic switch. Long runs should use 1000ms; zero explicitly disables it.
CUDA_MEMORY_POLL_INTERVAL_MS="${CUDA_MEMORY_POLL_INTERVAL_MS:-0}"
# Dynamic old/ref/actor micro-batch budget. This does not alter vLLM rollout
# scheduling. The validated cross-mode default preserves physical headroom.
POLICY_MAX_TOKENS_PER_GPU="${POLICY_MAX_TOKENS_PER_GPU:-12288}"
# Validated default. Reassigns samples among ranks while preserving each
# global GRPO minibatch's membership; set to 0 for rollback.
BALANCE_POLICY_TOKENS_ACROSS_RANKS="${BALANCE_POLICY_TOKENS_ACROSS_RANKS:-1}"
# Small, explicitly non-reportable raw-skill diagnostic subset per task type.
RAW_SKILL_AB_TASKS_PER_TYPE="${RAW_SKILL_AB_TASKS_PER_TYPE:-2}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
if [[ ! "${INFO_SKILL_CPU_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INFO_SKILL_CPU_THREADS must be a positive integer" >&2
  exit 2
fi
if [[ "${ENVIRONMENT_BACKEND}" != "individual" && "${ENVIRONMENT_BACKEND}" != "native_batch" ]]; then
  echo "ENVIRONMENT_BACKEND must be individual or native_batch" >&2
  exit 2
fi
if [[ "${ENVIRONMENT_BACKEND}" == "native_batch" && "${ENVIRONMENT_WORKERS}" != "1" ]]; then
  echo "native_batch owns its process count; ENVIRONMENT_WORKERS must remain 1" >&2
  exit 2
fi
if [[ "${RAW_SKILL_PROMPT_FORMAT}" != "compact" && "${RAW_SKILL_PROMPT_FORMAT}" != "full" ]]; then
  echo "RAW_SKILL_PROMPT_FORMAT must be compact or full" >&2
  exit 2
fi
if [[ ! "${CUDA_MEMORY_POLL_INTERVAL_MS}" =~ ^[0-9]+$ ]]; then
  echo "CUDA_MEMORY_POLL_INTERVAL_MS must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${POLICY_MAX_TOKENS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLICY_MAX_TOKENS_PER_GPU must be a positive integer" >&2
  exit 2
fi
if [[ "${BALANCE_POLICY_TOKENS_ACROSS_RANKS}" != "0" && "${BALANCE_POLICY_TOKENS_ACROSS_RANKS}" != "1" ]]; then
  echo "BALANCE_POLICY_TOKENS_ACROSS_RANKS must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${RAW_SKILL_AB_TASKS_PER_TYPE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RAW_SKILL_AB_TASKS_PER_TYPE must be a positive integer" >&2
  exit 2
fi
if [[ ! "${GROUNDING_WORKER_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GROUNDING_WORKER_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ ! "${GROUNDING_WORKER_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GROUNDING_WORKER_PROCESSES must be a positive integer" >&2
  exit 2
fi
if [[ "${GROUNDING_REPLAY_BACKEND}" != "individual" && "${GROUNDING_REPLAY_BACKEND}" != "native_batch" ]]; then
  echo "GROUNDING_REPLAY_BACKEND must be individual or native_batch" >&2
  exit 2
fi
if [[ ! "${GROUNDING_NATIVE_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GROUNDING_NATIVE_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ ! "${GROUNDING_WORKER_INACTIVITY_TIMEOUT_SECONDS}" =~ ^([1-9][0-9]*([.][0-9]+)?|0[.][0-9]*[1-9][0-9]*)$ ]]; then
  echo "GROUNDING_WORKER_INACTIVITY_TIMEOUT_SECONDS must be positive" >&2
  exit 2
fi
if [[ "${GROUNDING_REPLAY_BACKEND}" == "native_batch" ]] && (( GROUNDING_NATIVE_BATCH_SIZE < 2 )); then
  echo "native_batch requires GROUNDING_NATIVE_BATCH_SIZE of at least 2" >&2
  exit 2
fi
if [[ "${GROUNDING_PARITY_CANDIDATE_BACKEND}" != "process_parallel" && "${GROUNDING_PARITY_CANDIDATE_BACKEND}" != "native_batch" ]]; then
  echo "GROUNDING_PARITY_CANDIDATE_BACKEND must be process_parallel or native_batch" >&2
  exit 2
fi
if [[ ! "${GROUNDING_PARITY_MINIMUM_SPEEDUP}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "GROUNDING_PARITY_MINIMUM_SPEEDUP must be a non-negative number" >&2
  exit 2
fi
if [[ ! "${GROUNDING_DIAGNOSTIC_TASKS_PER_TYPE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GROUNDING_DIAGNOSTIC_TASKS_PER_TYPE must be a positive integer" >&2
  exit 2
fi
if [[ ! "${GROUNDING_DIAGNOSTIC_MAX_REPLAY_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GROUNDING_DIAGNOSTIC_MAX_REPLAY_STEPS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${PLANNER_PILOT_TASKS_PER_TYPE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PLANNER_PILOT_TASKS_PER_TYPE must be a positive integer" >&2
  exit 2
fi
if [[ ! "${PLANNER_PILOT_MAX_REPLAY_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PLANNER_PILOT_MAX_REPLAY_STEPS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${GROUNDING_PARITY_TASKS_PER_TYPE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GROUNDING_PARITY_TASKS_PER_TYPE must be a positive integer" >&2
  exit 2
fi
if [[ ! "${GROUNDING_PARITY_WORKER_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GROUNDING_PARITY_WORKER_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ ! "${GROUNDING_PARITY_PARALLEL_WORKERS}" =~ ^[0-9]+$ ]] \
  || (( GROUNDING_PARITY_PARALLEL_WORKERS < 2 )); then
  echo "GROUNDING_PARITY_PARALLEL_WORKERS must be an integer of at least 2" >&2
  exit 2
fi
if [[ ! "${PLANNER_LOOP_SUCCESS_CONTROLS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PLANNER_LOOP_SUCCESS_CONTROLS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${PLANNER_LOOP_MAX_REPLAY_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PLANNER_LOOP_MAX_REPLAY_STEPS must be a positive integer" >&2
  exit 2
fi
export OMP_NUM_THREADS="${INFO_SKILL_CPU_THREADS}"
export MKL_NUM_THREADS="${INFO_SKILL_CPU_THREADS}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"

EXTRA_ARGS=()
if [[ -n "${RUN_NAME}" ]]; then
  EXTRA_ARGS+=(--run-name "${RUN_NAME}")
fi

RETRIEVAL_ARGS=()
if [[ -n "${RETRIEVAL_MODE}" ]]; then
  case "${RETRIEVAL_MODE}" in
    embedding|template) RETRIEVAL_ARGS+=(--retrieval-mode "${RETRIEVAL_MODE}") ;;
    *)
      echo "RETRIEVAL_MODE must be embedding, template, or empty" >&2
      exit 2
      ;;
  esac
fi

echo "[INFO-SKILL] action=${ACTION} mode=${MODE} gpus=${GPUS} config=${CONFIG}"

case "${ACTION}" in
  validate)
    python -m infoskill.cli validate --config "${CONFIG}" --mode "${MODE}" "${RETRIEVAL_ARGS[@]}"
    ;;
  eval)
    IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
    if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
      echo "GPUS must contain at least one physical GPU index" >&2
      exit 2
    fi
    for gpu_id in "${GPU_IDS[@]}"; do
      if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU index in GPUS=${GPUS}: ${gpu_id}" >&2
        exit 2
      fi
    done
    EVAL_ARGS=(
      --config "${CONFIG}"
      --mode "${MODE}"
      "${RETRIEVAL_ARGS[@]}"
      --raw-skill-prompt-format "${RAW_SKILL_PROMPT_FORMAT}"
      --checkpoint-step "${CHECKPOINT_STEP}"
      --backend "${EVAL_BACKEND}"
      --num-gpus "${#GPU_IDS[@]}"
      --environment-backend "${ENVIRONMENT_BACKEND}"
    )
    if [[ -n "${RUN_NAME}" ]]; then
      EVAL_ARGS+=(--run-name "${RUN_NAME}")
    fi
    if [[ -n "${POLICY_CHECKPOINT}" ]]; then
      EVAL_ARGS+=(--policy-checkpoint "${POLICY_CHECKPOINT}")
    fi
    case "${PERSISTENT_ROLLOUT_SESSION}" in
      1) EVAL_ARGS+=(--persistent-rollout-session) ;;
      0) EVAL_ARGS+=(--no-persistent-rollout-session) ;;
      *)
        echo "PERSISTENT_ROLLOUT_SESSION must be 0 or 1" >&2
        exit 2
        ;;
    esac
    case "${VERBOSE_RUNTIME_LOGS}" in
      0) ;;
      1) EVAL_ARGS+=(--verbose-runtime-logs) ;;
      *)
        echo "VERBOSE_RUNTIME_LOGS must be 0 or 1" >&2
        exit 2
        ;;
    esac
    python -m infoskill.cli eval "${EVAL_ARGS[@]}"
    ;;
  raw-skill-ab|unified-skill-causal|skillrl-rl-exact|skillrl-sft-exact|skillrl-sft-causal)
    IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
    if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
      echo "GPUS must contain at least one physical GPU index" >&2
      exit 2
    fi
    for gpu_id in "${GPU_IDS[@]}"; do
      if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU index in GPUS=${GPUS}: ${gpu_id}" >&2
        exit 2
      fi
    done
    if [[ "${ACTION}" == "unified-skill-causal" && "${RAW_SKILL_AB_TASKS_PER_TYPE}" != "2" ]]; then
      echo "unified-skill-causal requires exactly 2 tasks per task type" >&2
      exit 2
    fi
    RAW_SKILL_AB_ARGS=(
      --config "${CONFIG}"
      --num-gpus "${#GPU_IDS[@]}"
      --tasks-per-type "${RAW_SKILL_AB_TASKS_PER_TYPE}"
      --environment-backend "${ENVIRONMENT_BACKEND}"
    )
    if [[ "${ACTION}" == "skillrl-rl-exact" ]]; then
      RAW_SKILL_AB_ARGS+=(--variants skillrl-rl-exact)
    elif [[ "${ACTION}" == "skillrl-sft-exact" ]]; then
      RAW_SKILL_AB_ARGS+=(--variants skillrl-sft-exact)
    elif [[ "${ACTION}" == "skillrl-sft-causal" ]]; then
      RAW_SKILL_AB_ARGS+=(
        --variants
        skillrl-sft-shell-no-skills-deterministic
        no-skill-sampled-t0.4
        skillrl-sft-exact-sampled-t0.4
      )
    elif [[ "${ACTION}" == "unified-skill-causal" ]]; then
      RAW_SKILL_AB_ARGS+=(
        --variants
        unified-no-skill-deterministic
        unified-empty-skills-deterministic
        unified-template-skills-deterministic
        unified-embedding-skills-deterministic
      )
    fi
    if [[ -n "${RUN_NAME}" ]]; then
      RAW_SKILL_AB_ARGS+=(--run-name "${RUN_NAME}")
    fi
    case "${PERSISTENT_ROLLOUT_SESSION}" in
      1) RAW_SKILL_AB_ARGS+=(--persistent-rollout-session) ;;
      0) RAW_SKILL_AB_ARGS+=(--no-persistent-rollout-session) ;;
      *)
        echo "PERSISTENT_ROLLOUT_SESSION must be 0 or 1" >&2
        exit 2
        ;;
    esac
    case "${VERBOSE_RUNTIME_LOGS}" in
      0) ;;
      1) RAW_SKILL_AB_ARGS+=(--verbose-runtime-logs) ;;
      *)
        echo "VERBOSE_RUNTIME_LOGS must be 0 or 1" >&2
        exit 2
        ;;
    esac
    python -m infoskill.cli raw-skill-ab "${RAW_SKILL_AB_ARGS[@]}"
    ;;
  checkpoint-effect)
    IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
    if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
      echo "GPUS must contain at least one physical GPU index" >&2
      exit 2
    fi
    for gpu_id in "${GPU_IDS[@]}"; do
      if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU index in GPUS=${GPUS}: ${gpu_id}" >&2
        exit 2
      fi
    done
    if [[ -z "${POLICY_CHECKPOINT}" ]]; then
      echo "POLICY_CHECKPOINT is required for checkpoint-effect" >&2
      exit 2
    fi
    EFFECT_ARGS=(
      --config "${CONFIG}"
      --policy-checkpoint "${POLICY_CHECKPOINT}"
      --num-gpus "${#GPU_IDS[@]}"
      --max-new-tokens "${CHECKPOINT_EFFECT_MAX_NEW_TOKENS}"
    )
    if [[ -n "${RUN_NAME}" ]]; then
      EFFECT_ARGS+=(--run-name "${RUN_NAME}")
    fi
    case "${VERBOSE_RUNTIME_LOGS}" in
      0) ;;
      1) EFFECT_ARGS+=(--verbose-runtime-logs) ;;
      *)
        echo "VERBOSE_RUNTIME_LOGS must be 0 or 1" >&2
        exit 2
        ;;
    esac
    python -m infoskill.cli checkpoint-effect "${EFFECT_ARGS[@]}"
    ;;
  grounding)
    GROUNDING_ARGS=(
      --config "${CONFIG}"
      --worker-batch-size "${GROUNDING_WORKER_BATCH_SIZE}"
      --worker-processes "${GROUNDING_WORKER_PROCESSES}"
      --replay-backend "${GROUNDING_REPLAY_BACKEND}"
      --native-batch-size "${GROUNDING_NATIVE_BATCH_SIZE}"
      --worker-inactivity-timeout-seconds "${GROUNDING_WORKER_INACTIVITY_TIMEOUT_SECONDS}"
    )
    if [[ -n "${GROUNDING_RESUME_RUN}" ]]; then
      if [[ -n "${RUN_NAME}" ]]; then
        echo "GROUNDING_RESUME_RUN cannot be combined with RUN_NAME" >&2
        exit 2
      fi
      GROUNDING_ARGS+=(--resume-run "${GROUNDING_RESUME_RUN}")
    else
      GROUNDING_ARGS+=("${EXTRA_ARGS[@]}")
    fi
    python -m infoskill.cli grounding "${GROUNDING_ARGS[@]}"
    ;;
  grounding-expert-diagnostic)
    if [[ -z "${GROUNDING_SOURCE_RUN}" ]]; then
      echo "GROUNDING_SOURCE_RUN is required for grounding-expert-diagnostic" >&2
      exit 2
    fi
    CUDA_VISIBLE_DEVICES="" python -m infoskill.cli grounding-expert-diagnostic \
      --config "${CONFIG}" \
      --source-grounding-run "${GROUNDING_SOURCE_RUN}" \
      --tasks-per-type "${GROUNDING_DIAGNOSTIC_TASKS_PER_TYPE}" \
      --max-replay-steps "${GROUNDING_DIAGNOSTIC_MAX_REPLAY_STEPS}" \
      "${EXTRA_ARGS[@]}"
    ;;
  grounding-planner-pilot)
    CUDA_VISIBLE_DEVICES="" python -m infoskill.cli grounding-planner-pilot \
      --config "${CONFIG}" \
      --tasks-per-type "${PLANNER_PILOT_TASKS_PER_TYPE}" \
      --worker-batch-size "${GROUNDING_WORKER_BATCH_SIZE}" \
      --worker-processes "${GROUNDING_WORKER_PROCESSES}" \
      --replay-backend "${GROUNDING_REPLAY_BACKEND}" \
      --native-batch-size "${GROUNDING_NATIVE_BATCH_SIZE}" \
      --max-replay-steps "${PLANNER_PILOT_MAX_REPLAY_STEPS}" \
      "${EXTRA_ARGS[@]}"
    ;;
  grounding-planner-parity)
    CUDA_VISIBLE_DEVICES="" python -m infoskill.cli grounding-planner-parity \
      --config "${CONFIG}" \
      --tasks-per-type "${GROUNDING_PARITY_TASKS_PER_TYPE}" \
      --worker-batch-size "${GROUNDING_PARITY_WORKER_BATCH_SIZE}" \
      --parallel-workers "${GROUNDING_PARITY_PARALLEL_WORKERS}" \
      --candidate-backend "${GROUNDING_PARITY_CANDIDATE_BACKEND}" \
      --native-batch-size "${GROUNDING_NATIVE_BATCH_SIZE}" \
      --minimum-speedup "${GROUNDING_PARITY_MINIMUM_SPEEDUP}" \
      --max-replay-steps "${PLANNER_PILOT_MAX_REPLAY_STEPS}" \
      "${EXTRA_ARGS[@]}"
    ;;
  grounding-planner-loop-diagnostic)
    if [[ -z "${GROUNDING_SOURCE_RUN}" ]]; then
      echo "GROUNDING_SOURCE_RUN is required for grounding-planner-loop-diagnostic" >&2
      exit 2
    fi
    CUDA_VISIBLE_DEVICES="" python -m infoskill.cli grounding-planner-loop-diagnostic \
      --config "${CONFIG}" \
      --source-pilot-run "${GROUNDING_SOURCE_RUN}" \
      --successful-two-object-controls "${PLANNER_LOOP_SUCCESS_CONTROLS}" \
      --max-replay-steps "${PLANNER_LOOP_MAX_REPLAY_STEPS}" \
      "${EXTRA_ARGS[@]}"
    ;;
  train)
    IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
    if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
      echo "GPUS must contain at least one physical GPU index" >&2
      exit 2
    fi
    for gpu_id in "${GPU_IDS[@]}"; do
      if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU index in GPUS=${GPUS}: ${gpu_id}" >&2
        exit 2
      fi
    done
    TRAIN_ARGS=(
      --config "${CONFIG}"
      --mode "${MODE}"
      "${RETRIEVAL_ARGS[@]}"
      --raw-skill-prompt-format "${RAW_SKILL_PROMPT_FORMAT}"
      --profile "${PROFILE}"
      --num-gpus "${#GPU_IDS[@]}"
      --environment-workers "${ENVIRONMENT_WORKERS}"
      --environment-backend "${ENVIRONMENT_BACKEND}"
      --cuda-memory-poll-interval-ms "${CUDA_MEMORY_POLL_INTERVAL_MS}"
      --policy-max-tokens-per-gpu "${POLICY_MAX_TOKENS_PER_GPU}"
    )
    if [[ -n "${MAX_UPDATES}" ]]; then
      TRAIN_ARGS+=(--max-updates "${MAX_UPDATES}")
    fi
    if [[ -n "${RUN_NAME}" ]]; then
      TRAIN_ARGS+=(--run-name "${RUN_NAME}")
    fi
    if [[ -n "${RESUME}" ]]; then
      TRAIN_ARGS+=(--resume "${RESUME}")
    fi
    if [[ -n "${GROUNDING_DATA}" ]]; then
      TRAIN_ARGS+=(--grounding-data "${GROUNDING_DATA}")
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
      TRAIN_ARGS+=(--dry-run)
    fi
    case "${BALANCE_POLICY_TOKENS_ACROSS_RANKS}" in
      0) TRAIN_ARGS+=(--no-balance-policy-tokens-across-ranks) ;;
      1) TRAIN_ARGS+=(--balance-policy-tokens-across-ranks) ;;
    esac
    case "${PERSISTENT_ROLLOUT_SESSION}" in
      1) TRAIN_ARGS+=(--persistent-rollout-session) ;;
      0) TRAIN_ARGS+=(--no-persistent-rollout-session) ;;
      *)
        echo "PERSISTENT_ROLLOUT_SESSION must be 0 or 1" >&2
        exit 2
        ;;
    esac
    case "${VERBOSE_RUNTIME_LOGS}" in
      0) ;;
      1) TRAIN_ARGS+=(--verbose-runtime-logs) ;;
      *)
        echo "VERBOSE_RUNTIME_LOGS must be 0 or 1" >&2
        exit 2
        ;;
    esac
    python -m infoskill.cli train "${TRAIN_ARGS[@]}"
    ;;
  *)
    echo "Unknown ACTION=${ACTION}; expected validate, eval, a diagnostic action, grounding, grounding-expert-diagnostic, grounding-planner-pilot, grounding-planner-parity, grounding-planner-loop-diagnostic, or train" >&2
    exit 2
    ;;
esac
