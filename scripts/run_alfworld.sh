#!/usr/bin/env bash
set -euo pipefail

# Central parameter panel. Every value can also be overridden as an environment variable.
ACTION="${ACTION:-${1:-eval}}"                 # validate | eval | grounding | train
MODE="${MODE:-${2:-no_skill}}"                # no_skill | raw_skill_prompt | infoskill
CONFIG="${CONFIG:-${3:-configs/alfworld_qwen25_7b.yaml}}"
GPUS="${GPUS:-${4:-0}}"                       # examples: 0 or 0,1 or 0,1,2,3
RUN_NAME="${RUN_NAME:-}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-0}"
PROFILE="${PROFILE:-smoke}"                   # smoke | integration | benchmark | pilot | formal
MAX_UPDATES="${MAX_UPDATES:-}"
RESUME="${RESUME:-}"
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
# Diagnostic only. Zero avoids polling overhead in normal/formal runs.
CUDA_MEMORY_POLL_INTERVAL_MS="${CUDA_MEMORY_POLL_INTERVAL_MS:-0}"
# Dynamic old/ref/actor micro-batch budget. This does not alter vLLM rollout
# scheduling. Keep the validated default unless running a monitored A/B gate.
POLICY_MAX_TOKENS_PER_GPU="${POLICY_MAX_TOKENS_PER_GPU:-16384}"
# Candidate only. Reassigns samples among ranks while preserving each global
# PPO minibatch's membership.
BALANCE_POLICY_TOKENS_ACROSS_RANKS="${BALANCE_POLICY_TOKENS_ACROSS_RANKS:-0}"

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
export OMP_NUM_THREADS="${INFO_SKILL_CPU_THREADS}"
export MKL_NUM_THREADS="${INFO_SKILL_CPU_THREADS}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"

EXTRA_ARGS=()
if [[ -n "${RUN_NAME}" ]]; then
  EXTRA_ARGS+=(--run-name "${RUN_NAME}")
fi

echo "[INFO-SKILL] action=${ACTION} mode=${MODE} gpus=${GPUS} config=${CONFIG}"

case "${ACTION}" in
  validate)
    python -m infoskill.cli validate --config "${CONFIG}" --mode "${MODE}"
    ;;
  eval)
    python -m infoskill.cli eval \
      --config "${CONFIG}" \
      --mode "${MODE}" \
      --checkpoint-step "${CHECKPOINT_STEP}" \
      "${EXTRA_ARGS[@]}"
    ;;
  grounding)
    python -m infoskill.cli grounding --config "${CONFIG}" "${EXTRA_ARGS[@]}"
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
    echo "Unknown ACTION=${ACTION}; expected validate, eval, grounding, or train" >&2
    exit 2
    ;;
esac
