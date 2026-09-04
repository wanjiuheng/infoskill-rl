#!/usr/bin/env bash
set -euo pipefail

# Central parameter panel. Every value can also be overridden as an environment variable.
ACTION="${ACTION:-${1:-eval}}"                 # validate | eval | grounding | train
MODE="${MODE:-${2:-no_skill}}"                # no_skill | raw_skill_prompt | infoskill
CONFIG="${CONFIG:-${3:-configs/alfworld_qwen25_7b.yaml}}"
GPUS="${GPUS:-${4:-0}}"                       # examples: 0 or 0,1 or 0,1,2,3
RUN_NAME="${RUN_NAME:-}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-0}"
PROFILE="${PROFILE:-smoke}"                   # smoke | integration | pilot | formal
MAX_UPDATES="${MAX_UPDATES:-}"
RESUME="${RESUME:-}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

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
    python -m infoskill.cli train "${TRAIN_ARGS[@]}"
    ;;
  *)
    echo "Unknown ACTION=${ACTION}; expected validate, eval, grounding, or train" >&2
    exit 2
    ;;
esac
