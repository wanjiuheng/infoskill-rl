#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
ACTION="${1:-prepare}" # prepare | train | finalize
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

GROUNDING_DATA="${GROUNDING_DATA:-}"
PREPARED_DATA="${PREPARED_DATA:-${PROJECT_ROOT}/artifacts/alfworld-imitation-data}"
PLANNER_SKILL_BANK="${PLANNER_SKILL_BANK:-${PROJECT_ROOT}/artifacts/alfworld-planner-skills.json}"
POLICY_MODEL="${POLICY_MODEL:-/root/autodl-tmp/wjh/models/Qwen/Qwen2.5-7B-Instruct}"
SFT_OUTPUT="${SFT_OUTPUT:-${PROJECT_ROOT}/runs/actor-imitation-warmstart}"
HANDOFF_OUTPUT="${HANDOFF_OUTPUT:-${PROJECT_ROOT}/artifacts/m1-handoff}"
BASE_MODEL_ID="${BASE_MODEL_ID:-qwen2.5-7b-instruct}"
GPUS="${GPUS:-0,1,2}"

case "${ACTION}" in
  prepare)
    [[ -n "${GROUNDING_DATA}" ]] || { echo "GROUNDING_DATA is required" >&2; exit 2; }
    "${PYTHON_BIN}" -m infoskill.imitation.cli prepare \
      --grounding "${GROUNDING_DATA}" \
      --output "${PREPARED_DATA}" \
      --skill-bank "${PLANNER_SKILL_BANK}"
    ;;
  train)
    [[ -f "${PREPARED_DATA}/manifest.json" ]] || { echo "prepared data is missing" >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="${GPUS}"
    IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
    TRAIN_ARGS=(
      --model "${POLICY_MODEL}"
      --data "${PREPARED_DATA}"
      --output "${SFT_OUTPUT}"
      --learning-rate "${IMITATION_LEARNING_RATE:-1e-4}"
      --epochs "${IMITATION_EPOCHS:-2}"
      --per-device-batch-size "${IMITATION_BATCH_SIZE:-2}"
      --gradient-accumulation-steps "${IMITATION_GRAD_ACCUM:-8}"
      --max-length "${IMITATION_MAX_LENGTH:-4352}"
    )
    if [[ -n "${IMITATION_RESUME:-}" ]]; then
      TRAIN_ARGS+=(--resume-from-checkpoint "${IMITATION_RESUME}")
    fi
    "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone --nproc_per_node="${#GPU_IDS[@]}" \
      -m infoskill.imitation.cli train \
      "${TRAIN_ARGS[@]}"
    ;;
  finalize)
    "${PYTHON_BIN}" -m infoskill.imitation.cli finalize \
      --adapter "${SFT_OUTPUT}/final-adapter" \
      --imitation-manifest "${PREPARED_DATA}/manifest.json" \
      --skill-bank "${PLANNER_SKILL_BANK}" \
      --base-model-id "${BASE_MODEL_ID}" \
      --output "${HANDOFF_OUTPUT}"
    ;;
  *) echo "unknown action: ${ACTION}" >&2; exit 2 ;;
esac
