#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/wjh/models/Qwen/Qwen2.5-3B-Instruct}"
CONFIG="${CONFIG:-configs/alfworld_qwen25_3b.yaml}"
BASE_MODEL_ID="${BASE_MODEL_ID:-qwen2.5-3b-instruct}"
GPUS="${GPUS:-0,1,2}"
PREPARED_DATA="${PREPARED_DATA:-${PROJECT_ROOT}/artifacts/alfworld-imitation-data}"
GROUNDING_DATA="${GROUNDING_DATA:-${PROJECT_ROOT}/artifacts/alfworld-imitation-data-grounding}"
PLANNER_SKILL_BANK="${PLANNER_SKILL_BANK:-${PROJECT_ROOT}/artifacts/alfworld-planner-skills.json}"
SFT_OUTPUT="${SFT_OUTPUT:-${PROJECT_ROOT}/runs/qwen25-3b-actor-imitation-warmstart-v1}"
HANDOFF="${HANDOFF:-${PROJECT_ROOT}/artifacts/qwen25-3b-m1-handoff-v1}"
TARGET_UPDATES="${TARGET_UPDATES:-200}"
PIPELINE_STAMP="${PIPELINE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MINIMUM_FREE_DISK_GB="${MINIMUM_FREE_DISK_GB:-15}"

# The optimizer batch remains 48: 4 examples x 4 accumulation x 3 ranks.
IMITATION_BATCH_SIZE="${IMITATION_BATCH_SIZE:-4}"
IMITATION_GRAD_ACCUM="${IMITATION_GRAD_ACCUM:-4}"
EXPECTED_EFFECTIVE_BATCH_SIZE="${EXPECTED_EFFECTIVE_BATCH_SIZE:-48}"

# These only enlarge dynamic execution batches.  The formal 8 x 8 GRPO
# sample membership is owned by the training profile and remains unchanged.
POLICY_MAX_TOKENS_PER_GPU="${POLICY_MAX_TOKENS_PER_GPU:-16384}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-24576}"
ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE:-3e-6}"
LOGPROB_ALIGNMENT_PROFILE="${LOGPROB_ALIGNMENT_PROFILE:-qwen25_3b_calibrated}"

EVAL_RUN_NAME="${EVAL_RUN_NAME:-m1-qwen25-3b-handoff-valid-seen-u0-${PIPELINE_STAMP}}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-m1-qwen25-3b-handoff-lr3e6-u${TARGET_UPDATES}-${PIPELINE_STAMP}}"

export PYTHON
export PATH="$(dirname -- "${PYTHON}"):${PATH}"

log() {
  printf '%s | %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  [[ -f "$1" ]] || {
    log "required file is missing: $1"
    exit 2
  }
}

require_file "${PYTHON}"
require_file "${MODEL}/config.json"
require_file "${CONFIG}"
require_file "${PREPARED_DATA}/manifest.json"
require_file "${GROUNDING_DATA}/manifest.json"
require_file "${PLANNER_SKILL_BANK}"

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
EXPECTED_WORLD_SIZE="${#GPU_IDS[@]}"
if (( EXPECTED_WORLD_SIZE != 3 )); then
  log "the registered throughput configuration requires exactly 3 GPUs"
  exit 2
fi
if (( IMITATION_BATCH_SIZE * IMITATION_GRAD_ACCUM * EXPECTED_WORLD_SIZE != EXPECTED_EFFECTIVE_BATCH_SIZE )); then
  log "imitation settings must preserve effective batch ${EXPECTED_EFFECTIVE_BATCH_SIZE}"
  exit 2
fi

active="$(pgrep -af '[p]ython -m infoskill.cli (train|eval)|[t]orch.distributed.run.*infoskill.imitation' || true)"
if [[ -n "${active}" ]]; then
  log "another INFO-SKILL train/eval process is active; refusing launch"
  printf '%s\n' "${active}"
  exit 2
fi

available_bytes="$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')"
minimum_bytes=$((MINIMUM_FREE_DISK_GB * 1024 * 1024 * 1024))
if (( available_bytes < minimum_bytes )); then
  log "free disk is below ${MINIMUM_FREE_DISK_GB} GiB; refusing launch"
  df -h /root/autodl-tmp
  exit 2
fi

PYTHONPATH=src "${PYTHON}" - "${MODEL}" "${BASE_MODEL_ID}" <<'PY'
import json
import sys

from infoskill.persistence.model_identity import verify_policy_model_identity

identity = verify_policy_model_identity(sys.argv[1], model_id=sys.argv[2])
print(json.dumps({
    "model_id": identity.model_id,
    "revision": identity.revision,
    "sha256": identity.sha256,
}, indent=2))
PY

if [[ ! -f "${SFT_OUTPUT}/final-adapter/imitation-training-manifest.json" ]]; then
  if [[ -d "${SFT_OUTPUT}" && -z "${IMITATION_RESUME:-}" ]]; then
    log "incomplete warm-start output exists; set IMITATION_RESUME to an exact checkpoint or move it aside: ${SFT_OUTPUT}"
    exit 2
  fi
  log "starting Qwen2.5-3B actor imitation with batch=${IMITATION_BATCH_SIZE}, accumulation=${IMITATION_GRAD_ACCUM}, effective=${EXPECTED_EFFECTIVE_BATCH_SIZE}"
  POLICY_MODEL="${MODEL}" \
  BASE_MODEL_ID="${BASE_MODEL_ID}" \
  PREPARED_DATA="${PREPARED_DATA}" \
  PLANNER_SKILL_BANK="${PLANNER_SKILL_BANK}" \
  SFT_OUTPUT="${SFT_OUTPUT}" \
  GPUS="${GPUS}" \
  IMITATION_BATCH_SIZE="${IMITATION_BATCH_SIZE}" \
  IMITATION_GRAD_ACCUM="${IMITATION_GRAD_ACCUM}" \
  IMITATION_RESUME="${IMITATION_RESUME:-}" \
    bash scripts/run_actor_imitation.sh train
else
  log "completed warm-start already exists: ${SFT_OUTPUT}"
fi

log "continuing automatically through immutable handoff, update-0 evaluation, and M1 update ${TARGET_UPDATES}"
CONFIG="${CONFIG}" \
BASE_MODEL_ID="${BASE_MODEL_ID}" \
SFT_OUTPUT="${SFT_OUTPUT}" \
HANDOFF="${HANDOFF}" \
GROUNDING_DATA="${GROUNDING_DATA}" \
TARGET_UPDATES="${TARGET_UPDATES}" \
GPUS="${GPUS}" \
EVAL_BATCH_SIZE=64 \
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE}" \
EXPECTED_EFFECTIVE_BATCH_SIZE="${EXPECTED_EFFECTIVE_BATCH_SIZE}" \
MINIMUM_FREE_DISK_GB="${MINIMUM_FREE_DISK_GB}" \
PIPELINE_STAMP="${PIPELINE_STAMP}" \
EVAL_RUN_NAME="${EVAL_RUN_NAME}" \
TRAIN_RUN_NAME="${TRAIN_RUN_NAME}" \
POLICY_MAX_TOKENS_PER_GPU="${POLICY_MAX_TOKENS_PER_GPU}" \
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS}" \
ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE}" \
LOGPROB_ALIGNMENT_PROFILE="${LOGPROB_ALIGNMENT_PROFILE}" \
CUDA_MEMORY_POLL_INTERVAL_MS=1000 \
DRIFT_GUARD_PPO_KL_THRESHOLD=0.02 \
DRIFT_GUARD_INVALID_ACTION_RATE_THRESHOLD=0.05 \
DRIFT_GUARD_CONSECUTIVE_UPDATES=2 \
  bash scripts/run_actor_imitation_to_m1_pipeline.sh
