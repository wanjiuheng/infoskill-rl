#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:?RESUME_CHECKPOINT is required}"
HANDOFF="${HANDOFF:-${PROJECT_ROOT}/artifacts/qwen25-3b-m1-handoff-v1}"
GROUNDING_DATA="${GROUNDING_DATA:-${PROJECT_ROOT}/artifacts/alfworld-imitation-data-grounding}"
CONFIG="${CONFIG:-configs/alfworld_qwen25_3b.yaml}"
GPUS="${GPUS:-0,1,2}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-m1-qwen25-3b-alignment-diagnostic-${STAMP}}"

for required in \
  "${PYTHON}" \
  "${RESUME_CHECKPOINT}/checkpoint.complete.json" \
  "${HANDOFF}/checkpoint.complete.json" \
  "${HANDOFF}/skill-bank.json" \
  "${HANDOFF}/skill-bank-manifest.json" \
  "${GROUNDING_DATA}/manifest.json" \
  "${CONFIG}"
do
  [[ -f "${required}" ]] || {
    echo "required file is missing: ${required}" >&2
    exit 2
  }
done

active="$(pgrep -af '[p]ython -m infoskill.cli (train|eval)' || true)"
if [[ -n "${active}" ]]; then
  echo "another INFO-SKILL train/eval process is active; refusing launch" >&2
  printf '%s\n' "${active}" >&2
  exit 2
fi

IFS=',' read -r -a gpu_ids <<< "${GPUS}"
if (( ${#gpu_ids[@]} != 3 )); then
  echo "the registered diagnostic requires the original 3-GPU layout" >&2
  exit 2
fi

export PYTHON
export PATH="$(dirname -- "${PYTHON}"):${PATH}"

echo "RESUME_CHECKPOINT=${RESUME_CHECKPOINT}"
echo "RUN_NAME=${RUN_NAME}"
echo "The optimizer remains behind the update-0 alignment gate."

GPUS="${GPUS}" \
PROFILE=formal \
MAX_UPDATES=445 \
SEGMENT_END_UPDATE=200 \
RESUME="${RESUME_CHECKPOINT}" \
GROUNDING_DATA="${GROUNDING_DATA}" \
WARMSTART_HANDOFF= \
SKILL_BANK="${HANDOFF}/skill-bank.json" \
SKILL_BANK_MANIFEST="${HANDOFF}/skill-bank-manifest.json" \
EVAL_BATCH_SIZE=64 \
PERSISTENT_ROLLOUT_SESSION=1 \
ENVIRONMENT_BACKEND=native_batch \
CUDA_MEMORY_POLL_INTERVAL_MS=1000 \
POLICY_MAX_TOKENS_PER_GPU=12288 \
ROLLOUT_MAX_BATCHED_TOKENS=16384 \
ACTOR_LEARNING_RATE=3e-6 \
HYBRID_PREFIX_CUDA_GRAPH=1 \
LORA_SHRINK_SPLIT_K_ONE=1 \
POLICY_GRADIENT_CLIP_MODE=joint \
CHECKPOINT_KEEP_RECENT=5 \
CHECKPOINT_KEEP_BEST_VALID=1 \
DRIFT_GUARD_PPO_KL_THRESHOLD=0.02 \
DRIFT_GUARD_INVALID_ACTION_RATE_THRESHOLD=0.05 \
DRIFT_GUARD_CONSECUTIVE_UPDATES=2 \
RUN_NAME="${RUN_NAME}" \
  bash scripts/run_alfworld.sh train infoskill "${CONFIG}"
