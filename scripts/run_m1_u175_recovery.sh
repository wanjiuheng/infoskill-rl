#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}"
SOURCE_RUN="${SOURCE_RUN:-${PROJECT_ROOT}/runs/20260920T044319Z-m1-handoff-lr3e6-formal-u200-20260920_124241}"
RESUME_CHECKPOINT="${SOURCE_RUN}/checkpoints/step-000175"
PROTECTED_CHECKPOINTS=(
  "${SOURCE_RUN}/checkpoints/step-000175"
  "${SOURCE_RUN}/checkpoints/step-000195"
  "${SOURCE_RUN}/checkpoints/step-000200"
)
GROUNDING_DATA="${GROUNDING_DATA:-${PROJECT_ROOT}/artifacts/alfworld-imitation-data-grounding}"
SKILL_BANK="${SKILL_BANK:-${PROJECT_ROOT}/artifacts/m1-handoff/skill-bank.json}"
SKILL_BANK_MANIFEST="${SKILL_BANK_MANIFEST:-${PROJECT_ROOT}/artifacts/m1-handoff/skill-bank-manifest.json}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-m1-handoff-recovery-s175-lr1e6-u445-${STAMP}}"

[[ -x "${PYTHON_BIN}" ]] || {
  echo "Python interpreter is unavailable: ${PYTHON_BIN}" >&2
  exit 2
}

# These three immutable source checkpoints are inputs/evidence. The recovery
# run owns a separate directory and its retention policy cannot touch them.
for checkpoint in "${PROTECTED_CHECKPOINTS[@]}"; do
  [[ -f "${checkpoint}/checkpoint.complete.json" ]] || {
    echo "Required protected checkpoint is incomplete: ${checkpoint}" >&2
    exit 2
  }
done

for required in \
  "${GROUNDING_DATA}/manifest.json" \
  "${SKILL_BANK}" \
  "${SKILL_BANK_MANIFEST}"
do
  [[ -f "${required}" ]] || {
    echo "Required recovery artifact is missing: ${required}" >&2
    exit 2
  }
done

active="$(pgrep -af '[p]ython -m infoskill.cli (train|eval)' || true)"
if [[ -n "${active}" ]]; then
  echo "Another training or evaluation process is active; refusing to start:" >&2
  echo "${active}" >&2
  exit 2
fi

free_bytes="$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')"
if (( free_bytes < 15 * 1024 * 1024 * 1024 )); then
  echo "Free disk is below 15 GiB; refusing to start" >&2
  df -h /root/autodl-tmp >&2
  exit 2
fi

"${PYTHON_BIN}" -c \
  'import sentence_transformers; print("sentence-transformers:", sentence_transformers.__version__)'

echo "SOURCE_RUN=${SOURCE_RUN}"
echo "RESUME=${RESUME_CHECKPOINT}"
echo "RUN_NAME=${RUN_NAME}"
echo "Protected source checkpoints: step-000175 step-000195 step-000200"

exec env \
  PYTHON="${PYTHON_BIN}" \
  PATH="$(dirname "${PYTHON_BIN}"):${PATH}" \
  GPUS="${GPUS:-0,1,2}" \
  PROFILE=formal \
  MAX_UPDATES=445 \
  RESUME="${RESUME_CHECKPOINT}" \
  GROUNDING_DATA="${GROUNDING_DATA}" \
  SKILL_BANK="${SKILL_BANK}" \
  SKILL_BANK_MANIFEST="${SKILL_BANK_MANIFEST}" \
  EVAL_BATCH_SIZE=64 \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  ENVIRONMENT_WORKERS=1 \
  POLICY_MAX_TOKENS_PER_GPU=12288 \
  ROLLOUT_MAX_BATCHED_TOKENS=16384 \
  CUDA_MEMORY_POLL_INTERVAL_MS=1000 \
  INFO_SKILL_CPU_THREADS=1 \
  ACTOR_LEARNING_RATE=1e-6 \
  POLICY_GRADIENT_CLIP_MODE=joint \
  HYBRID_PREFIX_CUDA_GRAPH=1 \
  LORA_SHRINK_SPLIT_K_ONE=1 \
  FUSE_KL_PPO_FORWARD=0 \
  SKIP_UNUSED_OLD_LOGPROB_ENTROPY=0 \
  BALANCE_POLICY_TOKENS_ACROSS_RANKS=1 \
  CHECKPOINT_KEEP_RECENT=5 \
  CHECKPOINT_KEEP_BEST_VALID=1 \
  DRIFT_GUARD_PPO_KL_THRESHOLD=0.02 \
  DRIFT_GUARD_INVALID_ACTION_RATE_THRESHOLD=0.05 \
  DRIFT_GUARD_CONSECUTIVE_UPDATES=2 \
  RUN_NAME="${RUN_NAME}" \
  bash scripts/run_alfworld.sh train infoskill
