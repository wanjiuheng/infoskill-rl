#!/usr/bin/env bash
set -euo pipefail

# Bounded Qwen3-1.7B M1 recovery: fork the best pre-drift checkpoint and
# update only the LoRA actor through update 100. The source run is read-only.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}"
SOURCE_RUN="${SOURCE_RUN:-${PROJECT_ROOT}/runs/20260922T233507Z-m1-qwen3-1p7b-handoff-lr3e6-u200-20260923_064630}"
SOURCE_CONFIG="${SOURCE_RUN}/resolved_config.json"
RESUME_CHECKPOINT="${SOURCE_RUN}/checkpoints/step-000075"
SEGMENT_END_UPDATE="${SEGMENT_END_UPDATE:-100}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-m1-qwen3-1p7b-recovery-s75-actoronly-lr1e6-u100-${STAMP}}"

[[ -x "${PYTHON_BIN}" ]] || {
  echo "Python interpreter is unavailable: ${PYTHON_BIN}" >&2
  exit 2
}
[[ -f "${SOURCE_CONFIG}" ]] || {
  echo "Source run has no resolved config: ${SOURCE_CONFIG}" >&2
  exit 2
}
[[ -f "${RESUME_CHECKPOINT}/checkpoint.complete.json" ]] || {
  echo "Source step-000075 is incomplete: ${RESUME_CHECKPOINT}" >&2
  exit 2
}

mapfile -t SOURCE_PATHS < <(
  "${PYTHON_BIN}" - "${SOURCE_CONFIG}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
paths = config["app_config"]["paths"]
for key in ("grounding_data", "skill_bank", "skill_bank_manifest"):
    value = paths.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"source resolved config has no non-empty {key}")
    print(value)
PY
)
[[ "${#SOURCE_PATHS[@]}" -eq 3 ]] || {
  echo "Could not resolve the source conditioning paths" >&2
  exit 2
}
RECOVERY_GROUNDING_DATA="${SOURCE_PATHS[0]}"
RECOVERY_SKILL_BANK="${SOURCE_PATHS[1]}"
RECOVERY_SKILL_BANK_MANIFEST="${SOURCE_PATHS[2]}"

for required in \
  "${RECOVERY_GROUNDING_DATA}/manifest.json" \
  "${RECOVERY_SKILL_BANK}" \
  "${RECOVERY_SKILL_BANK_MANIFEST}"
do
  [[ -f "${required}" ]] || {
    echo "Required source artifact is missing: ${required}" >&2
    exit 2
  }
done

active="$(pgrep -af '[p]ython -m infoskill.cli (train|eval)' || true)"
if [[ -n "${active}" ]]; then
  echo "Another InfoSkill train/eval process is active; refusing to start:" >&2
  echo "${active}" >&2
  exit 2
fi

free_bytes="$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')"
if (( free_bytes < 12 * 1024 * 1024 * 1024 )); then
  echo "Free disk is below 12 GiB; refusing to start" >&2
  df -h /root/autodl-tmp >&2
  exit 2
fi

"${PYTHON_BIN}" -c \
  'import sentence_transformers; print("sentence-transformers:", sentence_transformers.__version__)'

echo "SOURCE_RUN=${SOURCE_RUN}"
echo "RESUME=${RESUME_CHECKPOINT}"
echo "RUN_NAME=${RUN_NAME}"
echo "SEGMENT_END_UPDATE=${SEGMENT_END_UPDATE}"
echo "Source run and step-000075 remain unchanged."

exec env \
  PYTHON="${PYTHON_BIN}" \
  PATH="$(dirname "${PYTHON_BIN}"):${PATH}" \
  GPUS="${GPUS:-0,1,2}" \
  PROFILE=formal \
  MAX_UPDATES=445 \
  SEGMENT_END_UPDATE="${SEGMENT_END_UPDATE}" \
  RESUME="${RESUME_CHECKPOINT}" \
  GROUNDING_DATA="${RECOVERY_GROUNDING_DATA}" \
  SKILL_BANK="${RECOVERY_SKILL_BANK}" \
  SKILL_BANK_MANIFEST="${RECOVERY_SKILL_BANK_MANIFEST}" \
  EVAL_BATCH_SIZE=64 \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  ENVIRONMENT_WORKERS=1 \
  POLICY_MAX_TOKENS_PER_GPU=12288 \
  ROLLOUT_MAX_BATCHED_TOKENS=16384 \
  CUDA_MEMORY_POLL_INTERVAL_MS=1000 \
  INFO_SKILL_CPU_THREADS=1 \
  ACTOR_LEARNING_RATE=1e-6 \
  INVALID_ACTION_PENALTY=0.1 \
  FREEZE_INFOSKILL_CONDITIONING=1 \
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
  bash scripts/run_alfworld.sh \
    train infoskill configs/alfworld_qwen3_1p7b.yaml "${GPUS:-0,1,2}"
