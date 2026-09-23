#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
HANDOFF="${HANDOFF:?HANDOFF is required}"
CONFIG="${CONFIG:-configs/alfworld_qwen25_7b.yaml}"
SKILL_BANK="${SKILL_BANK:-${HANDOFF}/skill-bank.json}"
SKILL_BANK_MANIFEST="${SKILL_BANK_MANIFEST:-${HANDOFF}/skill-bank-manifest.json}"

GPUS="${GPUS:-0,1,2}" \
EVAL_BACKEND=verl \
CHECKPOINT_STEP=0 \
WARMSTART_HANDOFF="${HANDOFF}" \
SKILL_BANK="${SKILL_BANK}" \
SKILL_BANK_MANIFEST="${SKILL_BANK_MANIFEST}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}" \
PERSISTENT_ROLLOUT_SESSION=1 \
ENVIRONMENT_BACKEND=native_batch \
HYBRID_PREFIX_CUDA_GRAPH=1 \
LORA_SHRINK_SPLIT_K_ONE=1 \
RUN_NAME="${RUN_NAME:-m1-handoff-valid-seen-update0}" \
bash scripts/run_alfworld.sh eval infoskill "${CONFIG}"
