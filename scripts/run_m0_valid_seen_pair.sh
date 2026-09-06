#!/usr/bin/env bash
set -euo pipefail

# Evaluate the shared SFT initialization and one portable M0 checkpoint with
# the same deterministic four-GPU VERL/vLLM protocol.
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:?set POLICY_CHECKPOINT to checkpoints/step-*}"
GPUS="${GPUS:-0,1,2,3}"
CONFIG="${CONFIG:-configs/alfworld_qwen25_7b.yaml}"
BASE_RUN_NAME="${BASE_RUN_NAME:-m0-sft-valid-seen-update0}"
CHECKPOINT_RUN_NAME="${CHECKPOINT_RUN_NAME:-m0-sft-valid-seen-checkpoint}"
ENVIRONMENT_BACKEND="${ENVIRONMENT_BACKEND:-native_batch}"
PERSISTENT_ROLLOUT_SESSION="${PERSISTENT_ROLLOUT_SESSION:-1}"
VERBOSE_RUNTIME_LOGS="${VERBOSE_RUNTIME_LOGS:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "[INFO-SKILL] paired valid_seen evaluation: base update 0"
GPUS="${GPUS}" \
CONFIG="${CONFIG}" \
RUN_NAME="${BASE_RUN_NAME}" \
CHECKPOINT_STEP=0 \
EVAL_BACKEND=verl \
POLICY_CHECKPOINT="" \
ENVIRONMENT_BACKEND="${ENVIRONMENT_BACKEND}" \
PERSISTENT_ROLLOUT_SESSION="${PERSISTENT_ROLLOUT_SESSION}" \
VERBOSE_RUNTIME_LOGS="${VERBOSE_RUNTIME_LOGS}" \
bash "${SCRIPT_DIR}/run_alfworld.sh" eval no_skill

echo "[INFO-SKILL] paired valid_seen evaluation: portable checkpoint"
GPUS="${GPUS}" \
CONFIG="${CONFIG}" \
RUN_NAME="${CHECKPOINT_RUN_NAME}" \
CHECKPOINT_STEP=0 \
EVAL_BACKEND=verl \
POLICY_CHECKPOINT="${POLICY_CHECKPOINT}" \
ENVIRONMENT_BACKEND="${ENVIRONMENT_BACKEND}" \
PERSISTENT_ROLLOUT_SESSION="${PERSISTENT_ROLLOUT_SESSION}" \
VERBOSE_RUNTIME_LOGS="${VERBOSE_RUNTIME_LOGS}" \
bash "${SCRIPT_DIR}/run_alfworld.sh" eval no_skill
