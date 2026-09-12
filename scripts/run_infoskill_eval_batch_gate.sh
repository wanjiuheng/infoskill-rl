#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPUS="${GPUS:-0,1,2}"
CONFIG="${CONFIG:-configs/alfworld_qwen25_7b.yaml}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-}"
PRESSURE_MANIFEST="${PRESSURE_MANIFEST:-configs/m1_eval_batch_pressure_valid_seen.json}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-8}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-12}"
MINIMUM_ROLLOUT_SPEEDUP="${MINIMUM_ROLLOUT_SPEEDUP:-1.10}"
MINIMUM_PHYSICAL_FREE_GB="${MINIMUM_PHYSICAL_FREE_GB:-8.0}"
CUDA_MEMORY_POLL_INTERVAL_MS="${CUDA_MEMORY_POLL_INTERVAL_MS:-200}"
GROUPED_INFOSKILL_CONDITIONING="${GROUPED_INFOSKILL_CONDITIONING:-0}"
RUN_FULL_EVAL_ON_PASS="${RUN_FULL_EVAL_ON_PASS:-1}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-m1-infoskill-eval-batch-gate}"

if [[ -z "${POLICY_CHECKPOINT}" ]]; then
  echo "POLICY_CHECKPOINT is required so the gate verifies portable M1 loading" >&2
  exit 2
fi
if [[ ! -f "${PRESSURE_MANIFEST}" ]]; then
  echo "pressure manifest does not exist: ${PRESSURE_MANIFEST}" >&2
  exit 2
fi
if [[ "${RUN_FULL_EVAL_ON_PASS}" != "0" && "${RUN_FULL_EVAL_ON_PASS}" != "1" ]]; then
  echo "RUN_FULL_EVAL_ON_PASS must be 0 or 1" >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BASELINE_NAME="${RUN_NAME_PREFIX}-batch${BASELINE_BATCH_SIZE}-${STAMP}"
CANDIDATE_NAME="${RUN_NAME_PREFIX}-batch${CANDIDATE_BATCH_SIZE}-${STAMP}"

run_diagnostic() {
  local batch_size="$1"
  local run_name="$2"
  env \
    GPUS="${GPUS}" \
    EVAL_BACKEND=verl \
    POLICY_CHECKPOINT="${POLICY_CHECKPOINT}" \
    PERSISTENT_ROLLOUT_SESSION=1 \
    ENVIRONMENT_BACKEND=native_batch \
    GROUPED_INFOSKILL_CONDITIONING="${GROUPED_INFOSKILL_CONDITIONING}" \
    EVAL_TASK_MANIFEST="${PRESSURE_MANIFEST}" \
    EVAL_BATCH_SIZE="${batch_size}" \
    CUDA_MEMORY_POLL_INTERVAL_MS="${CUDA_MEMORY_POLL_INTERVAL_MS}" \
    INFO_SKILL_CPU_THREADS=1 \
    RUN_NAME="${run_name}" \
    bash scripts/run_alfworld.sh eval infoskill
}

echo "[INFO-SKILL] pressure A/B: batch ${BASELINE_BATCH_SIZE}"
run_diagnostic "${BASELINE_BATCH_SIZE}" "${BASELINE_NAME}"
echo "[INFO-SKILL] pressure A/B: batch ${CANDIDATE_BATCH_SIZE}"
run_diagnostic "${CANDIDATE_BATCH_SIZE}" "${CANDIDATE_NAME}"

BASELINE_RUN="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d -name "*-${BASELINE_NAME}" | sort | tail -n 1)"
CANDIDATE_RUN="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d -name "*-${CANDIDATE_NAME}" | sort | tail -n 1)"
if [[ -z "${BASELINE_RUN}" || -z "${CANDIDATE_RUN}" ]]; then
  echo "failed to resolve pressure A/B run directories" >&2
  exit 2
fi

set +e
python scripts/compare_infoskill_eval_batch_runs.py \
  "${BASELINE_RUN}" \
  "${CANDIDATE_RUN}" \
  --baseline-batch-size "${BASELINE_BATCH_SIZE}" \
  --candidate-batch-size "${CANDIDATE_BATCH_SIZE}" \
  --minimum-rollout-speedup "${MINIMUM_ROLLOUT_SPEEDUP}" \
  --minimum-physical-free-gb "${MINIMUM_PHYSICAL_FREE_GB}" \
  | tee "${CANDIDATE_RUN}/eval-batch-gate.json"
GATE_RC="${PIPESTATUS[0]}"
set -e

echo "BASELINE_RUN=${BASELINE_RUN}"
echo "CANDIDATE_RUN=${CANDIDATE_RUN}"
echo "GATE_REPORT=${CANDIDATE_RUN}/eval-batch-gate.json"
if (( GATE_RC != 0 )); then
  echo "[INFO-SKILL] batch-size gate failed; full 140-task evaluation was not started" >&2
  exit "${GATE_RC}"
fi

if [[ "${RUN_FULL_EVAL_ON_PASS}" == "1" ]]; then
  FULL_NAME="${RUN_NAME_PREFIX}-full140-batch${CANDIDATE_BATCH_SIZE}-${STAMP}"
  echo "[INFO-SKILL] gate passed; starting full 140-task evaluation"
  env \
    GPUS="${GPUS}" \
    EVAL_BACKEND=verl \
    POLICY_CHECKPOINT="${POLICY_CHECKPOINT}" \
    PERSISTENT_ROLLOUT_SESSION=1 \
    ENVIRONMENT_BACKEND=native_batch \
    GROUPED_INFOSKILL_CONDITIONING="${GROUPED_INFOSKILL_CONDITIONING}" \
    EVAL_TASK_MANIFEST="" \
    EVAL_BATCH_SIZE="${CANDIDATE_BATCH_SIZE}" \
    CUDA_MEMORY_POLL_INTERVAL_MS="${CUDA_MEMORY_POLL_INTERVAL_MS}" \
    INFO_SKILL_CPU_THREADS=1 \
    RUN_NAME="${FULL_NAME}" \
    bash scripts/run_alfworld.sh eval infoskill
fi
