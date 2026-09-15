#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-}"
GROUNDING_DATA="${GROUNDING_DATA:-}"
GPUS="${GPUS:-0,1,2}"
PYTHON="${PYTHON:-$(command -v python)}"
CONFIG="${CONFIG:-configs/alfworld_qwen25_7b.yaml}"
MINIMUM_PHYSICAL_FREE_GB="${MINIMUM_PHYSICAL_FREE_GB:-8.0}"
MINIMUM_FREE_DISK_GB="${MINIMUM_FREE_DISK_GB:-15}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-m1-gradient-clip-gate}"
CONTROL_RUN="${CONTROL_RUN:-}"

if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}/checkpoint.complete.json" ]]; then
  echo "SOURCE_CHECKPOINT must be a committed portable checkpoint" >&2
  exit 2
fi
if [[ -z "${GROUNDING_DATA}" || ! -f "${GROUNDING_DATA}/manifest.json" ]]; then
  echo "GROUNDING_DATA must be a finalized grounding run" >&2
  exit 2
fi
if [[ ! "$(basename -- "${SOURCE_CHECKPOINT}")" =~ ^step-([0-9]{6})$ ]]; then
  echo "SOURCE_CHECKPOINT must end in checkpoints/step-NNNNNN" >&2
  exit 2
fi

SOURCE_UPDATE="$((10#${BASH_REMATCH[1]}))"
TARGET_UPDATE="$((SOURCE_UPDATE + 1))"
ACTIVE="$(pgrep -af '[p]ython -m infoskill.cli train' || true)"
if [[ -n "${ACTIVE}" ]]; then
  echo "another INFO-SKILL training process is active; refusing concurrent A/B:" >&2
  echo "${ACTIVE}" >&2
  exit 2
fi

FREE_BYTES="$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')"
MINIMUM_FREE_BYTES="$((MINIMUM_FREE_DISK_GB * 1024 * 1024 * 1024))"
if (( FREE_BYTES < MINIMUM_FREE_BYTES )); then
  echo "free disk is below ${MINIMUM_FREE_DISK_GB} GiB; refusing to start" >&2
  df -h /root/autodl-tmp >&2
  exit 2
fi

"${PYTHON}" -c \
  'import sentence_transformers; print("sentence-transformers:", sentence_transformers.__version__)'

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONTROL_NAME="${RUN_NAME_PREFIX}-joint-s${SOURCE_UPDATE}-u${TARGET_UPDATE}-${STAMP}"
CANDIDATE_NAME="${RUN_NAME_PREFIX}-separate-s${SOURCE_UPDATE}-u${TARGET_UPDATE}-${STAMP}"

run_case() {
  local run_name="$1"
  local clip_mode="$2"
  echo "[INFO-SKILL] gradient-clip A/B: ${clip_mode}, update ${SOURCE_UPDATE}->${TARGET_UPDATE}"
  env \
    GPUS="${GPUS}" \
    PROFILE=formal \
    MAX_UPDATES=445 \
    SEGMENT_END_UPDATE="${TARGET_UPDATE}" \
    RESUME="${SOURCE_CHECKPOINT}" \
    GROUNDING_DATA="${GROUNDING_DATA}" \
    PERSISTENT_ROLLOUT_SESSION=1 \
    ENVIRONMENT_BACKEND=native_batch \
    ENVIRONMENT_WORKERS=1 \
    POLICY_MAX_TOKENS_PER_GPU=12288 \
    ROLLOUT_MAX_BATCHED_TOKENS=16384 \
    BALANCE_POLICY_TOKENS_ACROSS_RANKS=1 \
    SKIP_UNUSED_OLD_LOGPROB_ENTROPY=0 \
    HYBRID_PREFIX_CUDA_GRAPH=1 \
    FUSE_KL_PPO_FORWARD=0 \
    POLICY_GRADIENT_CLIP_MODE="${clip_mode}" \
    EVAL_BATCH_SIZE=64 \
    CHECKPOINT_KEEP_RECENT=5 \
    CHECKPOINT_KEEP_BEST_VALID=1 \
    CUDA_MEMORY_POLL_INTERVAL_MS=1000 \
    INFO_SKILL_CPU_THREADS=1 \
    RUN_NAME="${run_name}" \
    bash scripts/run_alfworld.sh train infoskill
}

if [[ -n "${CONTROL_RUN}" ]]; then
  if [[ ! -f "${CONTROL_RUN}/training_summary.json" ]]; then
    echo "CONTROL_RUN must contain a completed one-update control run" >&2
    exit 2
  fi
  echo "[INFO-SKILL] reusing gradient-clip control: ${CONTROL_RUN}"
else
  run_case "${CONTROL_NAME}" joint
  CONTROL_RUN="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d -name "*-${CONTROL_NAME}" | sort | tail -n 1)"
fi
run_case "${CANDIDATE_NAME}" separate

CANDIDATE_RUN="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d -name "*-${CANDIDATE_NAME}" | sort | tail -n 1)"
if [[ -z "${CONTROL_RUN}" || -z "${CANDIDATE_RUN}" ]]; then
  echo "failed to resolve gradient-clip A/B run directories" >&2
  exit 2
fi

GATE_REPORT="${PROJECT_ROOT}/m1-gradient-clip-gate-s${SOURCE_UPDATE}-${STAMP}.json"
set +e
"${PYTHON}" scripts/compare_infoskill_training_optimization_runs.py \
  "${CONTROL_RUN}" \
  "${CANDIDATE_RUN}" \
  --candidate-mode separate-grad-clip \
  --minimum-core-speedup 0 \
  --minimum-physical-free-gb "${MINIMUM_PHYSICAL_FREE_GB}" \
  | tee "${GATE_REPORT}"
COMPARE_RC="${PIPESTATUS[0]}"
set -e

# A different post-update checkpoint is expected for an algorithm candidate.
# Independent stochastic forks need the same scheduled task/rollout identities,
# not identical sampled actions. Efficacy is evaluated separately.
set +e
"${PYTHON}" - "${GATE_REPORT}" "${COMPARE_RC}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
checks = {
    "settings_valid": report.get("settings_valid") is True,
    "same_training_workload": (
        report.get("trace_workload_comparison", {}).get("passed") is True
    ),
    "physical_memory_valid": report.get("physical_memory_valid") is True,
    "algorithm_change_registered": report.get("algorithm_change_requested") is True,
    "efficacy_gate_required": report.get("efficacy_gate_required") is True,
}
summary = {
    "comparator_rc_expected_nonzero": int(sys.argv[2]),
    "infrastructure_gate_passed": all(checks.values()),
    "checks": checks,
    "control_run": report.get("baseline"),
    "candidate_run": report.get("candidate"),
    "source_update": report.get("source_update"),
    "target_update": report.get("target_update"),
    "control_core_seconds": report.get("baseline_performance", {}).get("core_seconds"),
    "candidate_core_seconds": report.get("candidate_performance", {}).get("core_seconds"),
    "candidate_physical_min_free_gb": report.get("candidate_physical_min_free_gb"),
    "rollout_trace_exact_diagnostic": (
        report.get("trace_comparison", {}).get("passed") is True
    ),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not summary["infrastructure_gate_passed"]:
    raise SystemExit(2)
PY
INFRASTRUCTURE_RC="$?"
set -e

CONTROL_REL="${CONTROL_RUN#${PROJECT_ROOT}/}"
CANDIDATE_REL="${CANDIDATE_RUN#${PROJECT_ROOT}/}"
GATE_REL="${GATE_REPORT#${PROJECT_ROOT}/}"
ARCHIVE="${PROJECT_ROOT}/m1-gradient-clip-gate-s${SOURCE_UPDATE}-${STAMP}.tar.gz"
tar -czf "${ARCHIVE}" \
  --exclude='*/checkpoints/*/runtime/*' \
  --exclude='*/traces/*' \
  -C "${PROJECT_ROOT}" \
  "${GATE_REL}" \
  "${CONTROL_REL}" \
  "${CANDIDATE_REL}"

echo "CONTROL_RUN=${CONTROL_RUN}"
echo "CANDIDATE_RUN=${CANDIDATE_RUN}"
echo "GATE_REPORT=${GATE_REPORT}"
echo "ARCHIVE=${ARCHIVE}"
exit "${INFRASTRUCTURE_RC}"
