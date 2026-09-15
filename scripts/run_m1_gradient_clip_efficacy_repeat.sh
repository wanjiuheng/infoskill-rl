#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONTROL_RUN="${CONTROL_RUN:-}"
CANDIDATE_RUN="${CANDIDATE_RUN:-}"
ORIGINAL_CONTROL_EVAL="${ORIGINAL_CONTROL_EVAL:-}"
ORIGINAL_CANDIDATE_EVAL="${ORIGINAL_CANDIDATE_EVAL:-}"
PYTHON="${PYTHON:-$(command -v python)}"
GPUS="${GPUS:-0,1,2}"
TARGET_UPDATE="${TARGET_UPDATE:-205}"
MINIMUM_FREE_DISK_GB="${MINIMUM_FREE_DISK_GB:-12}"

fail() {
  echo "$1" >&2
  exit 2
}

[[ -x "${PYTHON}" ]] || fail "PYTHON must point to an executable interpreter"
export PATH="$(dirname -- "${PYTHON}"):${PATH}"
[[ "$(command -v python)" == "${PYTHON}" ]] || fail "failed to lock python"
"${PYTHON}" -c \
  'import sentence_transformers; print("sentence-transformers:", sentence_transformers.__version__)'

for REQUIRED in \
  "${CONTROL_RUN}/checkpoints/step-$(printf '%06d' "${TARGET_UPDATE}")/checkpoint.complete.json" \
  "${CANDIDATE_RUN}/checkpoints/step-$(printf '%06d' "${TARGET_UPDATE}")/checkpoint.complete.json" \
  "${ORIGINAL_CONTROL_EVAL}/valid_seen_summary.json" \
  "${ORIGINAL_CANDIDATE_EVAL}/valid_seen_summary.json"
do
  [[ -f "${REQUIRED}" ]] || fail "missing required input: ${REQUIRED}"
done

ACTIVE="$(
  pgrep -af \
    '[p]ython -m infoskill.cli train|[p]ython -m infoskill.cli eval' || true
)"
if [[ -n "${ACTIVE}" ]]; then
  echo "another INFO-SKILL train/eval process is active:" >&2
  echo "${ACTIVE}" >&2
  exit 2
fi

STATE_KEY="$(
  "${PYTHON}" - \
    "${CONTROL_RUN}" \
    "${CANDIDATE_RUN}" \
    "${ORIGINAL_CONTROL_EVAL}" \
    "${ORIGINAL_CANDIDATE_EVAL}" <<'PY'
import hashlib
import sys

print(hashlib.sha256("\0".join(sys.argv[1:]).encode()).hexdigest()[:16])
PY
)"
STATE_DIR="${PROJECT_ROOT}/runs/.m1-gradient-clip-repeat-${STATE_KEY}"
mkdir -p "${STATE_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="${PROJECT_ROOT}/m1-gradient-clip-repeat-u${TARGET_UPDATE}-${STAMP}.tar.gz"

archive_diagnostics() {
  local exit_code="$?"
  trap - EXIT
  set +e
  local -a members=("${STATE_DIR#${PROJECT_ROOT}/}")
  local run state_file
  for run in "${ORIGINAL_CONTROL_EVAL}" "${ORIGINAL_CANDIDATE_EVAL}"; do
    members+=("${run#${PROJECT_ROOT}/}")
  done
  for state_file in \
    "${STATE_DIR}/joint-evaluation-run.txt" \
    "${STATE_DIR}/separate-evaluation-run.txt"
  do
    if [[ -s "${state_file}" ]]; then
      run="$(<"${state_file}")"
      [[ -d "${run}" ]] && members+=("${run#${PROJECT_ROOT}/}")
    fi
  done
  tar -czf "${ARCHIVE}" \
    --exclude='*/traces/*' \
    -C "${PROJECT_ROOT}" \
    "${members[@]}"
  if [[ "$?" == 0 ]]; then
    echo "ARCHIVE=${ARCHIVE}"
  else
    echo "failed to create repeat archive: ${ARCHIVE}" >&2
  fi
  exit "${exit_code}"
}
trap archive_diagnostics EXIT

require_disk_headroom() {
  local free_bytes minimum_free_bytes
  free_bytes="$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')"
  minimum_free_bytes="$((MINIMUM_FREE_DISK_GB * 1024 * 1024 * 1024))"
  if (( free_bytes < minimum_free_bytes )); then
    echo "free disk is below ${MINIMUM_FREE_DISK_GB} GiB" >&2
    df -h /root/autodl-tmp >&2
    exit 2
  fi
}

evaluation_is_reusable() {
  local state_file="$1"
  local checkpoint="$2"
  [[ -s "${state_file}" ]] || return 1
  local evaluation_run
  evaluation_run="$(<"${state_file}")"
  [[ -f "${evaluation_run}/valid_seen_summary.json" ]] || return 1
  [[ -f "${evaluation_run}/checkpoint-load.json" ]] || return 1
  "${PYTHON}" - "${evaluation_run}" "${checkpoint}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
checkpoint = str(Path(sys.argv[2]).resolve())
summary = json.loads((run / "valid_seen_summary.json").read_text(encoding="utf-8"))
loaded = json.loads((run / "checkpoint-load.json").read_text(encoding="utf-8"))
valid = (
    summary.get("is_complete") is True
    and summary.get("evaluated") == 140
    and loaded.get("status") == "loaded"
    and loaded.get("loaded") is True
    and str(Path(loaded.get("checkpoint", "")).resolve()) == checkpoint
)
raise SystemExit(0 if valid else 1)
PY
}

run_evaluation() {
  local training_run="$1"
  local clip_mode="$2"
  local state_file="${STATE_DIR}/${clip_mode}-evaluation-run.txt"
  local checkpoint="${training_run}/checkpoints/step-$(printf '%06d' "${TARGET_UPDATE}")"
  if evaluation_is_reusable "${state_file}" "${checkpoint}"; then
    echo "[INFO-SKILL] reusing repeated ${clip_mode} evaluation: $(<"${state_file}")"
    return
  fi

  require_disk_headroom
  local run_name="m1-gradient-clip-repeat-${clip_mode}-u${TARGET_UPDATE}-${STAMP}"
  echo "[INFO-SKILL] repeated fixed valid_seen evaluation: ${clip_mode}"
  env \
    GPUS="${GPUS}" \
    EVAL_BACKEND=verl \
    CHECKPOINT_STEP="${TARGET_UPDATE}" \
    POLICY_CHECKPOINT="${checkpoint}" \
    PERSISTENT_ROLLOUT_SESSION=1 \
    ENVIRONMENT_BACKEND=native_batch \
    ENVIRONMENT_WORKERS=1 \
    GROUPED_INFOSKILL_CONDITIONING=0 \
    HYBRID_PREFIX_CUDA_GRAPH=1 \
    EVAL_BATCH_SIZE=64 \
    CUDA_MEMORY_POLL_INTERVAL_MS=1000 \
    INFO_SKILL_CPU_THREADS=1 \
    OMP_NUM_THREADS=1 \
    RUN_NAME="${run_name}" \
    bash scripts/run_alfworld.sh eval infoskill

  local evaluation_run
  evaluation_run="$(
    find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
      -name "*-${run_name}" | sort | tail -n 1
  )"
  [[ -n "${evaluation_run}" ]] \
    || fail "failed to resolve repeated ${clip_mode} evaluation"
  printf '%s\n' "${evaluation_run}" >"${state_file}"
  evaluation_is_reusable "${state_file}" "${checkpoint}" \
    || fail "repeated ${clip_mode} evaluation is incomplete"
}

run_evaluation "${CONTROL_RUN}" joint
run_evaluation "${CANDIDATE_RUN}" separate

REPEAT_CONTROL_EVAL="$(<"${STATE_DIR}/joint-evaluation-run.txt")"
REPEAT_CANDIDATE_EVAL="$(<"${STATE_DIR}/separate-evaluation-run.txt")"
REPORT="${STATE_DIR}/aggregate-efficacy-decision.json"

set +e
"${PYTHON}" scripts/compare_infoskill_efficacy_repeats.py \
  --pair "${ORIGINAL_CONTROL_EVAL}" "${ORIGINAL_CANDIDATE_EVAL}" \
  --pair "${REPEAT_CONTROL_EVAL}" "${REPEAT_CANDIDATE_EVAL}" \
  --minimum-macro-delta 0.03 \
  --minimum-overall-delta 0.0 \
  --expected-task-count 140 \
  | tee "${REPORT}"
DECISION_RC="${PIPESTATUS[0]}"
set -e

echo "REPEAT_CONTROL_EVAL=${REPEAT_CONTROL_EVAL}"
echo "REPEAT_CANDIDATE_EVAL=${REPEAT_CANDIDATE_EVAL}"
echo "AGGREGATE_DECISION=${REPORT}"
exit "${DECISION_RC}"
