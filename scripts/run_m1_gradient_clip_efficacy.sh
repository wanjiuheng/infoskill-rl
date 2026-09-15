#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONTROL_RUN="${CONTROL_RUN:-}"
CANDIDATE_RUN="${CANDIDATE_RUN:-}"
GROUNDING_DATA="${GROUNDING_DATA:-}"
PYTHON="${PYTHON:-$(command -v python)}"
GPUS="${GPUS:-0,1,2}"
TARGET_UPDATE="${TARGET_UPDATE:-205}"
MINIMUM_FREE_DISK_GB="${MINIMUM_FREE_DISK_GB:-12}"
MINIMUM_MACRO_DELTA="${MINIMUM_MACRO_DELTA:-0.03}"
MINIMUM_OVERALL_DELTA="${MINIMUM_OVERALL_DELTA:-0.0}"

fail() {
  echo "$1" >&2
  exit 2
}

[[ -x "${PYTHON}" ]] || fail "PYTHON must point to an executable interpreter"
export PATH="$(dirname -- "${PYTHON}"):${PATH}"
[[ "$(command -v python)" == "${PYTHON}" ]] || fail "failed to lock python"
"${PYTHON}" -c \
  'import sentence_transformers; print("sentence-transformers:", sentence_transformers.__version__)'

[[ -n "${CONTROL_RUN}" ]] || fail "CONTROL_RUN is required"
[[ -n "${CANDIDATE_RUN}" ]] || fail "CANDIDATE_RUN is required"
[[ -f "${GROUNDING_DATA}/manifest.json" ]] \
  || fail "GROUNDING_DATA must be a finalized grounding run"
[[ "${TARGET_UPDATE}" =~ ^[1-9][0-9]*$ ]] \
  || fail "TARGET_UPDATE must be a positive integer"
[[ "${MINIMUM_FREE_DISK_GB}" =~ ^[1-9][0-9]*$ ]] \
  || fail "MINIMUM_FREE_DISK_GB must be a positive integer"
"${PYTHON}" - "${MINIMUM_MACRO_DELTA}" "${MINIMUM_OVERALL_DELTA}" <<'PY'
import math
import sys

for name, raw in zip(("MINIMUM_MACRO_DELTA", "MINIMUM_OVERALL_DELTA"), sys.argv[1:]):
    try:
        value = float(raw)
    except ValueError as error:
        raise SystemExit(f"{name} must be a finite number") from error
    if not math.isfinite(value):
        raise SystemExit(f"{name} must be a finite number")
PY

current_update() {
  "${PYTHON}" - "$1/training_summary.json" <<'PY'
import json
import sys

print(int(json.load(open(sys.argv[1], encoding="utf-8"))["global_update"]))
PY
}

CONTROL_UPDATE="$(current_update "${CONTROL_RUN}")"
CANDIDATE_UPDATE="$(current_update "${CANDIDATE_RUN}")"
(( CONTROL_UPDATE <= TARGET_UPDATE )) \
  || fail "control is already beyond TARGET_UPDATE"
(( CANDIDATE_UPDATE <= TARGET_UPDATE )) \
  || fail "candidate is already beyond TARGET_UPDATE"

for REQUIRED in \
  "${CONTROL_RUN}/checkpoints/step-$(printf '%06d' "${CONTROL_UPDATE}")/checkpoint.complete.json" \
  "${CANDIDATE_RUN}/checkpoints/step-$(printf '%06d' "${CANDIDATE_UPDATE}")/checkpoint.complete.json"
do
  [[ -f "${REQUIRED}" ]] || fail "missing committed checkpoint: ${REQUIRED}"
done

ACTIVE="$(pgrep -af '[p]ython -m infoskill.cli train' || true)"
if [[ -n "${ACTIVE}" ]]; then
  echo "another INFO-SKILL training process is active:" >&2
  echo "${ACTIVE}" >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="${PROJECT_ROOT}/m1-gradient-clip-efficacy-u${TARGET_UPDATE}-${STAMP}.tar.gz"
STATE_KEY="$(
  "${PYTHON}" - "${CONTROL_RUN}" "${CANDIDATE_RUN}" "${TARGET_UPDATE}" <<'PY'
import hashlib
import sys

print(hashlib.sha256("\0".join(sys.argv[1:]).encode()).hexdigest()[:16])
PY
)"
STATE_DIR="${PROJECT_ROOT}/runs/.m1-gradient-clip-efficacy-${STATE_KEY}"
mkdir -p "${STATE_DIR}"

archive_diagnostics() {
  local exit_code="$?"
  trap - EXIT
  set +e
  local control_relative="${CONTROL_RUN#${PROJECT_ROOT}/}"
  local candidate_relative="${CANDIDATE_RUN#${PROJECT_ROOT}/}"
  local state_relative="${STATE_DIR#${PROJECT_ROOT}/}"
  local -a members=(
    "${control_relative}"
    "${candidate_relative}"
    "${state_relative}"
  )
  local state_file evaluation_run evaluation_relative
  for state_file in \
    "${STATE_DIR}/joint-evaluation-run.txt" \
    "${STATE_DIR}/separate-evaluation-run.txt"
  do
    if [[ -s "${state_file}" ]]; then
      evaluation_run="$(<"${state_file}")"
      if [[ -d "${evaluation_run}" ]]; then
        evaluation_relative="${evaluation_run#${PROJECT_ROOT}/}"
        members+=("${evaluation_relative}")
      fi
    fi
  done
  tar -czf "${ARCHIVE}" \
    --exclude='*/checkpoints/*/runtime/*' \
    --exclude='*/traces/*' \
    -C "${PROJECT_ROOT}" \
    "${members[@]}"
  local archive_code="$?"
  if (( archive_code == 0 )); then
    echo "ARCHIVE=${ARCHIVE}"
  else
    echo "failed to create diagnostic archive: ${ARCHIVE}" >&2
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

run_branch() {
  local run="$1"
  local source_update="$2"
  local clip_mode="$3"
  local checkpoint
  checkpoint="${run}/checkpoints/step-$(printf '%06d' "${source_update}")"

  if (( source_update == TARGET_UPDATE )); then
    echo "[INFO-SKILL] gradient-clip efficacy: ${clip_mode} already at update ${TARGET_UPDATE}"
    return
  fi

  require_disk_headroom
  echo "[INFO-SKILL] gradient-clip efficacy: ${clip_mode}, update ${source_update}->${TARGET_UPDATE}"
  env \
    GPUS="${GPUS}" \
    PROFILE=formal \
    MAX_UPDATES=445 \
    SEGMENT_END_UPDATE="${TARGET_UPDATE}" \
    RESUME="${checkpoint}" \
    RUN_NAME="" \
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
    OMP_NUM_THREADS=1 \
    bash scripts/run_alfworld.sh train infoskill
}

run_branch "${CONTROL_RUN}" "${CONTROL_UPDATE}" joint
run_branch "${CANDIDATE_RUN}" "${CANDIDATE_UPDATE}" separate

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
    echo "[INFO-SKILL] reusing completed ${clip_mode} evaluation: $(<"${state_file}")"
    return
  fi

  require_disk_headroom
  local run_name="m1-gradient-clip-efficacy-${clip_mode}-u${TARGET_UPDATE}-${STAMP}"
  echo "[INFO-SKILL] fixed valid_seen evaluation: ${clip_mode}, update ${TARGET_UPDATE}"
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
    || fail "failed to resolve ${clip_mode} evaluation run"
  printf '%s\n' "${evaluation_run}" >"${state_file}"
  evaluation_is_reusable "${state_file}" "${checkpoint}" \
    || fail "${clip_mode} evaluation did not produce a complete fixed-140 result"
}

run_evaluation "${CONTROL_RUN}" joint
run_evaluation "${CANDIDATE_RUN}" separate

CONTROL_EVAL="$(<"${STATE_DIR}/joint-evaluation-run.txt")"
CANDIDATE_EVAL="$(<"${STATE_DIR}/separate-evaluation-run.txt")"
COMPARISON="${STATE_DIR}/efficacy-comparison.json"
DECISION="${STATE_DIR}/efficacy-decision.json"

set +e
"${PYTHON}" scripts/compare_infoskill_optimization_efficacy.py \
  "${CONTROL_EVAL}" \
  "${CANDIDATE_EVAL}" \
  --expected-task-count 140 \
  | tee "${COMPARISON}"
COMPARISON_RC="${PIPESTATUS[0]}"
set -e

set +e
"${PYTHON}" - \
  "${COMPARISON}" \
  "${DECISION}" \
  "${COMPARISON_RC}" \
  "${MINIMUM_MACRO_DELTA}" \
  "${MINIMUM_OVERALL_DELTA}" <<'PY'
import json
import sys
from pathlib import Path

comparison_path = Path(sys.argv[1])
decision_path = Path(sys.argv[2])
comparison_rc = int(sys.argv[3])
minimum_macro_delta = float(sys.argv[4])
minimum_overall_delta = float(sys.argv[5])
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
macro_delta = float(comparison["candidate_minus_baseline"]["macro_success"])
overall_delta = float(comparison["candidate_minus_baseline"]["overall_success"])
controls_valid = comparison.get("controls_valid") is True

if not controls_valid:
    classification = "invalid_controls"
    exit_code = 2
elif macro_delta >= minimum_macro_delta and overall_delta >= minimum_overall_delta:
    classification = "candidate_selected"
    exit_code = 0
elif -minimum_macro_delta < macro_delta < minimum_macro_delta:
    classification = "repeat_required"
    exit_code = 3
else:
    classification = "candidate_rejected"
    exit_code = 4

decision = {
    "schema_version": 1,
    "classification": classification,
    "controls_valid": controls_valid,
    "generic_comparator_rc": comparison_rc,
    "minimum_macro_delta": minimum_macro_delta,
    "minimum_overall_delta": minimum_overall_delta,
    "macro_delta": macro_delta,
    "overall_delta": overall_delta,
    "control_run": comparison["baseline"]["run"],
    "candidate_run": comparison["candidate"]["run"],
    "exit_code": exit_code,
}
decision_path.write_text(
    json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(decision, ensure_ascii=False, indent=2))
raise SystemExit(exit_code)
PY
DECISION_RC="$?"
set -e

echo "CONTROL_EVAL=${CONTROL_EVAL}"
echo "CANDIDATE_EVAL=${CANDIDATE_EVAL}"
echo "EFFICACY_COMPARISON=${COMPARISON}"
echo "EFFICACY_DECISION=${DECISION}"
exit "${DECISION_RC}"
