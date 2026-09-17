#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
GROUNDING_DATA="${GROUNDING_DATA:?GROUNDING_DATA is required}"
CONTROL_TRAIN="${CONTROL_TRAIN:?CONTROL_TRAIN is required}"
CONTROL_EVAL="${CONTROL_EVAL:?CONTROL_EVAL is required}"
GPUS="${GPUS:-0,1,2}"
MINIMUM_FREE_DISK_GB="${MINIMUM_FREE_DISK_GB:-10}"
TAG="${GATE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
CANDIDATE_NAME="m1-precapture-separate-clip-s215-u225-${TAG}"
EVAL_NAME="m1-precapture-separate-clip-valid-seen-u225-${TAG}"
REPORT="${PROJECT_ROOT}/m1-precapture-clip-ab-${TAG}.json"
ARCHIVE="${PROJECT_ROOT}/m1-precapture-clip-ab-${TAG}.tar.gz"

fail() { echo "$1" >&2; exit 2; }

[[ -x "${PYTHON}" ]] || fail "PYTHON is not executable: ${PYTHON}"
export PATH="$(dirname -- "${PYTHON}"):${PATH}"
[[ "$(command -v python)" == "${PYTHON}" ]] || fail "failed to lock python"
[[ "$(basename -- "${SOURCE_CHECKPOINT}")" == "step-000215" ]] \
  || fail "source must be the committed step-215 checkpoint"
[[ "${MINIMUM_FREE_DISK_GB}" =~ ^[1-9][0-9]*$ ]] \
  || fail "MINIMUM_FREE_DISK_GB must be a positive integer"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
[[ "${#GPU_IDS[@]}" -eq 3 ]] || fail "exactly three GPUs are required"

for path in \
  "${SOURCE_CHECKPOINT}/checkpoint.complete.json" \
  "${SOURCE_CHECKPOINT}/resolved_config.json" \
  "${GROUNDING_DATA}/manifest.json" \
  "${CONTROL_TRAIN}/training_summary.json" \
  "${CONTROL_TRAIN}/resolved_config.json" \
  "${CONTROL_TRAIN}/provenance.json" \
  "${CONTROL_TRAIN}/checkpoints/step-000225/checkpoint.complete.json" \
  "${CONTROL_EVAL}/valid_seen_summary.json" \
  "${CONTROL_EVAL}/checkpoint-load.json"; do
  [[ -f "${path}" ]] || fail "required input is missing: ${path}"
done
for run in "${CONTROL_TRAIN}" "${CONTROL_EVAL}"; do
  case "${run}" in
    "${PROJECT_ROOT}"/runs/*) ;;
    *) fail "control run must be inside ${PROJECT_ROOT}/runs: ${run}" ;;
  esac
done

for gpu_id in "${GPU_IDS[@]}"; do
  [[ "${gpu_id}" =~ ^[0-9]+$ ]] || fail "invalid GPU index: ${gpu_id}"
  gpu_processes="$(nvidia-smi -i "${gpu_id}" \
    --query-compute-apps=pid,used_memory \
    --format=csv,noheader,nounits 2>/dev/null)" \
    || fail "cannot inspect GPU ${gpu_id}"
  [[ -z "${gpu_processes}" ]] \
    || fail "GPU ${gpu_id} has compute processes: ${gpu_processes}"
done
active="$(pgrep -af '[p]ython -m infoskill.cli (train|eval)' || true)"
[[ -z "${active}" ]] || fail "another INFO-SKILL job is active: ${active}"

require_disk_headroom() {
  local free_bytes
  free_bytes="$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')"
  if (( free_bytes < MINIMUM_FREE_DISK_GB * 1024 * 1024 * 1024 )); then
    df -h /root/autodl-tmp >&2
    fail "free disk is below ${MINIMUM_FREE_DISK_GB} GiB"
  fi
}

require_disk_headroom
"${PYTHON}" -c \
  'import sentence_transformers, zstandard; print("sentence-transformers:", sentence_transformers.__version__)'

"${PYTHON}" - "${SOURCE_CHECKPOINT}" "${GROUNDING_DATA}" \
  "${CONTROL_TRAIN}" "${CONTROL_EVAL}" <<'PY'
import json
import sys
from pathlib import Path

source, grounding, control, evaluation = map(Path, sys.argv[1:])
source_config = json.loads((source / "resolved_config.json").read_text(encoding="utf-8"))
control_config = json.loads((control / "resolved_config.json").read_text(encoding="utf-8"))
control_provenance = json.loads((control / "provenance.json").read_text(encoding="utf-8"))
control_summary = json.loads((control / "training_summary.json").read_text(encoding="utf-8"))
evaluation_summary = json.loads((evaluation / "valid_seen_summary.json").read_text(encoding="utf-8"))
evaluation_load = json.loads((evaluation / "checkpoint-load.json").read_text(encoding="utf-8"))
grounding_manifest = json.loads((grounding / "manifest.json").read_text(encoding="utf-8"))
options = control_config["runtime_options"]
checks = {
    "source_and_control_same_protocol": source_config == control_config,
    "joint_control_started_from_source": (
        control_provenance.get("resume_source_checkpoint") == str(source.resolve())
        and control_provenance.get("resume_forked") is True
        and control_provenance.get("invocation")
        == {"segment_start_update": 215, "segment_end_update": 225}
    ),
    "joint_control_paused_at_225": (
        control_summary.get("status") == "paused"
        and control_summary.get("global_update") == 225
        and control_summary.get("max_updates") == 445
    ),
    "control_is_graph_split_k_one_joint": (
        options.get("hybrid_prefix_cuda_graph") is True
        and options.get("lora_shrink_split_k_one") is True
        and options.get("policy_gradient_clip_mode") == "joint"
        and control_config["evaluation_manifest"]["eval_batch_size"] == 64
    ),
    "grounding_passed": grounding_manifest.get("formal_gate_passed") is True,
    "control_evaluation_complete": (
        evaluation_summary.get("is_complete") is True
        and evaluation_summary.get("evaluated") == 140
        and evaluation_summary.get("task_manifest_sha256")
        == control_config["evaluation_manifest"]["sha256"]
    ),
    "control_evaluation_loaded_control_checkpoint": (
        evaluation_load.get("status") == "loaded"
        and evaluation_load.get("checkpoint_step") == 225
        and Path(evaluation_load.get("checkpoint", "")).resolve()
        == (control / "checkpoints" / "step-000225").resolve()
    ),
}
for name, passed in checks.items():
    print(f"preflight/{name}={passed}")
if not all(checks.values()):
    raise SystemExit("paired clipping preflight failed")
PY

candidate_run=""
candidate_eval=""
archive_diagnostics() {
  local exit_code="$?"
  trap - EXIT
  set +e
  local -a members=()
  local run rel item
  if [[ -z "${candidate_run}" ]]; then
    candidate_run="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
      -name "*-${CANDIDATE_NAME}" | sort | tail -n 1)"
  fi
  if [[ -z "${candidate_eval}" ]]; then
    candidate_eval="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
      -name "*-${EVAL_NAME}" | sort | tail -n 1)"
  fi
  for run in "${CONTROL_TRAIN}" "${CONTROL_EVAL}" "${candidate_run}" "${candidate_eval}"; do
    [[ -d "${run}" ]] || continue
    rel="${run#${PROJECT_ROOT}/}"
    for item in \
      training_summary.json resolved_config.json provenance.json \
      valid_seen_summary.json valid_seen-000225-summary.json \
      checkpoint-load.json evaluation-timing.json metrics.jsonl \
      console.log traces \
      checkpoints/step-000220/checkpoint.complete.json \
      checkpoints/step-000225/checkpoint.complete.json; do
      [[ -e "${run}/${item}" ]] && members+=("${rel}/${item}")
    done
  done
  [[ -f "${REPORT}" ]] && members+=("${REPORT#${PROJECT_ROOT}/}")
  if (( ${#members[@]} > 0 )); then
    tar -czf "${ARCHIVE}" -C "${PROJECT_ROOT}" "${members[@]}"
    if [[ "$?" -eq 0 ]]; then
      echo "ARCHIVE=${ARCHIVE}"
      ls -lh "${ARCHIVE}"
    else
      echo "failed to package diagnostics: ${ARCHIVE}" >&2
    fi
  fi
  exit "${exit_code}"
}
trap archive_diagnostics EXIT

echo "[INFO-SKILL] paired clip A/B: reusing joint control, training separate 215 -> 225"
env \
  GPUS="${GPUS}" \
  PROFILE=formal \
  MAX_UPDATES=445 \
  SEGMENT_END_UPDATE=225 \
  RESUME="${SOURCE_CHECKPOINT}" \
  RUN_NAME="${CANDIDATE_NAME}" \
  GROUNDING_DATA="${GROUNDING_DATA}" \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  ENVIRONMENT_WORKERS=1 \
  POLICY_MAX_TOKENS_PER_GPU=12288 \
  ROLLOUT_MAX_BATCHED_TOKENS=16384 \
  BALANCE_POLICY_TOKENS_ACROSS_RANKS=1 \
  SKIP_UNUSED_OLD_LOGPROB_ENTROPY=0 \
  HYBRID_PREFIX_CUDA_GRAPH=1 \
  LORA_SHRINK_SPLIT_K_ONE=1 \
  FUSE_KL_PPO_FORWARD=0 \
  POLICY_GRADIENT_CLIP_MODE=separate \
  EVAL_BATCH_SIZE=64 \
  CHECKPOINT_KEEP_RECENT=5 \
  CHECKPOINT_KEEP_BEST_VALID=1 \
  CUDA_MEMORY_POLL_INTERVAL_MS=1000 \
  INFO_SKILL_CPU_THREADS=1 \
  OMP_NUM_THREADS=1 \
  bash scripts/run_alfworld.sh train infoskill

candidate_run="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
  -name "*-${CANDIDATE_NAME}" | sort | tail -n 1)"
[[ -d "${candidate_run}" ]] || fail "candidate training run was not created"
"${PYTHON}" - "${candidate_run}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
if summary.get("global_update") != 225 or summary.get("status") != "paused":
    raise SystemExit(f"candidate did not pause at update 225: {summary}")
for step in (220, 225):
    marker = run / "checkpoints" / f"step-{step:06d}" / "checkpoint.complete.json"
    if not marker.is_file():
        raise SystemExit(f"missing committed checkpoint: {marker}")
rows = [
    json.loads(line)
    for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
training = [row for row in rows if row.get("phase") == "train"]
if {row["step"] for row in training} != set(range(216, 226)) or not all(
    row.get("policy/separate_gradient_clipping") == 1.0 for row in training
):
    raise SystemExit("candidate did not execute separate clipping in all ten updates")
PY
echo "CANDIDATE_TRAIN=${candidate_run}"

require_disk_headroom
echo "[INFO-SKILL] independent fixed 140-task valid_seen evaluation: separate update=225"
env \
  GPUS="${GPUS}" \
  EVAL_BACKEND=verl \
  CHECKPOINT_STEP=225 \
  POLICY_CHECKPOINT="${candidate_run}/checkpoints/step-000225" \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  ENVIRONMENT_WORKERS=1 \
  GROUPED_INFOSKILL_CONDITIONING=0 \
  HYBRID_PREFIX_CUDA_GRAPH=1 \
  LORA_SHRINK_SPLIT_K_ONE=1 \
  EVAL_BATCH_SIZE=64 \
  CUDA_MEMORY_POLL_INTERVAL_MS=200 \
  INFO_SKILL_CPU_THREADS=1 \
  OMP_NUM_THREADS=1 \
  RUN_NAME="${EVAL_NAME}" \
  bash scripts/run_alfworld.sh eval infoskill

candidate_eval="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
  -name "*-${EVAL_NAME}" | sort | tail -n 1)"
[[ -f "${candidate_eval}/valid_seen_summary.json" ]] \
  || fail "candidate update-225 evaluation did not finish"
echo "CANDIDATE_EVAL=${candidate_eval}"

"${PYTHON}" scripts/compare_m1_precapture_clip_ab.py \
  "${SOURCE_CHECKPOINT}" "${CONTROL_TRAIN}" "${CONTROL_EVAL}" \
  "${candidate_run}" "${candidate_eval}" --output "${REPORT}"
echo "REPORT=${REPORT}"
echo "[INFO-SKILL] paired clip A/B complete; report is exploratory, not automatic adoption"
