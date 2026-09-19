#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
GPUS="${GPUS:-0,1,2}"
TARGET_DELTA_UPDATES="${TARGET_DELTA_UPDATES:-5}"
MINIMUM_FREE_DISK_GB="${MINIMUM_FREE_DISK_GB:-8}"
TAG="${SWEEP_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
REPORT="${PROJECT_ROOT}/m0-lora-lr-sweep-${TAG}.json"
ARCHIVE="${PROJECT_ROOT}/m0-lora-lr-sweep-${TAG}.tar.gz"
LEARNING_RATES=(1e-6 3e-6 1e-5)

fail() { echo "$1" >&2; exit 2; }

[[ -x "${PYTHON}" ]] || fail "PYTHON is not executable: ${PYTHON}"
export PATH="$(dirname -- "${PYTHON}"):${PATH}"
[[ "$(command -v python)" == "${PYTHON}" ]] || fail "failed to lock python"
[[ "${TARGET_DELTA_UPDATES}" =~ ^[1-9][0-9]*$ ]] \
  || fail "TARGET_DELTA_UPDATES must be a positive integer"
[[ "${MINIMUM_FREE_DISK_GB}" =~ ^[1-9][0-9]*$ ]] \
  || fail "MINIMUM_FREE_DISK_GB must be a positive integer"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
[[ "${#GPU_IDS[@]}" -eq 3 ]] || fail "exactly three GPUs are required"
for gpu_id in "${GPU_IDS[@]}"; do
  [[ "${gpu_id}" =~ ^[0-9]+$ ]] || fail "invalid GPU index: ${gpu_id}"
done

for required in \
  "${SOURCE_CHECKPOINT}/checkpoint.complete.json" \
  "${SOURCE_CHECKPOINT}/resolved_config.json" \
  "${SOURCE_CHECKPOINT}/provenance.json" \
  "${SOURCE_CHECKPOINT}/trainer_state.json"; do
  [[ -f "${required}" ]] || fail "required source file is missing: ${required}"
done
case "${SOURCE_CHECKPOINT}" in
  "${PROJECT_ROOT}"/runs/*/checkpoints/step-*) ;;
  *) fail "SOURCE_CHECKPOINT must be inside ${PROJECT_ROOT}/runs" ;;
esac

read -r SOURCE_UPDATE TARGET_UPDATE MAX_UPDATES < <(
  "${PYTHON}" - "${SOURCE_CHECKPOINT}" "${TARGET_DELTA_UPDATES}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
delta = int(sys.argv[2])
marker = json.loads((source / "checkpoint.complete.json").read_text(encoding="utf-8"))
config = json.loads((source / "resolved_config.json").read_text(encoding="utf-8"))
start = int(marker["global_update"])
target = start + delta
maximum = int(config["training_plan"]["max_updates"])
checks = {
    "portable_checkpoint": marker.get("portable") is True,
    "non_emergency_checkpoint": marker.get("emergency") is not True,
    "m0_no_skill": config.get("mode") == "no_skill",
    "formal_profile": config["training_plan"].get("profile") == "formal",
    "three_gpus": config.get("num_gpus") == 3,
    "fixed_eval_140_batch64": (
        config.get("evaluation_manifest", {}).get("task_count") == 140
        and config.get("evaluation_manifest", {}).get("eval_batch_size") == 64
    ),
    "target_within_plan": target <= maximum,
    "checkpoint_boundary": target % int(config["training_plan"]["checkpoint_every"]) == 0,
}
for name, passed in checks.items():
    print(f"preflight/{name}={passed}", file=sys.stderr)
if not all(checks.values()):
    raise SystemExit("M0 LoRA LR sweep source preflight failed")
print(start, target, maximum)
PY
)

active="$(pgrep -af '[p]ython -m infoskill.cli (train|eval)' || true)"
[[ -z "${active}" ]] || fail "another INFO-SKILL job is active: ${active}"
for gpu_id in "${GPU_IDS[@]}"; do
  gpu_processes="$(nvidia-smi -i "${gpu_id}" \
    --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)" \
    || fail "cannot inspect GPU ${gpu_id}"
  [[ -z "${gpu_processes}" ]] \
    || fail "GPU ${gpu_id} has compute processes: ${gpu_processes}"
done

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

declare -a TRAIN_RUNS=()
declare -a EVAL_RUNS=()
CURRENT_TRAIN=""
CURRENT_EVAL=""

archive_diagnostics() {
  local exit_code="$?"
  trap - EXIT
  set +e
  local -a members=()
  local run rel item
  for run in "${TRAIN_RUNS[@]}" "${EVAL_RUNS[@]}" "${CURRENT_TRAIN}" "${CURRENT_EVAL}"; do
    [[ -d "${run}" ]] || continue
    rel="${run#${PROJECT_ROOT}/}"
    for item in \
      training_summary.json resolved_config.json provenance.json \
      valid_seen_summary.json checkpoint-load.json evaluation-timing.json \
      metrics.jsonl console.log traces \
      "checkpoints/step-$(printf '%06d' "${TARGET_UPDATE}")/checkpoint.complete.json"; do
      [[ -e "${run}/${item}" ]] && members+=("${rel}/${item}")
    done
  done
  [[ -f "${REPORT}" ]] && members+=("${REPORT#${PROJECT_ROOT}/}")
  if (( ${#members[@]} > 0 )); then
    mapfile -t members < <(printf '%s\n' "${members[@]}" | awk '!seen[$0]++')
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

find_run() {
  local name="$1"
  find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
    -name "*-${name}" | sort | tail -n 1
}

train_is_reusable() {
  local run="$1" rate="$2"
  [[ -f "${run}/training_summary.json" ]] || return 1
  [[ -f "${run}/checkpoints/step-$(printf '%06d' "${TARGET_UPDATE}")/checkpoint.complete.json" ]] || return 1
  "${PYTHON}" - "${run}" "${TARGET_UPDATE}" "${rate}" <<'PY'
import json, math, sys
from pathlib import Path
run, target, rate = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
ok = (
    summary.get("status") == "paused"
    and summary.get("global_update") == target
    and math.isclose(config["runtime_options"]["actor_learning_rate"], rate)
)
raise SystemExit(0 if ok else 1)
PY
}

evaluation_is_reusable() {
  local run="$1" checkpoint="$2"
  [[ -f "${run}/valid_seen_summary.json" && -f "${run}/checkpoint-load.json" ]] || return 1
  "${PYTHON}" - "${run}" "${checkpoint}" "${TARGET_UPDATE}" <<'PY'
import json, sys
from pathlib import Path
run, checkpoint, step = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), int(sys.argv[3])
summary = json.loads((run / "valid_seen_summary.json").read_text(encoding="utf-8"))
loaded = json.loads((run / "checkpoint-load.json").read_text(encoding="utf-8"))
ok = (
    summary.get("is_complete") is True
    and summary.get("evaluated") == 140
    and loaded.get("status") == "loaded"
    and loaded.get("checkpoint_step") == step
    and Path(loaded.get("checkpoint", "")).resolve() == checkpoint
)
raise SystemExit(0 if ok else 1)
PY
}

for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
  RATE_TAG="${LEARNING_RATE//-/m}"
  TRAIN_NAME="m0-lora-lr-${RATE_TAG}-s${SOURCE_UPDATE}-u${TARGET_UPDATE}-${TAG}"
  EVAL_NAME="m0-lora-lr-${RATE_TAG}-valid-seen-u${TARGET_UPDATE}-${TAG}"
  CURRENT_TRAIN="$(find_run "${TRAIN_NAME}")"
  if [[ -n "${CURRENT_TRAIN}" ]] && train_is_reusable "${CURRENT_TRAIN}" "${LEARNING_RATE}"; then
    echo "[INFO-SKILL] reusing completed training branch: ${CURRENT_TRAIN}"
  else
    require_disk_headroom
    echo "[INFO-SKILL] M0 LoRA LR=${LEARNING_RATE}: update ${SOURCE_UPDATE} -> ${TARGET_UPDATE}"
    env \
      GPUS="${GPUS}" \
      PROFILE=formal \
      MAX_UPDATES="${MAX_UPDATES}" \
      SEGMENT_END_UPDATE="${TARGET_UPDATE}" \
      RESUME="${SOURCE_CHECKPOINT}" \
      RUN_NAME="${TRAIN_NAME}" \
      ACTOR_LEARNING_RATE="${LEARNING_RATE}" \
      PERSISTENT_ROLLOUT_SESSION=1 \
      ENVIRONMENT_BACKEND=native_batch \
      ENVIRONMENT_WORKERS=1 \
      POLICY_MAX_TOKENS_PER_GPU=12288 \
      ROLLOUT_MAX_BATCHED_TOKENS=16384 \
      BALANCE_POLICY_TOKENS_ACROSS_RANKS=1 \
      SKIP_UNUSED_OLD_LOGPROB_ENTROPY=0 \
      HYBRID_PREFIX_CUDA_GRAPH=0 \
      LORA_SHRINK_SPLIT_K_ONE=0 \
      FUSE_KL_PPO_FORWARD=0 \
      POLICY_GRADIENT_CLIP_MODE=joint \
      EVAL_BATCH_SIZE=64 \
      CHECKPOINT_KEEP_RECENT=1 \
      CHECKPOINT_KEEP_BEST_VALID=0 \
      CUDA_MEMORY_POLL_INTERVAL_MS=1000 \
      INFO_SKILL_CPU_THREADS=1 \
      OMP_NUM_THREADS=1 \
      bash scripts/run_alfworld.sh train no_skill
    CURRENT_TRAIN="$(find_run "${TRAIN_NAME}")"
    train_is_reusable "${CURRENT_TRAIN}" "${LEARNING_RATE}" \
      || fail "training branch did not finish correctly: ${CURRENT_TRAIN}"
  fi
  TRAIN_RUNS+=("${CURRENT_TRAIN}")

  CHECKPOINT="${CURRENT_TRAIN}/checkpoints/step-$(printf '%06d' "${TARGET_UPDATE}")"
  CURRENT_EVAL="$(find_run "${EVAL_NAME}")"
  if [[ -n "${CURRENT_EVAL}" ]] && evaluation_is_reusable "${CURRENT_EVAL}" "${CHECKPOINT}"; then
    echo "[INFO-SKILL] reusing completed evaluation: ${CURRENT_EVAL}"
  else
    require_disk_headroom
    echo "[INFO-SKILL] fixed 140-task evaluation: LR=${LEARNING_RATE} update=${TARGET_UPDATE}"
    env \
      GPUS="${GPUS}" \
      EVAL_BACKEND=verl \
      CHECKPOINT_STEP="${TARGET_UPDATE}" \
      POLICY_CHECKPOINT="${CHECKPOINT}" \
      PERSISTENT_ROLLOUT_SESSION=1 \
      ENVIRONMENT_BACKEND=native_batch \
      ENVIRONMENT_WORKERS=1 \
      EVAL_BATCH_SIZE=64 \
      CUDA_MEMORY_POLL_INTERVAL_MS=200 \
      INFO_SKILL_CPU_THREADS=1 \
      OMP_NUM_THREADS=1 \
      RUN_NAME="${EVAL_NAME}" \
      bash scripts/run_alfworld.sh eval no_skill
    CURRENT_EVAL="$(find_run "${EVAL_NAME}")"
    evaluation_is_reusable "${CURRENT_EVAL}" "${CHECKPOINT}" \
      || fail "evaluation did not finish correctly: ${CURRENT_EVAL}"
  fi
  EVAL_RUNS+=("${CURRENT_EVAL}")
done

"${PYTHON}" scripts/compare_m0_lora_lr_sweep.py \
  "${SOURCE_CHECKPOINT}" \
  --train-runs "${TRAIN_RUNS[@]}" \
  --eval-runs "${EVAL_RUNS[@]}" \
  --expected-task-count 140 \
  --output "${REPORT}"
echo "REPORT=${REPORT}"
echo "[INFO-SKILL] M0 LoRA learning-rate sweep complete"
