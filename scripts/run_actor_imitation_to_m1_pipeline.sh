#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}"
CONFIG="${CONFIG:-configs/alfworld_qwen25_7b.yaml}"
BASE_MODEL_ID="${BASE_MODEL_ID:-qwen2.5-7b-instruct}"
SFT_OUTPUT="${SFT_OUTPUT:-${PROJECT_ROOT}/runs/actor-imitation-warmstart}"
HANDOFF="${HANDOFF:-${PROJECT_ROOT}/artifacts/m1-handoff}"
GROUNDING_DATA="${GROUNDING_DATA:-${PROJECT_ROOT}/artifacts/alfworld-imitation-data-grounding}"
TARGET_UPDATES="${TARGET_UPDATES:-200}"
GPUS="${GPUS:-0,1,2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-3}"
EXPECTED_EFFECTIVE_BATCH_SIZE="${EXPECTED_EFFECTIVE_BATCH_SIZE:-48}"
MINIMUM_FREE_DISK_GB="${MINIMUM_FREE_DISK_GB:-12}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PIPELINE_STAMP="${PIPELINE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-m1-handoff-valid-seen-update0-${PIPELINE_STAMP}}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-m1-handoff-formal-u${TARGET_UPDATES}-${PIPELINE_STAMP}}"

export PYTHON
export PATH="$(dirname -- "${PYTHON}"):${PATH}"

log() {
  printf '%s | %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  [[ -f "$1" ]] || {
    log "required file is missing: $1"
    return 1
  }
}

require_free_disk() {
  local available_bytes minimum_bytes
  available_bytes="$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')"
  minimum_bytes=$((MINIMUM_FREE_DISK_GB * 1024 * 1024 * 1024))
  if (( available_bytes < minimum_bytes )); then
    log "free disk is below ${MINIMUM_FREE_DISK_GB} GiB; stopping pipeline"
    df -h /root/autodl-tmp
    return 1
  fi
}

wait_for_idle_gpus() {
  local attempts="${1:-20}" active attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    active="$(nvidia-smi --query-compute-apps=pid,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -z "${active}" ]]; then
      return 0
    fi
    log "waiting for GPU processes to exit (${attempt}/${attempts}): ${active//$'\n'/; }"
    sleep 15
  done
  log "GPU processes remained after the bounded wait"
  return 1
}

find_latest_imitation_pid_file() {
  find "${PROJECT_ROOT}/logs" -maxdepth 1 -type f \
    -name 'actor-imitation-warmstart-*.log.pid' -print \
    | sort | tail -n 1
}

wait_for_imitation() {
  local manifest pid_file pid command_line checks=0
  manifest="${SFT_OUTPUT}/final-adapter/imitation-training-manifest.json"
  if [[ -f "${manifest}" ]]; then
    log "actor imitation is already complete"
  else
    pid_file="${IMITATION_PID_FILE:-$(find_latest_imitation_pid_file)}"
    require_file "${pid_file}"
    pid="$(tr -d '[:space:]' <"${pid_file}")"
    [[ "${pid}" =~ ^[0-9]+$ ]] || {
      log "invalid imitation PID in ${pid_file}: ${pid}"
      return 1
    }
    log "waiting for actor imitation PID=${pid}"

    while [[ -r "/proc/${pid}/cmdline" ]]; do
      command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
      if [[ "${command_line}" != *"run_actor_imitation.sh train"* ]]; then
        log "PID ${pid} no longer belongs to actor imitation"
        break
      fi
      checks=$((checks + 1))
      if (( checks % 5 == 0 )); then
        log "actor imitation is still running"
      fi
      sleep "${POLL_SECONDS}"
    done
  fi

  require_file "${manifest}"
  "${PYTHON}" - "${manifest}" "${EXPECTED_WORLD_SIZE}" \
    "${EXPECTED_EFFECTIVE_BATCH_SIZE}" "${BASE_MODEL_ID}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected_world_size = int(sys.argv[2])
expected_effective_batch_size = int(sys.argv[3])
expected_model_id = sys.argv[4]
assert manifest.get("status") == "complete", manifest
assert manifest.get("world_size") == expected_world_size, manifest
assert manifest.get("effective_batch_size") == expected_effective_batch_size, manifest
assert manifest.get("policy_model", {}).get("model_id") == expected_model_id, manifest
print(json.dumps({
    "actor_imitation_status": manifest["status"],
    "world_size": manifest["world_size"],
    "effective_batch_size": manifest["effective_batch_size"],
    "learning_rate": manifest["learning_rate"],
    "epochs": manifest["epochs"],
}, indent=2))
PY
  log "actor imitation manifest passed"
}

finalize_handoff() {
  if [[ -f "${HANDOFF}/checkpoint.complete.json" ]]; then
    log "immutable handoff already exists: ${HANDOFF}"
    return 0
  fi
  if [[ -e "${HANDOFF}" ]]; then
    log "handoff path exists but is incomplete; refusing to overwrite: ${HANDOFF}"
    return 1
  fi
  require_free_disk
  log "creating immutable M1 handoff"
  HANDOFF_OUTPUT="${HANDOFF}" \
  SFT_OUTPUT="${SFT_OUTPUT}" \
  BASE_MODEL_ID="${BASE_MODEL_ID}" \
  PYTHON="${PYTHON}" \
    bash scripts/run_actor_imitation.sh finalize
  require_file "${HANDOFF}/checkpoint.complete.json"
  require_file "${HANDOFF}/m1-handoff.json"
  log "immutable M1 handoff is complete"
}

run_update0_evaluation() {
  local active eval_run
  active="$(pgrep -af '[p]ython -m infoskill.cli (train|eval)' || true)"
  if [[ -n "${active}" ]]; then
    log "another INFO-SKILL train/eval process is active; refusing update-0 evaluation"
    printf '%s\n' "${active}"
    return 1
  fi
  require_free_disk
  wait_for_idle_gpus 20
  log "starting fixed 140-task handoff update-0 evaluation"
  HANDOFF="${HANDOFF}" \
  CONFIG="${CONFIG}" \
  GPUS="${GPUS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  RUN_NAME="${EVAL_RUN_NAME}" \
  PYTHON="${PYTHON}" \
    bash scripts/run_m1_handoff_update0_eval.sh

  eval_run="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
    -name "*-${EVAL_RUN_NAME}" -print | sort | tail -n 1)"
  [[ -n "${eval_run}" ]] || {
    log "could not locate update-0 evaluation run"
    return 1
  }
  "${PYTHON}" - "${eval_run}" "${EXPECTED_WORLD_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
summary = json.loads((run / "valid_seen_summary.json").read_text(encoding="utf-8"))
loaded = json.loads((run / "checkpoint-load.json").read_text(encoding="utf-8"))
reports = loaded.get("worker_reports", [])
assert summary.get("is_complete") is True, summary
assert summary.get("evaluated") == 140, summary
assert loaded.get("status") == "loaded", loaded
assert len(reports) == int(sys.argv[2]), loaded
assert all(report.get("lora_state_loaded") is True for report in reports), loaded
assert all(report.get("optimizer_state_loaded") is False for report in reports), loaded
assert all(report.get("infoskill_state_loaded") is False for report in reports), loaded
print(json.dumps({
    "update0_run": str(run),
    "evaluated": summary["evaluated"],
    "success_count": summary.get("success_count"),
    "macro_success": summary.get("macro_success"),
    "overall_success": summary.get("overall_success"),
    "all_rank_actor_warmstart_loaded": True,
}, indent=2))
PY
  log "update-0 evaluation passed the technical gate: ${eval_run}"
}

run_m1() {
  local active
  active="$(pgrep -af '[p]ython -m infoskill.cli (train|eval)' || true)"
  if [[ -n "${active}" ]]; then
    log "another INFO-SKILL train/eval process is active; refusing M1 launch"
    printf '%s\n' "${active}"
    return 1
  fi
  require_free_disk
  wait_for_idle_gpus 20
  log "starting M1 from immutable handoff through update ${TARGET_UPDATES}"
  HANDOFF="${HANDOFF}" \
  CONFIG="${CONFIG}" \
  GROUNDING_DATA="${GROUNDING_DATA}" \
  TARGET_UPDATES="${TARGET_UPDATES}" \
  GPUS="${GPUS}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  POLICY_MAX_TOKENS_PER_GPU="${POLICY_MAX_TOKENS_PER_GPU:-12288}" \
  ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-16384}" \
  ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE:-1e-6}" \
  CUDA_MEMORY_POLL_INTERVAL_MS="${CUDA_MEMORY_POLL_INTERVAL_MS:-1000}" \
  DRIFT_GUARD_PPO_KL_THRESHOLD="${DRIFT_GUARD_PPO_KL_THRESHOLD:-}" \
  DRIFT_GUARD_INVALID_ACTION_RATE_THRESHOLD="${DRIFT_GUARD_INVALID_ACTION_RATE_THRESHOLD:-}" \
  DRIFT_GUARD_CONSECUTIVE_UPDATES="${DRIFT_GUARD_CONSECUTIVE_UPDATES:-2}" \
  RUN_NAME="${TRAIN_RUN_NAME}" \
  PYTHON="${PYTHON}" \
    bash scripts/run_m1_from_handoff.sh
  log "M1 segment finished"
}

mkdir -p "${PROJECT_ROOT}/logs"
exec 9>"${PROJECT_ROOT}/logs/actor-imitation-to-m1-pipeline.lock"
if ! flock -n 9; then
  log "another actor-imitation-to-M1 pipeline supervisor is active"
  exit 2
fi

trap 'status=$?; if (( status == 0 )); then log "pipeline complete"; else log "pipeline stopped with status=${status}"; fi' EXIT

require_file "${PYTHON}"
require_file "${CONFIG}"
require_file "${GROUNDING_DATA}/manifest.json"
require_file "${PROJECT_ROOT}/artifacts/alfworld-imitation-data/manifest.json"
wait_for_imitation
wait_for_idle_gpus 20
finalize_handoff
run_update0_evaluation
wait_for_idle_gpus 20
run_m1
