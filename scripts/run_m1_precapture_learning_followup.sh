#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
GROUNDING_DATA="${GROUNDING_DATA:?GROUNDING_DATA is required}"
BASELINE_205="${BASELINE_205:?BASELINE_205 is required}"
BASELINE_210="${BASELINE_210:?BASELINE_210 is required}"
BASELINE_215="${BASELINE_215:?BASELINE_215 is required}"
GPUS="${GPUS:-0,1,2}"
MINIMUM_FREE_DISK_GB="${MINIMUM_FREE_DISK_GB:-10}"
TAG="${GATE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_NAME="m1-precapture-learning-s215-u225-${TAG}"
EVAL_NAME="m1-precapture-valid-seen-u225-${TAG}"
REPORT="${PROJECT_ROOT}/m1-precapture-learning-followup-${TAG}.json"
ARCHIVE="${PROJECT_ROOT}/m1-precapture-learning-followup-${TAG}.tar.gz"

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

for path in \
  "${SOURCE_CHECKPOINT}/checkpoint.complete.json" \
  "${SOURCE_CHECKPOINT}/resolved_config.json" \
  "${GROUNDING_DATA}/manifest.json" \
  "${BASELINE_205}/valid_seen_summary.json" \
  "${BASELINE_210}/valid_seen_summary.json" \
  "${BASELINE_215}/valid_seen_summary.json"; do
  [[ -f "${path}" ]] || fail "required input is missing: ${path}"
done
for run in "${BASELINE_205}" "${BASELINE_210}" "${BASELINE_215}"; do
  case "${run}" in
    "${PROJECT_ROOT}"/runs/*) ;;
    *) fail "baseline run must be inside ${PROJECT_ROOT}/runs: ${run}" ;;
  esac
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
  'import sentence_transformers; print("sentence-transformers:", sentence_transformers.__version__)'

"${PYTHON}" - "${SOURCE_CHECKPOINT}" "${GROUNDING_DATA}" \
  "${BASELINE_205}" "${BASELINE_210}" "${BASELINE_215}" <<'PY'
import json
import sys
from pathlib import Path

source, grounding, *baselines = map(Path, sys.argv[1:])
config = json.loads((source / "resolved_config.json").read_text(encoding="utf-8"))
runtime = config["runtime_options"]
plan = config["training_plan"]
manifest = config["evaluation_manifest"]
grounding_manifest = json.loads((grounding / "manifest.json").read_text(encoding="utf-8"))
summaries = [
    json.loads((run / "valid_seen_summary.json").read_text(encoding="utf-8"))
    for run in baselines
]
loads = [
    json.loads((run / "checkpoint-load.json").read_text(encoding="utf-8"))
    for run in baselines
]
checks = {
    "formal_target": plan["profile"] == "formal" and plan["max_updates"] == 445,
    "graph_split_k_one": runtime["hybrid_prefix_cuda_graph"] is True
    and runtime["lora_shrink_split_k_one"] is True,
    "joint_gradient_clip": runtime["policy_gradient_clip_mode"] == "joint",
    "token_budgets": runtime["policy_max_tokens_per_gpu"] == 12288
    and runtime["rollout_max_batched_tokens"] == 16384,
    "bounded_checkpoint_retention": runtime["checkpoint_keep_recent"] == 5
    and runtime["checkpoint_keep_best_valid"] is True,
    "batch_64_validation": manifest["eval_batch_size"] == 64,
    "grounding_passed": grounding_manifest["formal_gate_passed"] is True,
    "baselines_complete_and_same_manifest": all(
        row["is_complete"] is True
        and row["evaluated"] == 140
        and row["task_manifest_sha256"] == manifest["sha256"]
        for row in summaries
    ),
    "baselines_load_expected_checkpoints": all(
        loaded.get("status") == "loaded"
        and loaded.get("loaded") is True
        and loaded.get("checkpoint_step") == step
        for loaded, step in zip(loads, (205, 210, 215))
    ),
    "source_matches_baseline_215": Path(loads[2].get("checkpoint", "")).resolve()
    == source.resolve(),
}
for name, passed in checks.items():
    print(f"preflight/{name}={passed}")
if not all(checks.values()):
    raise SystemExit("learning-followup preflight failed")
PY

training_run=""
evaluation_run=""
archive_diagnostics() {
  local exit_code="$?"
  trap - EXIT
  set +e
  local -a members=()
  local run rel item
  if [[ -z "${training_run}" ]]; then
    training_run="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
      -name "*-${TRAIN_NAME}" | sort | tail -n 1)"
  fi
  if [[ -z "${evaluation_run}" ]]; then
    evaluation_run="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
      -name "*-${EVAL_NAME}" | sort | tail -n 1)"
  fi
  for run in "${training_run}" "${evaluation_run}"; do
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
  for run in "${BASELINE_205}" "${BASELINE_210}" "${BASELINE_215}"; do
    rel="${run#${PROJECT_ROOT}/}"
    for item in valid_seen_summary.json provenance.json checkpoint-load.json; do
      [[ -f "${run}/${item}" ]] && members+=("${rel}/${item}")
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

echo "[INFO-SKILL] fixed-graph learning follow-up: update 215 -> 225"
env \
  GPUS="${GPUS}" \
  PROFILE=formal \
  MAX_UPDATES=445 \
  SEGMENT_END_UPDATE=225 \
  RESUME="${SOURCE_CHECKPOINT}" \
  RUN_NAME="${TRAIN_NAME}" \
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
  POLICY_GRADIENT_CLIP_MODE=joint \
  EVAL_BATCH_SIZE=64 \
  CHECKPOINT_KEEP_RECENT=5 \
  CHECKPOINT_KEEP_BEST_VALID=1 \
  CUDA_MEMORY_POLL_INTERVAL_MS=1000 \
  INFO_SKILL_CPU_THREADS=1 \
  OMP_NUM_THREADS=1 \
  bash scripts/run_alfworld.sh train infoskill

training_run="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
  -name "*-${TRAIN_NAME}" | sort | tail -n 1)"
[[ -d "${training_run}" ]] || fail "training run was not created"
"${PYTHON}" - "${training_run}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
if summary.get("global_update") != 225 or summary.get("status") != "paused":
    raise SystemExit(f"training did not pause at update 225: {summary}")
for step in (220, 225):
    marker = run / "checkpoints" / f"step-{step:06d}" / "checkpoint.complete.json"
    if not marker.is_file():
        raise SystemExit(f"missing committed checkpoint: {marker}")
PY
echo "TRAIN_RUN=${training_run}"

require_disk_headroom
checkpoint="${training_run}/checkpoints/step-000225"
echo "[INFO-SKILL] independent fixed 140-task valid_seen evaluation: update=225"
env \
  GPUS="${GPUS}" \
  EVAL_BACKEND=verl \
  CHECKPOINT_STEP=225 \
  POLICY_CHECKPOINT="${checkpoint}" \
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
evaluation_run="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
  -name "*-${EVAL_NAME}" | sort | tail -n 1)"
[[ -f "${evaluation_run}/valid_seen_summary.json" ]] \
  || fail "update-225 evaluation did not finish"
echo "EVAL_225=${evaluation_run}"

"${PYTHON}" - "${BASELINE_205}" "${BASELINE_210}" "${BASELINE_215}" \
  "${training_run}" "${evaluation_run}" "${REPORT}" <<'PY'
import json
import sys
from pathlib import Path

base_205, base_210, base_215, train, evaluation, output = map(Path, sys.argv[1:])

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

summaries = {
    step: read(run / "valid_seen_summary.json")
    for step, run in ((205, base_205), (210, base_210), (215, base_215),
                      (225, evaluation))
}
loaded = read(evaluation / "checkpoint-load.json")
runtime = read(evaluation / "provenance.json")["evaluation_runtime"]
expected = (train / "checkpoints" / "step-000225").resolve()
checks = {
    "all_complete_same_140_tasks": all(
        row["is_complete"] is True and row["evaluated"] == 140
        and row["task_manifest_sha256"] == summaries[205]["task_manifest_sha256"]
        for row in summaries.values()
    ),
    "checkpoint_225_loaded_on_three_ranks": (
        loaded["status"] == "loaded"
        and loaded["loaded"] is True
        and loaded["checkpoint_step"] == 225
        and Path(loaded["checkpoint"]).resolve() == expected
        and len(loaded["worker_reports"]) == 3
        and all(worker["lora_state_loaded"] and worker["infoskill_state_loaded"]
                for worker in loaded["worker_reports"])
    ),
    "same_graph_split_k_one_batch_64_protocol": (
        runtime["eval_batch_size"] == 64
        and runtime["hybrid_prefix_cuda_graph"] is True
        and runtime["lora_shrink_split_k_one"] is True
        and runtime["num_gpus"] == 3
    ),
}
results = {
    str(step): {
        "success_count": round(row["overall_success"] * row["evaluated"]),
        "overall_success": row["overall_success"],
        "macro_success": row["macro_success"],
        "invalid_action_rate": row["invalid_action_rate"],
        "mean_steps": row["mean_steps"],
    }
    for step, row in summaries.items()
}
report = {
    "schema_version": 1,
    "decision_scope": "bounded_learning_trend_not_formal_improvement_proof",
    "training_run": str(train),
    "evaluation_run": str(evaluation),
    "results": results,
    "update_225_minus_205": {
        "success_count": results["225"]["success_count"] - results["205"]["success_count"],
        "overall_success": results["225"]["overall_success"] - results["205"]["overall_success"],
        "macro_success": results["225"]["macro_success"] - results["205"]["macro_success"],
    },
    "update_225_minus_215": {
        "success_count": results["225"]["success_count"] - results["215"]["success_count"],
        "overall_success": results["225"]["overall_success"] - results["215"]["overall_success"],
        "macro_success": results["225"]["macro_success"] - results["215"]["macro_success"],
    },
    "controls": checks,
    "controls_valid": all(checks.values()),
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
if not report["controls_valid"]:
    raise SystemExit("learning-followup controls failed")
PY
echo "REPORT=${REPORT}"
echo "[INFO-SKILL] update-225 learning follow-up completed; inspect the trend before extending"
