#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
GROUNDING_DATA="${GROUNDING_DATA:?GROUNDING_DATA is required}"
BASELINE_A="${BASELINE_A:?BASELINE_A is required}"
BASELINE_B="${BASELINE_B:?BASELINE_B is required}"
GPUS="${GPUS:-0,1,2}"
MINIMUM_FREE_DISK_GB="${MINIMUM_FREE_DISK_GB:-10}"
TAG="${GATE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_NAME="m1-precapture-learning-s205-u215-${TAG}"
REPORT="${PROJECT_ROOT}/m1-precapture-learning-gate-${TAG}.json"
ARCHIVE="${PROJECT_ROOT}/m1-precapture-learning-gate-${TAG}.tar.gz"

fail() {
  echo "$1" >&2
  exit 2
}

[[ -x "${PYTHON}" ]] || fail "PYTHON is not executable: ${PYTHON}"
export PATH="$(dirname -- "${PYTHON}"):${PATH}"
[[ "$(command -v python)" == "${PYTHON}" ]] || fail "failed to lock python"
for path in \
  "${SOURCE_CHECKPOINT}/checkpoint.complete.json" \
  "${SOURCE_CHECKPOINT}/resolved_config.json" \
  "${GROUNDING_DATA}/manifest.json" \
  "${BASELINE_A}/valid_seen_summary.json" \
  "${BASELINE_B}/valid_seen_summary.json"; do
  [[ -f "${path}" ]] || fail "required input is missing: ${path}"
done
for run in "${BASELINE_A}" "${BASELINE_B}"; do
  case "${run}" in
    "${PROJECT_ROOT}"/runs/*) ;;
    *) fail "baseline run must be inside ${PROJECT_ROOT}/runs: ${run}" ;;
  esac
done
[[ "$(basename -- "${SOURCE_CHECKPOINT}")" == "step-000205" ]] \
  || fail "source must be the committed step-205 checkpoint"
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
  "${BASELINE_A}" "${BASELINE_B}" <<'PY'
import json
import sys
from pathlib import Path

source, grounding, baseline_a, baseline_b = map(Path, sys.argv[1:])
config = json.loads((source / "resolved_config.json").read_text(encoding="utf-8"))
runtime = config["runtime_options"]
plan = config["training_plan"]
manifest = config["evaluation_manifest"]
grounding_manifest = json.loads((grounding / "manifest.json").read_text(encoding="utf-8"))
summaries = [
    json.loads((run / "valid_seen_summary.json").read_text(encoding="utf-8"))
    for run in (baseline_a, baseline_b)
]
loads = [
    json.loads((run / "checkpoint-load.json").read_text(encoding="utf-8"))
    for run in (baseline_a, baseline_b)
]
baseline_fields = (
    "evaluated", "is_complete", "task_manifest_sha256",
    "overall_success", "macro_success", "invalid_action_rate", "mean_steps",
    "per_task_type_success",
)
checks = {
    "source_has_registered_formal_target": plan["profile"] == "formal"
    and plan["max_updates"] == 445,
    "source_uses_graph_and_split_k_one": runtime["hybrid_prefix_cuda_graph"] is True
    and runtime["lora_shrink_split_k_one"] is True,
    "source_uses_joint_gradient_clip": runtime["policy_gradient_clip_mode"] == "joint",
    "source_uses_bounded_checkpoint_retention": runtime["checkpoint_keep_recent"] == 5
    and runtime["checkpoint_keep_best_valid"] is True,
    "source_uses_batch_64_validation": manifest["eval_batch_size"] == 64,
    "grounding_passed": grounding_manifest["formal_gate_passed"] is True,
    "both_baselines_complete": all(
        row["is_complete"] is True and row["evaluated"] == 140
        for row in summaries
    ),
    "baselines_match_source_manifest": all(
        row["task_manifest_sha256"] == manifest["sha256"]
        for row in summaries
    ),
    "baseline_outcomes_match": all(
        summaries[0][key] == summaries[1][key] for key in baseline_fields
    ),
    "baseline_checkpoint_matches_source": all(
        loaded.get("loaded") is True
        and loaded.get("status") == "loaded"
        and Path(loaded.get("checkpoint", "")).resolve() == source.resolve()
        for loaded in loads
    ),
}
for name, passed in checks.items():
    print(f"preflight/{name}={passed}")
if not all(checks.values()):
    raise SystemExit("learning-gate preflight failed")
PY

training_run=""
eval_210=""
eval_215=""

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
  if [[ -z "${eval_210}" ]]; then
    eval_210="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
      -name "*-m1-precapture-valid-seen-u210-${TAG}" | sort | tail -n 1)"
  fi
  if [[ -z "${eval_215}" ]]; then
    eval_215="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
      -name "*-m1-precapture-valid-seen-u215-${TAG}" | sort | tail -n 1)"
  fi
  for run in "${training_run}" "${eval_210}" "${eval_215}"; do
    [[ -d "${run}" ]] || continue
    rel="${run#${PROJECT_ROOT}/}"
    for item in \
      training_summary.json resolved_config.json provenance.json \
      valid_seen_summary.json checkpoint-load.json evaluation-timing.json \
      metrics.jsonl console.log traces \
      checkpoints/step-000210/checkpoint.complete.json \
      checkpoints/step-000215/checkpoint.complete.json; do
      [[ -e "${run}/${item}" ]] && members+=("${rel}/${item}")
    done
  done
  for run in "${BASELINE_A}" "${BASELINE_B}"; do
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

echo "[INFO-SKILL] fixed-graph learning gate: update 205 -> 215"
env \
  GPUS="${GPUS}" \
  PROFILE=formal \
  MAX_UPDATES=445 \
  SEGMENT_END_UPDATE=215 \
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
if summary.get("global_update") != 215 or summary.get("status") != "paused":
    raise SystemExit(f"training did not pause at update 215: {summary}")
for step in (210, 215):
    marker = run / "checkpoints" / f"step-{step:06d}" / "checkpoint.complete.json"
    if not marker.is_file():
        raise SystemExit(f"missing committed checkpoint: {marker}")
PY
echo "TRAIN_RUN=${training_run}"

for step in 210 215; do
  require_disk_headroom
  checkpoint="${training_run}/checkpoints/step-$(printf '%06d' "${step}")"
  run_name="m1-precapture-valid-seen-u${step}-${TAG}"
  echo "[INFO-SKILL] fixed 140-task valid_seen evaluation: update=${step}"
  env \
    GPUS="${GPUS}" \
    EVAL_BACKEND=verl \
    CHECKPOINT_STEP="${step}" \
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
    RUN_NAME="${run_name}" \
    bash scripts/run_alfworld.sh eval infoskill
  run="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
    -name "*-${run_name}" | sort | tail -n 1)"
  [[ -f "${run}/valid_seen_summary.json" ]] \
    || fail "update-${step} evaluation did not finish"
  if [[ "${step}" -eq 210 ]]; then eval_210="${run}"; else eval_215="${run}"; fi
  echo "EVAL_${step}=${run}"
done

"${PYTHON}" - "${SOURCE_CHECKPOINT}" "${BASELINE_A}" "${BASELINE_B}" \
  "${training_run}" "${eval_210}" "${eval_215}" "${REPORT}" <<'PY'
import json
import sys
from pathlib import Path

source, baseline_a, baseline_b, train, eval_210, eval_215, output = map(
    Path, sys.argv[1:]
)

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

base = read(baseline_a / "valid_seen_summary.json")
evaluations = {}
checks = {}
for step, run in ((210, eval_210), (215, eval_215)):
    summary = read(run / "valid_seen_summary.json")
    loaded = read(run / "checkpoint-load.json")
    runtime = read(run / "provenance.json")["evaluation_runtime"]
    expected_checkpoint = str((train / "checkpoints" / f"step-{step:06d}").resolve())
    checks[f"update_{step}_complete"] = (
        summary["is_complete"] is True
        and summary["evaluated"] == 140
        and summary["task_manifest_sha256"] == base["task_manifest_sha256"]
    )
    checks[f"update_{step}_checkpoint_loaded"] = (
        loaded["status"] == "loaded"
        and loaded["loaded"] is True
        and str(Path(loaded["checkpoint"]).resolve()) == expected_checkpoint
        and len(loaded["worker_reports"]) == 3
        and all(
            worker["lora_state_loaded"] and worker["infoskill_state_loaded"]
            for worker in loaded["worker_reports"]
        )
    )
    checks[f"update_{step}_execution_protocol"] = (
        runtime["eval_batch_size"] == 64
        and runtime["hybrid_prefix_cuda_graph"] is True
        and runtime["lora_shrink_split_k_one"] is True
        and runtime["num_gpus"] == 3
    )
    evaluations[str(step)] = {
        "run": str(run),
        "success_count": round(summary["overall_success"] * summary["evaluated"]),
        "overall_success": summary["overall_success"],
        "macro_success": summary["macro_success"],
        "invalid_action_rate": summary["invalid_action_rate"],
        "mean_steps": summary["mean_steps"],
        "rollout_seconds": summary["timing_seconds"]["rollout_seconds"],
        "macro_minus_update_205": summary["macro_success"] - base["macro_success"],
        "overall_minus_update_205": summary["overall_success"] - base["overall_success"],
    }

report = {
    "schema_version": 1,
    "decision_scope": "short_learning_signal_only_not_formal_improvement_proof",
    "source_checkpoint": str(source),
    "training_run": str(train),
    "baseline_runs": [str(baseline_a), str(baseline_b)],
    "baseline_update_205": {
        "success_count": round(base["overall_success"] * base["evaluated"]),
        "overall_success": base["overall_success"],
        "macro_success": base["macro_success"],
    },
    "evaluations": evaluations,
    "controls": checks,
    "controls_valid": all(checks.values()),
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
if not report["controls_valid"]:
    raise SystemExit("learning-gate controls failed")
PY
echo "REPORT=${REPORT}"
echo "[INFO-SKILL] short learning gate completed; efficacy requires inspection"
