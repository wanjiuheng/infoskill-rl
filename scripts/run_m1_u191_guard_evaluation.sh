#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}"
SOURCE_RUN="${SOURCE_RUN:-${PROJECT_ROOT}/runs/20260920T044319Z-m1-handoff-lr3e6-formal-u200-20260920_124241}"
RECOVERY_RUN="${RECOVERY_RUN:-${PROJECT_ROOT}/runs/20260921T035441Z-m1-handoff-recovery-s175-lr1e6-u445-20260921_115359}"
RECOVERY_RESOLVED_CONFIG="${RECOVERY_RUN}/resolved_config.json"
CHECKPOINT="${RECOVERY_RUN}/checkpoints/step-000191"
BASELINE_SUMMARY="${SOURCE_RUN}/valid_seen-000175-summary.json"
STAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_NAME="m1-handoff-recovery-valid-seen-u191-${STAMP}"

for required in \
  "${PYTHON_BIN}" \
  "${RECOVERY_RESOLVED_CONFIG}" \
  "${CHECKPOINT}/checkpoint.complete.json" \
  "${BASELINE_SUMMARY}"
do
  [[ -e "${required}" ]] || {
    echo "Required evaluation input is missing: ${required}" >&2
    exit 2
  }
done

# Resolve the model-visible skill artifacts from the run being evaluated.
# Ambient experiment variables are deliberately ignored.
mapfile -t SOURCE_PATHS < <(
  "${PYTHON_BIN}" - "${RECOVERY_RESOLVED_CONFIG}" <<'PY'
import json
import sys

path = sys.argv[1]
config = json.load(open(path, encoding="utf-8"))
paths = config["app_config"]["paths"]
for key in ("skill_bank", "skill_bank_manifest"):
    value = paths.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"recovery resolved config has no non-empty {key}: {path}")
    print(value)
PY
)
[[ "${#SOURCE_PATHS[@]}" -eq 2 ]] || {
  echo "Could not resolve the two source skill paths" >&2
  exit 2
}
EVAL_SKILL_BANK="${SOURCE_PATHS[0]}"
EVAL_SKILL_BANK_MANIFEST="${SOURCE_PATHS[1]}"

for required in "${EVAL_SKILL_BANK}" "${EVAL_SKILL_BANK_MANIFEST}"; do
  [[ -f "${required}" ]] || {
    echo "Required source skill artifact is missing: ${required}" >&2
    exit 2
  }
done

active="$(pgrep -af '[p]ython -m infoskill.cli (train|eval)' || true)"
if [[ -n "${active}" ]]; then
  echo "Another training or evaluation process is active; refusing to start:" >&2
  echo "${active}" >&2
  exit 2
fi

free_bytes="$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')"
if (( free_bytes < 10 * 1024 * 1024 * 1024 )); then
  echo "Free disk is below 10 GiB; refusing to start" >&2
  df -h /root/autodl-tmp >&2
  exit 2
fi

"${PYTHON_BIN}" -c \
  'import sentence_transformers; print("sentence-transformers:", sentence_transformers.__version__)'

echo "SOURCE_RUN=${SOURCE_RUN}"
echo "RECOVERY_RUN=${RECOVERY_RUN}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "BASELINE_SUMMARY=${BASELINE_SUMMARY}"
echo "SKILL_BANK=${EVAL_SKILL_BANK}"
echo "SKILL_BANK_MANIFEST=${EVAL_SKILL_BANK_MANIFEST}"
echo "EVAL_NAME=${EVAL_NAME}"

env \
  PYTHON="${PYTHON_BIN}" \
  PATH="$(dirname "${PYTHON_BIN}"):${PATH}" \
  CONFIG=configs/alfworld_qwen25_7b.yaml \
  GPUS="${GPUS:-0,1,2}" \
  EVAL_BACKEND=verl \
  CHECKPOINT_STEP=191 \
  POLICY_CHECKPOINT="${CHECKPOINT}" \
  WARMSTART_HANDOFF="" \
  SKILL_BANK="${EVAL_SKILL_BANK}" \
  SKILL_BANK_MANIFEST="${EVAL_SKILL_BANK_MANIFEST}" \
  EVAL_TASK_MANIFEST="" \
  EVAL_BATCH_SIZE=64 \
  RETRIEVAL_MODE="" \
  RAW_SKILL_PROMPT_FORMAT=full \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  GROUPED_INFOSKILL_CONDITIONING=0 \
  HYBRID_PREFIX_CUDA_GRAPH=1 \
  LORA_SHRINK_SPLIT_K_ONE=1 \
  CUDA_MEMORY_POLL_INTERVAL_MS=0 \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME="${EVAL_NAME}" \
  bash scripts/run_alfworld.sh eval infoskill

EVAL_RUN="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
  -name "*-${EVAL_NAME}" | sort | tail -n 1)"
[[ -n "${EVAL_RUN}" && -f "${EVAL_RUN}/valid_seen_summary.json" ]] || {
  echo "Completed evaluation summary was not found for ${EVAL_NAME}" >&2
  exit 2
}

REPORT="${EVAL_RUN}/drift-guard-evaluation.json"
"${PYTHON_BIN}" - \
  "${BASELINE_SUMMARY}" \
  "${EVAL_RUN}/valid_seen_summary.json" \
  "${REPORT}" <<'PY'
import json
import sys
from pathlib import Path

baseline_path, candidate_path, output_path = map(Path, sys.argv[1:])
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

def complete(summary):
    return bool(summary.get("complete", summary.get("is_complete", False)))

def metric(summary, name):
    return float(summary[name])

def execution_mode(summary):
    explicit = summary.get("execution_mode")
    if explicit:
        return explicit
    performance = summary.get("rollout_performance", {})
    graph = bool(performance.get("perf/hybrid_prefix_cuda_graph", 0.0))
    split_k_one = bool(
        performance.get("perf/lora_shrink_split_k_one_verified", 0.0)
    )
    if graph and split_k_one:
        return "cuda_graph_split_k_one"
    if graph:
        return "cuda_graph"
    if split_k_one:
        return "eager_split_k_one"
    return "eager"

baseline_execution_mode = execution_mode(baseline)
candidate_execution_mode = execution_mode(candidate)
controls = {
    "baseline_complete": complete(baseline),
    "candidate_complete": complete(candidate),
    "candidate_evaluated_140": int(candidate.get("evaluated", 0)) == 140,
    "same_task_manifest": (
        baseline.get("task_manifest_sha256")
        == candidate.get("task_manifest_sha256")
    ),
    "same_execution_mode": baseline_execution_mode == candidate_execution_mode,
}
controls_valid = all(controls.values())
deltas = {
    key: metric(candidate, key) - metric(baseline, key)
    for key in ("macro_success", "overall_success", "invalid_action_rate")
}
if not controls_valid:
    classification = "invalid_controls"
elif (
    deltas["macro_success"] >= -0.02
    and deltas["overall_success"] >= -0.02
    and deltas["invalid_action_rate"] <= 0.05
):
    classification = "step_191_stable_near_step_175"
else:
    classification = "step_191_degraded_from_step_175"

report = {
    "schema_version": 1,
    "classification": classification,
    "controls": controls,
    "controls_valid": controls_valid,
    "decision_thresholds": {
        "minimum_macro_delta": -0.02,
        "minimum_overall_delta": -0.02,
        "maximum_invalid_action_rate_delta": 0.05,
    },
    "baseline": {
        "summary": str(baseline_path),
        "update": 175,
        "macro_success": metric(baseline, "macro_success"),
        "overall_success": metric(baseline, "overall_success"),
        "invalid_action_rate": metric(baseline, "invalid_action_rate"),
        "mean_steps": metric(baseline, "mean_steps"),
        "execution_mode": baseline_execution_mode,
    },
    "candidate": {
        "summary": str(candidate_path),
        "update": 191,
        "macro_success": metric(candidate, "macro_success"),
        "overall_success": metric(candidate, "overall_success"),
        "invalid_action_rate": metric(candidate, "invalid_action_rate"),
        "mean_steps": metric(candidate, "mean_steps"),
        "execution_mode": candidate_execution_mode,
    },
    "candidate_minus_baseline": deltas,
}
output_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

ARCHIVE="${PROJECT_ROOT}/m1-u191-drift-guard-evaluation-${STAMP}.tar.gz"
BASELINE_REL="${BASELINE_SUMMARY#${PROJECT_ROOT}/}"
EVAL_REL="${EVAL_RUN#${PROJECT_ROOT}/}"
FILES=(
  "${BASELINE_REL}"
  "${EVAL_REL}/valid_seen_summary.json"
  "${EVAL_REL}/drift-guard-evaluation.json"
)
for optional in \
  resolved_config.json \
  provenance.json \
  checkpoint-load.json \
  evaluation-timing.json \
  console.log
do
  [[ -f "${EVAL_RUN}/${optional}" ]] && FILES+=("${EVAL_REL}/${optional}")
done
tar -czf "${ARCHIVE}" -C "${PROJECT_ROOT}" "${FILES[@]}"

echo "EVAL_RUN=${EVAL_RUN}"
echo "REPORT=${REPORT}"
echo "ARCHIVE=${ARCHIVE}"
ls -lh "${ARCHIVE}"
