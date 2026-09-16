#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:?POLICY_CHECKPOINT is required}"
GPUS="${GPUS:-0,1,2}"
M1_REPRO_CASE_COUNT="${M1_REPRO_CASE_COUNT:-3}"
M1_REPRO_MAX_NEW_TOKENS="${M1_REPRO_MAX_NEW_TOKENS:-64}"
MINIMUM_FREE_DISK_GB="${MINIMUM_FREE_DISK_GB:-10}"
TAG="${MATRIX_TAG:-$(date +%Y%m%dT%H%M%SZ)}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "PYTHON is not executable: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${POLICY_CHECKPOINT}/checkpoint.complete.json" ]]; then
  echo "portable checkpoint is incomplete: ${POLICY_CHECKPOINT}" >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if [[ "${#GPU_IDS[@]}" -ne 3 ]]; then
  echo "fresh-runtime matrix requires exactly three GPU indices" >&2
  exit 2
fi
ACTIVE="$(pgrep -af '[p]ython -m infoskill.cli (train|eval|m1-)' || true)"
if [[ -n "${ACTIVE}" ]]; then
  echo "another INFO-SKILL GPU process is active; refusing to start" >&2
  echo "${ACTIVE}" >&2
  exit 2
fi
FREE_BYTES="$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')"
if (( FREE_BYTES < MINIMUM_FREE_DISK_GB * 1024 * 1024 * 1024 )); then
  echo "free disk is below ${MINIMUM_FREE_DISK_GB} GiB; refusing to start" >&2
  df -h /root/autodl-tmp >&2
  exit 2
fi

"${PYTHON}" -c \
  'import sentence_transformers; print("sentence-transformers:", sentence_transformers.__version__)'

run_cell() {
  local name="$1"
  local graph="$2"
  local split_k_one="$3"
  local intervention="$4"
  echo "[INFO-SKILL] fresh-runtime matrix cell=${name} graph=${graph} split_k_one=${split_k_one} intervention=${intervention}"
  PYTHON="${PYTHON}" \
  GPUS="${GPUS}" \
  POLICY_CHECKPOINT="${POLICY_CHECKPOINT}" \
  M1_REPRO_CASE_COUNT="${M1_REPRO_CASE_COUNT}" \
  M1_REPRO_MAX_NEW_TOKENS="${M1_REPRO_MAX_NEW_TOKENS}" \
  M1_REPRO_LORA_KERNEL_INTERVENTION="${intervention}" \
  HYBRID_PREFIX_CUDA_GRAPH="${graph}" \
  LORA_SHRINK_SPLIT_K_ONE="${split_k_one}" \
  INFO_SKILL_CPU_THREADS=1 \
  RUN_NAME="m1-fresh-runtime-${name}-${TAG}" \
  bash scripts/run_alfworld.sh m1-lora-reproducibility
}

run_cell graph-splitk1 1 1 none
run_cell eager-splitk1 0 1 none
run_cell eager-reference-full 0 0 reference_full

GRAPH_RUN="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
  -name "*-m1-fresh-runtime-graph-splitk1-${TAG}" | sort | tail -n 1)"
EAGER_RUN="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
  -name "*-m1-fresh-runtime-eager-splitk1-${TAG}" | sort | tail -n 1)"
REFERENCE_RUN="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
  -name "*-m1-fresh-runtime-eager-reference-full-${TAG}" | sort | tail -n 1)"
for run in "${GRAPH_RUN}" "${EAGER_RUN}" "${REFERENCE_RUN}"; do
  if [[ ! -f "${run}/m1_lora_reproducibility.json" ]]; then
    echo "matrix cell report is missing: ${run}" >&2
    exit 2
  fi
done

REPORT="${PROJECT_ROOT}/m1-splitk1-fresh-runtime-matrix-${TAG}.json"
"${PYTHON}" scripts/compare_m1_fresh_runtime_matrix.py \
  "${GRAPH_RUN}/m1_lora_reproducibility.json" \
  "${EAGER_RUN}/m1_lora_reproducibility.json" \
  "${REFERENCE_RUN}/m1_lora_reproducibility.json" \
  --output "${REPORT}"

ARCHIVE="${PROJECT_ROOT}/m1-splitk1-fresh-runtime-matrix-${TAG}.tar.gz"
GRAPH_REL="${GRAPH_RUN#${PROJECT_ROOT}/}"
EAGER_REL="${EAGER_RUN#${PROJECT_ROOT}/}"
REFERENCE_REL="${REFERENCE_RUN#${PROJECT_ROOT}/}"
REPORT_REL="${REPORT#${PROJECT_ROOT}/}"
tar -czf "${ARCHIVE}" -C "${PROJECT_ROOT}" \
  "${REPORT_REL}" \
  "${GRAPH_REL}/m1_lora_reproducibility.json" \
  "${GRAPH_REL}/resolved_config.json" \
  "${GRAPH_REL}/console.log" \
  "${EAGER_REL}/m1_lora_reproducibility.json" \
  "${EAGER_REL}/resolved_config.json" \
  "${EAGER_REL}/console.log" \
  "${REFERENCE_REL}/m1_lora_reproducibility.json" \
  "${REFERENCE_REL}/resolved_config.json" \
  "${REFERENCE_REL}/console.log"

echo "GRAPH_RUN=${GRAPH_RUN}"
echo "EAGER_RUN=${EAGER_RUN}"
echo "REFERENCE_RUN=${REFERENCE_RUN}"
echo "MATRIX_REPORT=${REPORT}"
echo "ARCHIVE=${ARCHIVE}"
ls -lh "${ARCHIVE}"
