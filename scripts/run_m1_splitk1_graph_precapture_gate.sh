#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:?POLICY_CHECKPOINT is required}"
EAGER_RUN="${EAGER_RUN:?EAGER_RUN from the accepted fresh-runtime matrix is required}"
REFERENCE_RUN="${REFERENCE_RUN:?REFERENCE_RUN from the accepted fresh-runtime matrix is required}"
GPUS="${GPUS:-0,1,2}"
M1_REPRO_CASE_COUNT="${M1_REPRO_CASE_COUNT:-3}"
M1_REPRO_MAX_NEW_TOKENS="${M1_REPRO_MAX_NEW_TOKENS:-64}"
MINIMUM_FREE_DISK_GB="${MINIMUM_FREE_DISK_GB:-10}"
TAG="${GATE_TAG:-$(date +%Y%m%dT%H%M%SZ)}"

for required in \
  "${POLICY_CHECKPOINT}/checkpoint.complete.json" \
  "${EAGER_RUN}/m1_lora_reproducibility.json" \
  "${REFERENCE_RUN}/m1_lora_reproducibility.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "required input is missing: ${required}" >&2
    exit 2
  fi
done
for reused_run in "${EAGER_RUN}" "${REFERENCE_RUN}"; do
  case "${reused_run}" in
    "${PROJECT_ROOT}"/runs/*) ;;
    *)
      echo "reused control run must be inside ${PROJECT_ROOT}/runs: ${reused_run}" >&2
      exit 2
      ;;
  esac
done
if [[ ! -x "${PYTHON}" ]]; then
  echo "PYTHON is not executable: ${PYTHON}" >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if [[ "${#GPU_IDS[@]}" -ne 3 ]]; then
  echo "graph pre-capture gate requires exactly three GPU indices" >&2
  exit 2
fi
for gpu_id in "${GPU_IDS[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "invalid GPU index in GPUS=${GPUS}: ${gpu_id}" >&2
    exit 2
  fi
  GPU_PROCESSES="$(nvidia-smi -i "${gpu_id}" \
    --query-compute-apps=pid,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "${GPU_PROCESSES}" ]]; then
    echo "GPU ${gpu_id} already has compute processes; refusing to start" >&2
    echo "${GPU_PROCESSES}" >&2
    exit 2
  fi
done
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

echo "[INFO-SKILL] graph pre-capture gate: CUDA Graph + SPLIT_K=1"
PYTHON="${PYTHON}" \
GPUS="${GPUS}" \
POLICY_CHECKPOINT="${POLICY_CHECKPOINT}" \
M1_REPRO_CASE_COUNT="${M1_REPRO_CASE_COUNT}" \
M1_REPRO_MAX_NEW_TOKENS="${M1_REPRO_MAX_NEW_TOKENS}" \
M1_REPRO_LORA_KERNEL_INTERVENTION=none \
HYBRID_PREFIX_CUDA_GRAPH=1 \
LORA_SHRINK_SPLIT_K_ONE=1 \
INFO_SKILL_CPU_THREADS=1 \
RUN_NAME="m1-graph-precapture-splitk1-${TAG}" \
bash scripts/run_alfworld.sh m1-lora-reproducibility

GRAPH_RUN="$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -type d \
  -name "*-m1-graph-precapture-splitk1-${TAG}" | sort | tail -n 1)"
if [[ ! -f "${GRAPH_RUN}/m1_lora_reproducibility.json" ]]; then
  echo "graph pre-capture report is missing: ${GRAPH_RUN}" >&2
  exit 2
fi

REPORT="${PROJECT_ROOT}/m1-splitk1-graph-precapture-gate-${TAG}.json"
"${PYTHON}" scripts/compare_m1_fresh_runtime_matrix.py \
  "${GRAPH_RUN}/m1_lora_reproducibility.json" \
  "${EAGER_RUN}/m1_lora_reproducibility.json" \
  "${REFERENCE_RUN}/m1_lora_reproducibility.json" \
  --output "${REPORT}"

ARCHIVE="${PROJECT_ROOT}/m1-splitk1-graph-precapture-gate-${TAG}.tar.gz"
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
echo "GATE_REPORT=${REPORT}"
echo "ARCHIVE=${ARCHIVE}"
ls -lh "${ARCHIVE}"
