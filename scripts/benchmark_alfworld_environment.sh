#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/alfworld_qwen25_7b.yaml}"
PROFILE="${PROFILE:-smoke}"                 # smoke: 4 slots; full: formal 64 slots
MINIMUM_SPEEDUP="${MINIMUM_SPEEDUP:-1.25}"
INFO_SKILL_CPU_THREADS="${INFO_SKILL_CPU_THREADS:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! "${INFO_SKILL_CPU_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "INFO_SKILL_CPU_THREADS must be a positive integer" >&2
  exit 2
fi
if [[ "${PROFILE}" != "smoke" && "${PROFILE}" != "full" ]]; then
  echo "PROFILE must be smoke or full" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${INFO_SKILL_CPU_THREADS}"
export MKL_NUM_THREADS="${INFO_SKILL_CPU_THREADS}"

echo "[INFO-SKILL] benchmark=alfworld-environment profile=${PROFILE} config=${CONFIG}"
python scripts/benchmark_alfworld_environment.py \
  --config "${CONFIG}" \
  --profile "${PROFILE}" \
  --minimum-speedup "${MINIMUM_SPEEDUP}" \
  "$@"
