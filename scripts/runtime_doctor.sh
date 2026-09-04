#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SKILLRL_SOURCE="${SKILLRL_SOURCE:-${PROJECT_ROOT}/../SkillRL}"
if [[ -d "${SKILLRL_SOURCE}/verl" ]]; then
  export PYTHONPATH="${PROJECT_ROOT}/src:${SKILLRL_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
else
  echo "WARNING: VERL source not found at ${SKILLRL_SOURCE}; set SKILLRL_SOURCE explicitly" >&2
  export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
fi
OUTPUT="${1:-runtime-doctor.json}"
python -m infoskill.runtime_doctor | tee "${OUTPUT}"
python -m json.tool "${OUTPUT}" >/dev/null
echo "Runtime report written to ${OUTPUT}"
