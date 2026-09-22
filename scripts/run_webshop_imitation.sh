#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ACTION=${1:-}
PYTHON_BIN=${PYTHON_BIN:-python}
WEBSHOP_SOURCE=${WEBSHOP_SOURCE:-${WEBSHOP_ROOT:-${PROJECT_ROOT}/../SkillRL/agent_system/environments/env_package/webshop/webshop}}
WEBSHOP_DATA_ROOT=${WEBSHOP_DATA_ROOT:-${PROJECT_ROOT}/../../data/webshop}
HUMAN_ARCHIVE=${HUMAN_ARCHIVE:-${WEBSHOP_DATA_ROOT}/raw/all_trajs.zip}
ARCHIVE_AUDIT=${ARCHIVE_AUDIT:-${WEBSHOP_DATA_ROOT}/raw/all_trajs.audit.json}
DEMONSTRATIONS=${DEMONSTRATIONS:-${WEBSHOP_DATA_ROOT}/baseline_models/data/il_trajs_finalized_images.jsonl}
HUMAN_GOALS=${HUMAN_GOALS:-${WEBSHOP_DATA_ROOT}/baseline_models/data/human_goals.json}
PREPARED_DATA=${PREPARED_DATA:-${WEBSHOP_DATA_ROOT}/processed/imitation-data-content-grouped-v2}
DATA_AUDIT=${DATA_AUDIT:-${WEBSHOP_DATA_ROOT}/processed/imitation-data-content-grouped-v2-audit.json}
SKILL_BANK=${SKILL_BANK:-${WEBSHOP_DATA_ROOT}/processed/webshop-skill-bank-content-grouped-v2.json}
SFT_OUTPUT=${SFT_OUTPUT:-${PROJECT_ROOT}/runs/webshop-actor-imitation-warmstart}
MODEL_PATH=${MODEL_PATH:-${PROJECT_ROOT}/../../models/Qwen/Qwen2.5-7B-Instruct}
BASE_MODEL_ID=${BASE_MODEL_ID:-qwen2.5-7b-instruct}
EXPECTED_TRAJECTORIES=${EXPECTED_TRAJECTORIES:-1010}
EXPECTED_DEMONSTRATIONS_SHA256=${EXPECTED_DEMONSTRATIONS_SHA256:-0f3ef1890245a283f8116b7abcabebd4acdf355d773edd99977e8ed6de63ec6c}
EXPECTED_HUMAN_GOALS_SHA256=${EXPECTED_HUMAN_GOALS_SHA256:-b68746ed66cd31fdc5f70eb3f5831a46b38163563b57a66ddcab8fd60ee0cbdc}
IMITATION_MAX_LENGTH=${IMITATION_MAX_LENGTH:-16384}
IMITATION_BATCH_SIZE=${IMITATION_BATCH_SIZE:-1}
IMITATION_GRADIENT_ACCUMULATION=${IMITATION_GRADIENT_ACCUMULATION:-16}

case "${ACTION}" in
  audit-archive)
    [[ -f "${HUMAN_ARCHIVE}" ]] || {
      echo "missing WebShop human archive: ${HUMAN_ARCHIVE}" >&2
      exit 2
    }
    exec env PYTHONPATH="${PROJECT_ROOT}/src" "${PYTHON_BIN}" \
      "${PROJECT_ROOT}/scripts/audit_webshop_human_archive.py" \
      --archive "${HUMAN_ARCHIVE}" \
      --output "${ARCHIVE_AUDIT}"
    ;;
  doctor)
    exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/webshop_asset_doctor.py" \
      --webshop-root "${WEBSHOP_SOURCE}" \
      --webshop-data-root "${WEBSHOP_DATA_ROOT}" \
      --output "${PROJECT_ROOT}/webshop-asset-doctor.json"
    ;;
  build-index)
    exec env PYTHONPATH="${WEBSHOP_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" \
      "${PROJECT_ROOT}/scripts/build_webshop_search_index.py" \
      --webshop-root "${WEBSHOP_SOURCE}" \
      --webshop-data-root "${WEBSHOP_DATA_ROOT}" \
      --threads "${WEBSHOP_INDEX_THREADS:-4}"
    ;;
  smoke-online)
    exec env PYTHONPATH="${PROJECT_ROOT}/src:${WEBSHOP_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" \
      -m infoskill.integrations.webshop.online_smoke \
      --webshop-root "${WEBSHOP_SOURCE}" \
      --webshop-data-root "${WEBSHOP_DATA_ROOT}" \
      --output "${WEBSHOP_ONLINE_SMOKE_REPORT:-${PROJECT_ROOT}/webshop-online-smoke.json}"
    ;;
  prepare)
    [[ -f "${DEMONSTRATIONS}" ]] || {
      echo "missing WebShop demonstrations: ${DEMONSTRATIONS}" >&2
      exit 2
    }
    [[ -f "${HUMAN_GOALS}" ]] || {
      echo "missing WebShop human goals: ${HUMAN_GOALS}" >&2
      exit 2
    }
    exec env PYTHONPATH="${PROJECT_ROOT}/src" "${PYTHON_BIN}" \
      -m infoskill.imitation.cli prepare-webshop \
      --demonstrations "${DEMONSTRATIONS}" \
      --human-goals "${HUMAN_GOALS}" \
      --output "${PREPARED_DATA}" \
      --expected-trajectories "${EXPECTED_TRAJECTORIES}" \
      --expected-demonstrations-sha256 "${EXPECTED_DEMONSTRATIONS_SHA256}" \
      --expected-human-goals-sha256 "${EXPECTED_HUMAN_GOALS_SHA256}"
    ;;
  audit)
    [[ -f "${PREPARED_DATA}/manifest.json" ]] || {
      echo "prepared WebShop imitation data is missing" >&2
      exit 2
    }
    exec env PYTHONPATH="${PROJECT_ROOT}/src" "${PYTHON_BIN}" \
      -m infoskill.imitation.cli audit-webshop \
      --data "${PREPARED_DATA}" \
      --model "${MODEL_PATH}" \
      --output "${DATA_AUDIT}" \
      --max-length "${IMITATION_MAX_LENGTH}"
    ;;
  build-skill-bank)
    [[ -f "${PREPARED_DATA}/manifest.json" ]] || {
      echo "prepared WebShop imitation data is missing" >&2
      exit 2
    }
    [[ -f "${DATA_AUDIT}" ]] || {
      echo "passed WebShop imitation audit is missing: ${DATA_AUDIT}" >&2
      exit 2
    }
    exec env PYTHONPATH="${PROJECT_ROOT}/src" "${PYTHON_BIN}" \
      -m infoskill.imitation.cli build-webshop-skill-bank \
      --data "${PREPARED_DATA}" \
      --audit "${DATA_AUDIT}" \
      --output "${SKILL_BANK}"
    ;;
  train)
    [[ -f "${PREPARED_DATA}/manifest.json" ]] || {
      echo "prepared WebShop imitation data is missing" >&2
      exit 2
    }
    [[ -f "${DATA_AUDIT}" ]] || {
      echo "passed WebShop imitation audit is missing: ${DATA_AUDIT}" >&2
      exit 2
    }
    env PYTHONPATH="${PROJECT_ROOT}/src" "${PYTHON_BIN}" \
      -m infoskill.imitation.cli verify-webshop-audit \
      --data "${PREPARED_DATA}" \
      --audit "${DATA_AUDIT}" \
      --model "${MODEL_PATH}" \
      --max-length "${IMITATION_MAX_LENGTH}"
    exec env PYTHONPATH="${PROJECT_ROOT}/src" "${PYTHON_BIN}" \
      -m torch.distributed.run \
      --standalone \
      --nproc_per_node "${NUM_GPUS:-3}" \
      -m infoskill.imitation.cli train \
      --model "${MODEL_PATH}" \
      --base-model-id "${BASE_MODEL_ID}" \
      --data "${PREPARED_DATA}" \
      --audit "${DATA_AUDIT}" \
      --output "${SFT_OUTPUT}" \
      --learning-rate "${IMITATION_LEARNING_RATE:-1e-4}" \
      --epochs "${IMITATION_EPOCHS:-2.0}" \
      --per-device-batch-size "${IMITATION_BATCH_SIZE}" \
      --gradient-accumulation-steps "${IMITATION_GRADIENT_ACCUMULATION}" \
      --max-length "${IMITATION_MAX_LENGTH}"
    ;;
  *)
    echo "usage: bash scripts/run_webshop_imitation.sh audit-archive|doctor|build-index|smoke-online|prepare|audit|build-skill-bank|train" >&2
    exit 2
    ;;
esac
