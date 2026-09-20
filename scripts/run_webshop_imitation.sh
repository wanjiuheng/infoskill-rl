#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ACTION=${1:-}
PYTHON_BIN=${PYTHON_BIN:-python}
WEBSHOP_SOURCE=${WEBSHOP_SOURCE:-${WEBSHOP_ROOT:-${PROJECT_ROOT}/../SkillRL/agent_system/environments/env_package/webshop/webshop}}
WEBSHOP_DATA_ROOT=${WEBSHOP_DATA_ROOT:-/root/autodl-tmp/wjh/data/webshop}
HUMAN_ARCHIVE=${HUMAN_ARCHIVE:-${WEBSHOP_DATA_ROOT}/raw/all_trajs.zip}
ARCHIVE_AUDIT=${ARCHIVE_AUDIT:-${WEBSHOP_DATA_ROOT}/raw/all_trajs.audit.json}
DEMONSTRATIONS=${DEMONSTRATIONS:-${WEBSHOP_DATA_ROOT}/baseline_models/data/il_trajs_finalized_images.jsonl}
HUMAN_GOALS=${HUMAN_GOALS:-${WEBSHOP_DATA_ROOT}/baseline_models/data/human_goals.json}
PREPARED_DATA=${PREPARED_DATA:-${WEBSHOP_DATA_ROOT}/processed/imitation-data}
DATA_AUDIT=${DATA_AUDIT:-${WEBSHOP_DATA_ROOT}/processed/imitation-data-audit.json}
SKILL_BANK=${SKILL_BANK:-${WEBSHOP_DATA_ROOT}/processed/webshop-skill-bank.json}
SFT_OUTPUT=${SFT_OUTPUT:-${PROJECT_ROOT}/runs/webshop-actor-imitation-warmstart}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/wjh/models/Qwen/Qwen2.5-7B-Instruct}
BASE_MODEL_ID=${BASE_MODEL_ID:-qwen2.5-7b-instruct}
EXPECTED_TRAJECTORIES=${EXPECTED_TRAJECTORIES:-1010}
EXPECTED_DEMONSTRATIONS_SHA256=${EXPECTED_DEMONSTRATIONS_SHA256:-0f3ef1890245a283f8116b7abcabebd4acdf355d773edd99977e8ed6de63ec6c}
EXPECTED_HUMAN_GOALS_SHA256=${EXPECTED_HUMAN_GOALS_SHA256:-b68746ed66cd31fdc5f70eb3f5831a46b38163563b57a66ddcab8fd60ee0cbdc}

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
      --max-length "${IMITATION_MAX_LENGTH:-4352}"
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
      --max-length "${IMITATION_MAX_LENGTH:-4352}"
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
      --per-device-batch-size "${IMITATION_BATCH_SIZE:-2}" \
      --gradient-accumulation-steps "${IMITATION_GRADIENT_ACCUMULATION:-8}" \
      --max-length "${IMITATION_MAX_LENGTH:-4352}"
    ;;
  *)
    echo "usage: bash scripts/run_webshop_imitation.sh audit-archive|doctor|prepare|audit|build-skill-bank|train" >&2
    exit 2
    ;;
esac
