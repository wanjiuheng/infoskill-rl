#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ACTION=${1:-}
PYTHON_BIN=${PYTHON_BIN:-python}
WEBSHOP_SOURCE=${WEBSHOP_SOURCE:-${WEBSHOP_ROOT:-${PROJECT_ROOT}/../SkillRL/agent_system/environments/env_package/webshop/webshop}}
WEBSHOP_DATA_ROOT=${WEBSHOP_DATA_ROOT:-/root/autodl-tmp/wjh/data/webshop}
DEMONSTRATIONS=${DEMONSTRATIONS:-${WEBSHOP_DATA_ROOT}/baseline_models/data/il_trajs_finalized_images.jsonl}
HUMAN_GOALS=${HUMAN_GOALS:-${WEBSHOP_DATA_ROOT}/baseline_models/data/human_goals.json}
PREPARED_DATA=${PREPARED_DATA:-${WEBSHOP_DATA_ROOT}/processed/imitation-data}
SFT_OUTPUT=${SFT_OUTPUT:-${PROJECT_ROOT}/runs/webshop-actor-imitation-warmstart}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/wjh/models/Qwen/Qwen2.5-7B-Instruct}
BASE_MODEL_ID=${BASE_MODEL_ID:-qwen2.5-7b-instruct}

case "${ACTION}" in
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
      --expected-trajectories "${EXPECTED_TRAJECTORIES:-1012}"
    ;;
  train)
    [[ -f "${PREPARED_DATA}/manifest.json" ]] || {
      echo "prepared WebShop imitation data is missing" >&2
      exit 2
    }
    exec env PYTHONPATH="${PROJECT_ROOT}/src" "${PYTHON_BIN}" \
      -m torch.distributed.run \
      --standalone \
      --nproc_per_node "${NUM_GPUS:-3}" \
      -m infoskill.imitation.cli train \
      --model "${MODEL_PATH}" \
      --base-model-id "${BASE_MODEL_ID}" \
      --data "${PREPARED_DATA}" \
      --output "${SFT_OUTPUT}" \
      --learning-rate "${IMITATION_LEARNING_RATE:-1e-4}" \
      --epochs "${IMITATION_EPOCHS:-2.0}" \
      --per-device-batch-size "${IMITATION_BATCH_SIZE:-2}" \
      --gradient-accumulation-steps "${IMITATION_GRADIENT_ACCUMULATION:-8}" \
      --max-length "${IMITATION_MAX_LENGTH:-4352}"
    ;;
  *)
    echo "usage: bash scripts/run_webshop_imitation.sh doctor|prepare|train" >&2
    exit 2
    ;;
esac
