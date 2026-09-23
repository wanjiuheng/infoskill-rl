#!/usr/bin/env bash
set -euo pipefail

# Continue the stable Qwen3-1.7B actor-only recovery from update 100 through
# update 150. The completed update-100 run remains an immutable input.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}"
SOURCE_RUN="${PROJECT_ROOT}/runs/20260923T112809Z-m1-qwen3-1p7b-recovery-s75-actoronly-lr1e6-u100-20260923_192737"
SOURCE_CONFIG="${SOURCE_RUN}/resolved_config.json"
SOURCE_SUMMARY="${SOURCE_RUN}/training_summary.json"
EXPECTED_SOURCE_UPDATE=100
RESUME_CHECKPOINT="${SOURCE_RUN}/checkpoints/step-000100"
CHECKPOINT_COMPLETE="${RESUME_CHECKPOINT}/checkpoint.complete.json"
TRAINER_STATE="${RESUME_CHECKPOINT}/trainer_state.json"
SEGMENT_END_UPDATE="${SEGMENT_END_UPDATE:-150}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-m1-qwen3-1p7b-recovery-s100-actoronly-lr1e6-u150-${STAMP}}"

[[ -x "${PYTHON_BIN}" ]] || {
  echo "Python interpreter is unavailable: ${PYTHON_BIN}" >&2
  exit 2
}
for required in \
  "${SOURCE_CONFIG}" \
  "${SOURCE_SUMMARY}" \
  "${CHECKPOINT_COMPLETE}" \
  "${TRAINER_STATE}"
do
  [[ -f "${required}" ]] || {
    echo "Required recovery source is missing: ${required}" >&2
    exit 2
  }
done

# The directory name alone is not authoritative. Pin every state component that
# affects an exact continuation so a copied or similarly configured update-100
# checkpoint cannot silently enter this registered recovery fork.
printf '%s  %s\n' \
  '7d25eccdb6b9a3e1958a0c4b92096784dd3e429d3fa8251c98a2b96e3a748678' "${CHECKPOINT_COMPLETE}" \
  '312ce9ef1903593045ed5c59d87574b5bb025b485eefc090e846ee6c913631e6' "${RESUME_CHECKPOINT}/provenance.json" \
  '48352c351681758fe4264dfc57b0f5cb8283784539664992ee6084e80cdf294a' "${RESUME_CHECKPOINT}/resolved_config.json" \
  'd7056c507d2da03cb74df4b592facd17f0e202f7fe8937d02af149be17b6fb9a' "${RESUME_CHECKPOINT}/runtime/actor/actor_manifest.json" \
  '4dc89bef3ab98b98d77c1247e971eaf04438759d0c7ba5477ea1d90df5fb1ae2' "${RESUME_CHECKPOINT}/runtime/actor/adapter_config.json" \
  'ba7e1a45164cde480f71b6d9b6c7db646195df1c0a02058ab405a92448b349f5' "${RESUME_CHECKPOINT}/runtime/actor/adapter_model.safetensors" \
  '6df89cdca00a4c7b84e388f19395258b56e832b186ffb382d2ab98f463325746' "${RESUME_CHECKPOINT}/runtime/actor/infoskill/infoskill_manifest.json" \
  'e476069c3df3b0a90fc7530115ba79791919a81e5f767f3fe593aca8c1a0d830' "${RESUME_CHECKPOINT}/runtime/actor/infoskill/infoskill_modules.pt" \
  '62948e83e70e2d335e41bb2e5951a7a901eebb881b5ffe291cd545b84db42fc2' "${RESUME_CHECKPOINT}/runtime/actor/infoskill/infoskill_optimizers.pt" \
  'c9dc5192cd646b0fc00eabe1b5233edb3a7750ef9a8b23169b184574f463c6be' "${RESUME_CHECKPOINT}/runtime/actor/infoskill/infoskill_rng_state.pt" \
  '8677f00990bfe25f207352d6acc4dde63e010ee4979e843cf08352284ba5bfa1' "${RESUME_CHECKPOINT}/runtime/actor/infoskill/infoskill_schedulers.pt" \
  'aba69a2f98aa169e81e839c4c06573763476d0e1de28cd92f4c68db900af6b02' "${RESUME_CHECKPOINT}/runtime/actor/lora_optimizer_full.pt" \
  'faf7c7a392a8271968c15ad781d516da93510f40324b49a530acbc986ffd3a4b' "${RESUME_CHECKPOINT}/runtime/actor/lora_scheduler.pt" \
  '53422f4d7ad1b802e73006bfd5a81a9537b4cb61d01faa1904244c36465af5c3' "${TRAINER_STATE}" \
  | sha256sum -c -

"${PYTHON_BIN}" - "${SOURCE_SUMMARY}" "${SOURCE_CONFIG}" \
  "${CHECKPOINT_COMPLETE}" "${EXPECTED_SOURCE_UPDATE}" <<'PY'
import json
import sys

summary_path, config_path, checkpoint_path, expected_update_text = sys.argv[1:]
expected_update = int(expected_update_text)
source_summary = json.load(open(summary_path, encoding="utf-8"))
source_config = json.load(open(config_path, encoding="utf-8"))
checkpoint = json.load(open(checkpoint_path, encoding="utf-8"))

if source_summary.get("status") != "paused":
    raise SystemExit("source recovery run is not paused")
if source_summary.get("pause_reason") != "segment_end_update":
    raise SystemExit("source recovery run did not stop at its segment boundary")
if source_summary.get("global_update") != expected_update:
    raise SystemExit(
        "source recovery update differs from the required update: "
        f"expected {expected_update}, got {source_summary.get('global_update')}"
    )
expected_cursor = expected_update * source_config["training_plan"][
    "task_groups_per_update"
]
if source_summary.get("task_cursor") != expected_cursor:
    raise SystemExit(
        "source recovery task cursor is not resume-equivalent: "
        f"expected {expected_cursor}, got {source_summary.get('task_cursor')}"
    )
if checkpoint.get("global_update") != expected_update:
    raise SystemExit("checkpoint marker update differs from the source summary")
if checkpoint.get("portable") is not True:
    raise SystemExit("source checkpoint is not authoritative portable state")

runtime = source_config["runtime_options"]
expected_controls = {
    "actor_learning_rate": 1e-6,
    "invalid_action_penalty": 0.1,
    "freeze_infoskill_conditioning": True,
    "checkpoint_keep_recent": 5,
    "checkpoint_keep_best_valid": True,
}
for key, expected in expected_controls.items():
    actual = runtime.get(key)
    if actual != expected:
        raise SystemExit(
            f"source recovery control differs for {key}: "
            f"expected {expected!r}, got {actual!r}"
        )
PY

if [[ "${SEGMENT_END_UPDATE}" != "150" ]]; then
  echo "This registered continuation must end at update 150" >&2
  exit 2
fi

mapfile -t SOURCE_PATHS < <(
  "${PYTHON_BIN}" - "${SOURCE_CONFIG}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
paths = config["app_config"]["paths"]
for key in ("grounding_data", "skill_bank", "skill_bank_manifest"):
    value = paths.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"source resolved config has no non-empty {key}")
    print(value)
PY
)
[[ "${#SOURCE_PATHS[@]}" -eq 3 ]] || {
  echo "Could not resolve the source conditioning paths" >&2
  exit 2
}
RECOVERY_GROUNDING_DATA="${SOURCE_PATHS[0]}"
RECOVERY_SKILL_BANK="${SOURCE_PATHS[1]}"
RECOVERY_SKILL_BANK_MANIFEST="${SOURCE_PATHS[2]}"

for required in \
  "${RECOVERY_GROUNDING_DATA}/manifest.json" \
  "${RECOVERY_SKILL_BANK}" \
  "${RECOVERY_SKILL_BANK_MANIFEST}"
do
  [[ -f "${required}" ]] || {
    echo "Required source artifact is missing: ${required}" >&2
    exit 2
  }
done

active="$(pgrep -af '[p]ython -m infoskill.cli (train|eval)' || true)"
if [[ -n "${active}" ]]; then
  echo "Another InfoSkill train/eval process is active; refusing to start:" >&2
  echo "${active}" >&2
  exit 2
fi

free_bytes="$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')"
if (( free_bytes < 12 * 1024 * 1024 * 1024 )); then
  echo "Free disk is below 12 GiB; refusing to start" >&2
  df -h /root/autodl-tmp >&2
  exit 2
fi

"${PYTHON_BIN}" -c \
  'import sentence_transformers; print("sentence-transformers:", sentence_transformers.__version__)'

echo "SOURCE_RUN=${SOURCE_RUN}"
echo "RESUME=${RESUME_CHECKPOINT}"
echo "RUN_NAME=${RUN_NAME}"
echo "SEGMENT_END_UPDATE=${SEGMENT_END_UPDATE}"
echo "Expected fixed valid_seen evaluations: update 125 and update 150"
echo "Source recovery run and checkpoint step-000100 remain unchanged."

exec env \
  PYTHON="${PYTHON_BIN}" \
  PATH="$(dirname "${PYTHON_BIN}"):${PATH}" \
  GPUS="${GPUS:-0,1,2}" \
  PROFILE=formal \
  MAX_UPDATES=445 \
  SEGMENT_END_UPDATE="${SEGMENT_END_UPDATE}" \
  RESUME="${RESUME_CHECKPOINT}" \
  GROUNDING_DATA="${RECOVERY_GROUNDING_DATA}" \
  SKILL_BANK="${RECOVERY_SKILL_BANK}" \
  SKILL_BANK_MANIFEST="${RECOVERY_SKILL_BANK_MANIFEST}" \
  EVAL_BATCH_SIZE=64 \
  PERSISTENT_ROLLOUT_SESSION=1 \
  ENVIRONMENT_BACKEND=native_batch \
  ENVIRONMENT_WORKERS=1 \
  POLICY_MAX_TOKENS_PER_GPU=12288 \
  ROLLOUT_MAX_BATCHED_TOKENS=16384 \
  CUDA_MEMORY_POLL_INTERVAL_MS=1000 \
  INFO_SKILL_CPU_THREADS=1 \
  ACTOR_LEARNING_RATE=1e-6 \
  INVALID_ACTION_PENALTY=0.1 \
  FREEZE_INFOSKILL_CONDITIONING=1 \
  POLICY_GRADIENT_CLIP_MODE=joint \
  HYBRID_PREFIX_CUDA_GRAPH=1 \
  LORA_SHRINK_SPLIT_K_ONE=1 \
  FUSE_KL_PPO_FORWARD=0 \
  SKIP_UNUSED_OLD_LOGPROB_ENTROPY=0 \
  BALANCE_POLICY_TOKENS_ACROSS_RANKS=1 \
  CHECKPOINT_KEEP_RECENT=5 \
  CHECKPOINT_KEEP_BEST_VALID=1 \
  DRIFT_GUARD_PPO_KL_THRESHOLD=0.02 \
  DRIFT_GUARD_INVALID_ACTION_RATE_THRESHOLD=0.05 \
  DRIFT_GUARD_CONSECUTIVE_UPDATES=2 \
  RUN_NAME="${RUN_NAME}" \
  bash scripts/run_alfworld.sh \
    train infoskill configs/alfworld_qwen3_1p7b.yaml "${GPUS:-0,1,2}"
