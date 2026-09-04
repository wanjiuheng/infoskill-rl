#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_ROOT="${PROJECT_ROOT}/third_party/patches/vllm-0.8.4"
SOURCE="${1:-}"
WHEEL_DIR="${2:-${PROJECT_ROOT}/dist/vllm}"
INSTALL_MODE="${3:-}"

if [[ -z "${SOURCE}" ]]; then
  echo "Usage: bash scripts/build_patched_vllm.sh /absolute/path/to/vllm-0.8.4 [wheel-dir] [--install]" >&2
  exit 2
fi

SOURCE="$(cd "${SOURCE}" && pwd)"
mkdir -p "${WHEEL_DIR}"
WHEEL_DIR="$(cd "${WHEEL_DIR}" && pwd)"

python "${PROJECT_ROOT}/scripts/verify_vllm_patch.py" "${SOURCE}"

BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/infoskill-vllm-build.XXXXXX")"
cleanup() {
  rm -rf -- "${BUILD_ROOT}"
}
trap cleanup EXIT

mkdir -p "${BUILD_ROOT}/source"
git -C "${SOURCE}" archive --format=tar HEAD | tar -xf - -C "${BUILD_ROOT}/source"
git -C "${BUILD_ROOT}/source" apply --check "${PATCH_ROOT}/0001-infoskill-hybrid-prefix.patch"
git -C "${BUILD_ROOT}/source" apply "${PATCH_ROOT}/0001-infoskill-hybrid-prefix.patch"

WHEEL=""
if [[ -n "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]]; then
  BASE_WHEEL_NAME="$(basename -- "${VLLM_PRECOMPILED_WHEEL_LOCATION}")"
  if [[ "${BASE_WHEEL_NAME}" != vllm-0.8.4-*.whl ]]; then
    echo "Expected an official vllm-0.8.4 wheel, got ${BASE_WHEEL_NAME}" >&2
    exit 1
  fi
  echo "Repackaging ${VLLM_PRECOMPILED_WHEEL_LOCATION} with patched Python files"
  python "${PROJECT_ROOT}/scripts/prepare_vllm_wheel.py" repackage \
    "${VLLM_PRECOMPILED_WHEEL_LOCATION}" \
    "${BUILD_ROOT}/source" \
    "${PATCH_ROOT}/manifest.json" \
    "${WHEEL_DIR}"
  WHEEL="${WHEEL_DIR}/vllm-0.8.4+infoskill1-${BASE_WHEEL_NAME#vllm-0.8.4-}"
else
  echo "VLLM_PRECOMPILED_WHEEL_LOCATION is unset; vLLM CUDA extensions will compile from source."
  export SETUPTOOLS_SCM_PRETEND_VERSION="0.8.4+infoskill1"
  python -m pip wheel "${BUILD_ROOT}/source" --no-deps --no-build-isolation --wheel-dir "${WHEEL_DIR}"
  WHEEL="$(find "${WHEEL_DIR}" -maxdepth 1 -type f -name 'vllm-0.8.4+infoskill1*.whl' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
fi

if [[ -z "${WHEEL}" || ! -f "${WHEEL}" ]]; then
  echo "Patched wheel was not found in ${WHEEL_DIR}" >&2
  exit 1
fi
python "${PROJECT_ROOT}/scripts/prepare_vllm_wheel.py" verify "${WHEEL}"
sha256sum "${WHEEL}" | tee "${WHEEL}.sha256"

if [[ "${INSTALL_MODE}" == "--install" ]]; then
  python -m pip install --force-reinstall --no-deps "${WHEEL}"
  python -c "import vllm; from vllm.inputs.data import INFOSKILL_HYBRID_PREFIX_API; print(vllm.__version__, INFOSKILL_HYBRID_PREFIX_API)"
else
  echo "Wheel built but not installed. Re-run with third argument --install, or install the wheel explicitly."
fi
