#!/usr/bin/env bash
set -euo pipefail

PYTHON_BASE=${PYTHON_BASE:-/root/autodl-tmp/wjh/my_new_env/infoskill/bin/python}
TOOLS_VENV=${TOOLS_VENV:-/root/autodl-tmp/wjh/webshop-tools}
WEBSHOP_DATA_ROOT=${WEBSHOP_DATA_ROOT:-/root/autodl-tmp/wjh/data/webshop}
ASSET_STAGING=${ASSET_STAGING:-${WEBSHOP_DATA_ROOT}/raw}
MINIMUM_FREE_GB=${MINIMUM_FREE_GB:-30}
OFFICIAL_ARCHIVE_ID=1GWC8UlUzfT9PRTRxgYOwuKSJp4hyV1dp
ARCHIVE=${ASSET_STAGING}/all_trajs.zip
PARTIAL=${ARCHIVE}.partial
LISTING=${ASSET_STAGING}/all_trajs.listing.txt

[[ -x "${PYTHON_BASE}" ]] || {
  echo "base Python is unavailable: ${PYTHON_BASE}" >&2
  exit 2
}
command -v unzip >/dev/null 2>&1 || {
  echo "unzip is required to validate the official archive" >&2
  exit 2
}

free_bytes=$(df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' ')
minimum_bytes=$((MINIMUM_FREE_GB * 1024 * 1024 * 1024))
if (( free_bytes < minimum_bytes )); then
  echo "free disk is below ${MINIMUM_FREE_GB} GiB; refusing to download" >&2
  df -h /root/autodl-tmp >&2
  exit 2
fi

mkdir -p "${ASSET_STAGING}"
if [[ ! -f "${ARCHIVE}" ]]; then
  if [[ ! -x "${TOOLS_VENV}/bin/gdown" ]]; then
    "${PYTHON_BASE}" -m venv --system-site-packages "${TOOLS_VENV}"
    "${TOOLS_VENV}/bin/python" -m pip install \
      --disable-pip-version-check \
      "gdown==5.2.0"
  fi
  "${TOOLS_VENV}/bin/gdown" \
    "https://drive.google.com/uc?id=${OFFICIAL_ARCHIVE_ID}" \
    -O "${PARTIAL}"
  unzip -tq "${PARTIAL}"
  mv "${PARTIAL}" "${ARCHIVE}"
else
  unzip -tq "${ARCHIVE}"
fi

unzip -Z1 "${ARCHIVE}" >"${LISTING}"
archive_sha256=$(sha256sum "${ARCHIVE}" | awk '{print $1}')
archive_bytes=$(stat -c '%s' "${ARCHIVE}")
archive_entries=$(wc -l <"${LISTING}" | tr -d ' ')

echo "OFFICIAL_SOURCE=https://drive.google.com/file/d/${OFFICIAL_ARCHIVE_ID}/view"
echo "HUMAN_ARCHIVE=${ARCHIVE}"
echo "HUMAN_ARCHIVE_SHA256=${archive_sha256}"
echo "HUMAN_ARCHIVE_BYTES=${archive_bytes}"
echo "HUMAN_ARCHIVE_ENTRIES=${archive_entries}"
echo "HUMAN_ARCHIVE_LISTING=${LISTING}"
echo "===== first entries ====="
sed -n '1,20p' "${LISTING}"
echo "===== last entries ====="
tail -n 20 "${LISTING}"
echo "===== disk ====="
df -h /root/autodl-tmp
