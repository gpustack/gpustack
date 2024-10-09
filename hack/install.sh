#!/usr/bin/env bash

# Set error handling
set -o errexit
set -o nounset
set -o pipefail

# Get the root directory and third_party directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Include the common functions
source "${ROOT_DIR}/hack/lib/init.sh"

function download_deps() {
  pip install poetry==1.7.1 pre-commit==3.7.1
  poetry install
  pre-commit install
}

function download_ui() {
  local default_tag="latest"
  local ui_path="${ROOT_DIR}/gpustack/ui"
  local tmp_ui_path="${ui_path}/tmp"
  local tag="latest"
  # local tag="${1}"

  rm -rf "${ui_path}"
  mkdir -p "${tmp_ui_path}/ui"

  gpustack::log::info "downloading ui assets"

  if ! curl --retry 3 --retry-all-errors --retry-delay 3 -sSfL "https://gpustack-ui-1303613262.cos.accelerate.myqcloud.com/releases/${tag}.tar.gz" 2>/dev/null |
    tar -xzf - --directory "${tmp_ui_path}/ui" 2>/dev/null; then

    if [[ "${tag:-}" =~ ^v([0-9]+)\.([0-9]+)(\.[0-9]+)?(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$ ]]; then
      gpustack::log::fatal "failed to download '${tag}' ui archive"
    fi

    gpustack::log::warn "failed to download '${tag}' ui archive, fallback to '${default_tag}' ui archive"
    if ! curl --retry 3 --retry-all-errors --retry-delay 3 -sSfL "https://gpustack-ui-1303613262.cos.accelerate.myqcloud.com/releases/${default_tag}.tar.gz" |
      tar -xzf - --directory "${PACKAGE_TMP_DIR}/ui" 2>/dev/null; then
      gpustack::log::fatal "failed to download '${default_tag}' ui archive"
    fi
  fi
  cp -a "${tmp_ui_path}/ui/dist/." "${ui_path}"

  rm -rf "${tmp_ui_path}"
}


#
# main
#

gpustack::log::info "+++ DEPENDENCIES +++"
download_deps
download_ui
gpustack::log::info "--- DEPENDENCIES ---"
