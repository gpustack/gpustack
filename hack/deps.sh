#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function download_deps() {
  pip install poetry==1.7.1
  poetry install
}

FASTFETCH_DIR="${ROOT_DIR}/gpustack/third_party/fastfetch"
FASTFETCH_TMP_DIR="${FASTFETCH_DIR}/tmp"
FASTFETCH_DEFAUT_VERSION="2.14.0"


function download_fastfetch() {
    local default_os="linux"
    local default_arch="amd64"

    local os="${1:-$default_os}"
    local arch="${2:-$default_arch}"
    local version="${3:-$FASTFETCH_DEFAUT_VERSION}"


    gpustack::log::info "downloading fastfetch-${os}-${arch} '${version}'  archive"
    
    local tmp_file="${FASTFETCH_TMP_DIR}/fastfetch-${os}-${arch}.zip"
    rm -rf "${FASTFETCH_TMP_DIR}"
    mkdir -p "${FASTFETCH_TMP_DIR}"

    curl --retry 3 --retry-all-errors --retry-delay 3 \
      -o  "${tmp_file}" \
      -sSfL "https://github.com/aiwantaozi/fastfetch/releases/download/${version}/fastfetch-${os}-${arch}.zip"
    
    unzip -qu "${tmp_file}" -d "${FASTFETCH_TMP_DIR}"
    
    cp "${FASTFETCH_TMP_DIR}/fastfetch-${os}-${arch}/usr/bin/fastfetch" "${FASTFETCH_DIR}/fastfetch-${os}-${arch}"
    rm -rf "${tmp_file}"
}

function download_fastfetches() {
    download_fastfetch linux amd64
    download_fastfetch linux aarch64
    download_fastfetch macos universal
    rm -rf "${FASTFETCH_TMP_DIR}"
}

#
# main
#

gpustack::log::info "+++ DEPENDENCIES +++"
download_deps
download_fastfetches
gpustack::log::info "--- DEPENDENCIES ---"