#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
THIRD_PARTY_DIR="${ROOT_DIR}/gpustack/third_party"

source "${ROOT_DIR}/hack/lib/init.sh"

function download_deps() {
  pip install poetry==1.7.1
  poetry install
}

function download_fastfetch() {
    local default_os="linux"
    local default_arch="amd64"
    local default_version="2.14.0"

    local os="${1:-$default_os}"
    local arch="${2:-$default_arch}"
    local version="${3:-$default_version}"

    local fastfetch_dir="${THIRD_PARTY_DIR}/fastfetch"
    local fastfetch_tmp_dir="${fastfetch_dir}/tmp"


    local target_file="${fastfetch_dir}/fastfetch-${os}-${arch}"
    if [ -f "${target_file}" ]; then
        gpustack::log::info "fastfetch-${os}-${arch} already exists, skipping download"
        return
    fi

    gpustack::log::info "downloading fastfetch-${os}-${arch} '${version}'  archive"
    
    local tmp_file="${fastfetch_tmp_dir}/fastfetch-${os}-${arch}.zip"
    rm -rf "${fastfetch_tmp_dir}"
    mkdir -p "${fastfetch_tmp_dir}"

    curl --retry 3 --retry-all-errors --retry-delay 3 \
      -o  "${tmp_file}" \
      -sSfL "https://github.com/aiwantaozi/fastfetch/releases/download/${version}/fastfetch-${os}-${arch}.zip"
    
    unzip -qu "${tmp_file}" -d "${fastfetch_tmp_dir}"
    
    cp "${fastfetch_tmp_dir}/fastfetch-${os}-${arch}/usr/bin/fastfetch" "${target_file}"
    rm -rf "${fastfetch_tmp_dir}"
}


function download_llama_cpp_server() {
    local version="b3135"
    local llama_cpp_dir="${THIRD_PARTY_DIR}/llama_cpp"
    local llama_cpp_tmp_dir="${llama_cpp_dir}/tmp"

    platforms=("macos-arm64" "macos-x64" "ubuntu-x64")

    for platform in "${platforms[@]}"; do
      local target_file="${llama_cpp_dir}/server-${platform}"
      if [ -f "${target_file}" ]; then
          gpustack::log::info "llama.cpp server-${platform} already exists, skipping download"
          continue
      fi

      local tmp_file="${llama_cpp_tmp_dir}/llama-${version}-bin-${platform}.zip"
      rm -rf "${llama_cpp_tmp_dir}"
      mkdir -p "${llama_cpp_tmp_dir}"

      gpustack::log::info "downloading llama-${version}-bin-${platform} ${version} archive"

      curl --retry 3 --retry-all-errors --retry-delay 3 \
        -o  "${tmp_file}" \
        -sSfL "https://github.com/ggerganov/llama.cpp/releases/download/${version}/llama-${version}-bin-${platform}.zip"

      unzip -qu "${tmp_file}" -d "${llama_cpp_tmp_dir}"

      cp "${llama_cpp_tmp_dir}/build/bin/server" "${llama_cpp_dir}/server-${platform}"
      rm -rf "${llama_cpp_tmp_dir}"
    done
}

function download_fastfetches() {
    download_fastfetch linux amd64
    download_fastfetch linux aarch64
    download_fastfetch macos universal
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
download_fastfetches
download_llama_cpp_server
download_ui
gpustack::log::info "--- DEPENDENCIES ---"