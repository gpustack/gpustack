#!/usr/bin/env bash

# Set error handling
set -o errexit
set -o nounset
set -o pipefail

# Get the root directory and third_party directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
THIRD_PARTY_DIR="${ROOT_DIR}/gpustack/third_party/bin"

# Include the common functions
source "${ROOT_DIR}/hack/lib/init.sh"

function download_deps() {
  pip install poetry==1.7.1 pre-commit==3.7.1
  poetry install
  pre-commit install
}

function download_fastfetch() {
    local version="2.18.1.4"

    local fastfetch_dir="${THIRD_PARTY_DIR}/fastfetch"
    local fastfetch_tmp_dir="${fastfetch_dir}/tmp"

    if gpustack::util::is_darwin; then
      platforms=("macos-universal")
    elif gpustack::util::is_linux; then
      platforms=("linux-amd64" "linux-aarch64")
    fi

    for platform in "${platforms[@]}"; do
      local target_file="${fastfetch_dir}/fastfetch-${platform}"
      if [ -f "${target_file}" ]; then
          gpustack::log::info "fastfetch-${platform} already exists, skipping download"
          continue
      fi

      gpustack::log::info "downloading fastfetch-${platform} '${version}'  archive"
    
      local tmp_file="${fastfetch_tmp_dir}/fastfetch-${platform}.zip"
      rm -rf "${fastfetch_tmp_dir}"
      mkdir -p "${fastfetch_tmp_dir}"

      curl --retry 3 --retry-all-errors --retry-delay 3 \
        -o  "${tmp_file}" \
        -sSfL "https://github.com/gpustack/fastfetch/releases/download/${version}/fastfetch-${platform}.zip"
    
      unzip -qu "${tmp_file}" -d "${fastfetch_tmp_dir}"
    
      cp "${fastfetch_tmp_dir}/fastfetch-${platform}/usr/bin/fastfetch" "${target_file}"
    done

    rm -rf "${fastfetch_tmp_dir}"
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

function download_gguf_parser() {
    local version="v0.5.1"

    local gguf_parser_dir="${THIRD_PARTY_DIR}/gguf-parser"
    mkdir -p "${gguf_parser_dir}"

    if gpustack::util::is_darwin; then
      platforms=("darwin-universal")
    elif gpustack::util::is_linux; then
      platforms=("linux-amd64" "linux-arm64")
    fi

    for platform in "${platforms[@]}"; do
      local target_file="${gguf_parser_dir}/gguf-parser-${platform}"
      if [ -f "${target_file}" ]; then
          gpustack::log::info "gguf-parser-${platform} already exists, skipping download"
          continue
      fi

      gpustack::log::info "downloading gguf-parser-${platform} '${version}'  archive"
    
      curl --retry 3 --retry-all-errors --retry-delay 3 \
        -o  "${target_file}" \
        -sSfL "https://github.com/gpustack/gguf-parser-go/releases/download/${version}/gguf-parser-${platform}"

      chmod +x "${target_file}"
    done
}


function download_llama_box() {
    local version="v0.0.21"

    local llama_box_dir="${THIRD_PARTY_DIR}/llama-box"
    local llama_box_tmp_dir="${llama_box_dir}/tmp"
    
    if gpustack::util::is_darwin; then
      platforms=("darwin-amd64-metal" "darwin-arm64-metal")
    elif gpustack::util::is_linux; then
      platforms=("linux-amd64-cuda-12.5-s")
    fi

    for platform in "${platforms[@]}"; do
      local target_file="${llama_box_dir}/llama-box-${platform%-s}" # cut off the suffix
      if [ -f "${target_file}" ]; then
          gpustack::log::info "llama-box-${platform} already exists, skipping download"
          continue
      fi

      gpustack::log::info "downloading llama-box-${platform} '${version}'  archive"

      local llama_box_platform_tmp_dir="${llama_box_tmp_dir}/${platform}"
      rm -rf "${llama_box_platform_tmp_dir}"
      mkdir -p "${llama_box_platform_tmp_dir}"

      local tmp_file="${llama_box_tmp_dir}/llama-box-${version}-${platform}.zip"
      curl --retry 3 --retry-all-errors --retry-delay 3 \
        -o  "${tmp_file}" \
        -sSfL "https://github.com/gpustack/llama-box/releases/download/${version}/llama-box-${platform}.zip"

      unzip -qu "${tmp_file}" -d "${llama_box_platform_tmp_dir}"
      cp "${llama_box_platform_tmp_dir}/llama-box" "${target_file}"
      chmod +x "${target_file}"
    done

    rm -rf "${llama_box_tmp_dir}"
}

#
# main
#

gpustack::log::info "+++ DEPENDENCIES +++"
download_deps
download_llama_box
download_gguf_parser
download_fastfetch
download_ui
gpustack::log::info "--- DEPENDENCIES ---"
