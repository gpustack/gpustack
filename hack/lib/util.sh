#!/usr/bin/env bash


function gpustack::util::sed() {
  if ! sed -i "$@" >/dev/null 2>&1; then
    # back off none GNU sed
    sed -i "" "$@"
  fi
}

function gpustack::util::get_os_name() {
  # Support overriding by BUILD_OS for cross-building
  local os_name="${BUILD_OS:-}"
  if [[ -n "$os_name" ]]; then
    echo "$os_name" | tr '[:upper:]' '[:lower:]'
  else
    uname -s | tr '[:upper:]' '[:lower:]'
  fi
}

function gpustack::util::is_darwin() {
  [[ "$(gpustack::util::get_os_name)" == "darwin" ]]
}

function gpustack::util::is_linux() {
  [[ "$(gpustack::util::get_os_name)" == "linux" ]]
}
