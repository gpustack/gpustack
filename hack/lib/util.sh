#!/usr/bin/env bash


function gpustack::util::sed() {
  if ! sed -i "$@" >/dev/null 2>&1; then
    # back off none GNU sed
    sed -i "" "$@"
  fi
}

function gpustack::util::is_darwin() {
  [[ "$(uname -s)" == "Darwin" ]]
}

function gpustack::util::is_linux() {
  [[ "$(uname -s)" == "Linux" ]]
}
