#!/usr/bin/env bash


function gpustack::util::sed() {
  if ! sed -i "$@" >/dev/null 2>&1; then
    # back off none GNU sed
    sed -i "" "$@"
  fi
}
