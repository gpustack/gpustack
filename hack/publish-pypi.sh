#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

if [[ -z "${PYPI_API_TOKEN:-}" ]]; then
  gpustack::log::error "PYPI_API_TOKEN is not set"
  exit 1
fi

poetry publish --username __token__ --password $PYPI_API_TOKEN
