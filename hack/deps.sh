#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function deps() {
  uv sync --all-packages --locked
  uv lock
  uv tree
}

#
# main
#

gpustack::log::info "+++ DEPS +++"
deps
gpustack::log::info "--- DEPS ---"
