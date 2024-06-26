#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function lint() {
  local path="$1"

  gpustack::log::info "linting ${path}"
  pre-commit run --all-files
}

#
# main
#

gpustack::log::info "+++ LINT +++"
lint "gpustack"
gpustack::log::info "--- LINT ---"
