#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function build() {
  poetry build
}

function prepare_deps() {
  bash "${ROOT_DIR}/hack/deps.sh"
}

#
# main
#

gpustack::log::info "+++ BUILD +++"
prepare_deps
build
gpustack::log::info "--- BUILD ---"