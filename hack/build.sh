#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function build() {
  poetry build
}

function prepare_dependencies() {
  bash "${ROOT_DIR}/hack/install.sh"
}

#
# main
#

gpustack::log::info "+++ BUILD +++"
prepare_dependencies
build
gpustack::log::info "--- BUILD ---"
