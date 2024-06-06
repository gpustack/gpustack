#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function build() {
  poetry build
}

#
# main
#

gpustack::log::info "+++ BUILD +++"
build
gpustack::log::info "--- BUILD ---"