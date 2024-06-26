#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function generate() {
  poetry run gen
}

#
# main
#

gpustack::log::info "+++ GENERATE +++"
generate
gpustack::log::info "--- GENERATE ---"
