#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

function test() {
  uv run pytest
}

#
# main
#

gpustack::log::info "+++ TEST +++"
test
gpustack::log::info "--- TEST ---"
