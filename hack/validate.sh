#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

check_sha256sum() {
  local file=$1

  # Generate the current checksum
  sha256sum "$file" > "${file}.generated.sha256sum"

  # Compare with the expected checksum
  if cmp -s "${file}.sha256sum" "${file}.generated.sha256sum"; then
    gpustack::log::info "Checksums match."
    rm "${file}.generated.sha256sum"
    return 0
  else
    gpustack::log::fatal 'Checksums do not match!\nPlease run "sha256sum install.sh > install.sh.sha256sum" to update the checksum.'
  fi
}


check_sha256sum "install.sh"
