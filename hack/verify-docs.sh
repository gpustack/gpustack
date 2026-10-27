#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT_DIR}"

# The default Docker context excludes both artifacts. Supply them explicitly.
wheel_dir="$(mktemp -d)"
trap 'rm -rf "${wheel_dir}"' EXIT
bash hack/build-docs.sh --offline
uv build --wheel --out-dir "${wheel_dir}"
docker build \
  --build-context "wheel=${wheel_dir}" \
  --build-context "help=${ROOT_DIR}/gpustack/help" \
  --file pack/Dockerfile.verify --tag gpustack-help-verify .
