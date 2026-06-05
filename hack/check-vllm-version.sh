#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Extract vllm version from model_registry.py sync comment
# Expected format: # Synced with https://github.com/vllm-project/vllm/blob/v0.X.Y/vllm/model_executor/models/registry.py
REGISTRY_VERSION=$(sed -n 's/.*vllm\/blob\/v\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)\/.*/\1/p' "${ROOT_DIR}/gpustack/scheduler/model_registry.py" | head -1)

if [ -z "${REGISTRY_VERSION}" ]; then
  echo "ERROR: Could not extract vllm version from model_registry.py sync comment"
  exit 1
fi

# Extract vllm version from pyproject.toml optional-dependencies
PYPROJECT_VERSION=$(sed -n 's/.*vllm==\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' "${ROOT_DIR}/pyproject.toml" | head -1)

if [ -z "${PYPROJECT_VERSION}" ]; then
  echo "ERROR: Could not extract vllm version from pyproject.toml"
  exit 1
fi

if [ "${REGISTRY_VERSION}" != "${PYPROJECT_VERSION}" ]; then
  echo "ERROR: vllm version mismatch detected!"
  echo "  model_registry.py synced from: v${REGISTRY_VERSION}"
  echo "  pyproject.toml pinned to:      v${PYPROJECT_VERSION}"
  echo ""
  echo "These versions must match. Update pyproject.toml or re-sync model_registry.py."
  exit 1
fi

echo "vllm version alignment check passed: v${REGISTRY_VERSION}"
