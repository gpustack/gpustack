#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -d "${ROOT_DIR}/docs-zh" || -d "${ROOT_DIR}/docs-ja" || -d "${ROOT_DIR}/docs-ru" || -d "${ROOT_DIR}/docs-tr" ]]; then
  exec bash "${ROOT_DIR}/hack/build-docs.sh" --serve
fi
uv run mkdocs serve --livereload
