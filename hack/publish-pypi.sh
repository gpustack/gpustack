#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

DIST="dist/*.whl"
if [[ ${PUBLISH_SOURCE:-} == "1" ]]; then
  DIST="dist/*"
fi

# shellcheck disable=SC2086
uv run twine check $DIST
# shellcheck disable=SC2086
uv run twine upload $DIST
