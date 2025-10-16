#!/bin/bash
# shellcheck disable=SC1091,SC1090

set -e

cd "$(dirname -- "$0")"
ROOT=$(pwd)
cd - >/dev/null
source "$ROOT/base.sh"

export GPUSTACK_GATEWAY_CONFIG
export GPUSTACK_GATEWAY_DIR

GPUSTACK_EXTRA_ARGS=${GPUSTACK_EXTRA_ARGS:-""}
# shellcheck disable=SC2086
exec /usr/local/bin/gpustack $GPUSTACK_EXTRA_ARGS
