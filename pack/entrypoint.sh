#!/bin/bash
# shellcheck disable=SC1091,SC1090

set -e

source /etc/s6-overlay/scripts/base.sh
ARGS_FILE="/var/lib/gpustack/run/args/gpustack"
mkdir -p "$(dirname "$ARGS_FILE")"

# If any arguments are passed to the container, save them to the args file
if [ "$#" -gt 0 ]; then
    echo "[INFO] Saving docker run args to $ARGS_FILE"
    : > "$ARGS_FILE"
    for arg in "$@"; do
        printf '%s\n' "$arg" >> "$ARGS_FILE"
    done
else
    echo "[INFO] No docker run args passed."
    : > "$ARGS_FILE"
fi

# remove generated gateway config to force regeneration
rm -rf "${GPUSTACK_GATEWAY_CONFIG}"

exec /init
