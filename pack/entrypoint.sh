#!/bin/bash
# shellcheck disable=SC1091,SC1090

set -e

source /etc/s6-overlay/scripts/base.sh
ARGS_FILE="/run/gpustack/args"
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

# Check and ensure nofile ulimit is at least 65535
REQUIRED_NOFILE=65535
CURRENT_NOFILE=$(ulimit -n)
if [ "$CURRENT_NOFILE" != "unlimited" ] && [ "$CURRENT_NOFILE" -lt "$REQUIRED_NOFILE" ]; then
    echo "[INFO] Current nofile ulimit ($CURRENT_NOFILE) is below required ($REQUIRED_NOFILE), attempting to increase..."
    if ! ulimit -n "$REQUIRED_NOFILE" 2>/dev/null; then
        echo "[WARN] Failed to set nofile ulimit to $REQUIRED_NOFILE. Current value: $CURRENT_NOFILE."
        echo "[WARN] To fix this, try the following:"
        echo "[WARN]   1. Run the container with: --ulimit nofile=${REQUIRED_NOFILE}:${REQUIRED_NOFILE}"
        echo "[WARN]   2. If that still fails, check the host kernel limit: sysctl fs.nr_open"
        echo "[WARN]      and raise it if needed: sysctl -w fs.nr_open=1048576"
    else
        echo "[INFO] nofile ulimit set to $(ulimit -n)."
    fi
fi

# remove generated gateway config to force regeneration
rm -rf "${GPUSTACK_GATEWAY_CONFIG}"

export S6_STAGE2_HOOK="/etc/s6-overlay/scripts/gpustack-prerun.sh"

# shellcheck disable=SC2068
exec /init gpustack start $@
