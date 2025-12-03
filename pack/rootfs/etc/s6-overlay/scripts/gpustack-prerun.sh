#!/command/with-contenv /bin/bash
# shellcheck disable=SC1008,SC1091
# ================================
# GPUStack migration oneshot service
# ================================


source /etc/profile
SCRIPT_ROOT=/etc/s6-overlay/scripts
source "$SCRIPT_ROOT/base.sh"
ARGS_FILE="/var/lib/gpustack/run/args/gpustack"

# Read arguments from the args file if it exists and is not empty
set --
if [ -s "$ARGS_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        [ -z "$line" ] && continue
        set -- "$@" "$line"
    done < "$ARGS_FILE"
fi

echo "[INFO] Starting gpustack prerun check."
if [ "$#" -gt 0 ]; then
    exec gpustack prerun "$@"
else
    exec gpustack prerun
fi
