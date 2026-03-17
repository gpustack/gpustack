#!/command/with-contenv /bin/bash
# shellcheck disable=SC1008,SC1091
# ================================
# GPUStack migration oneshot service
# ================================


source /etc/profile
SCRIPT_ROOT=/etc/s6-overlay/scripts
source "$SCRIPT_ROOT/base.sh"
ARGS_FILE="/run/gpustack/args"
CUSTOM_CA_DIR="${GPUSTACK_CUSTOM_CA_DIR:-/usr/local/share/ca-certificates}"

# Read arguments from the args file if it exists and is not empty
set --
if [ -s "$ARGS_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        [ -z "$line" ] && continue
        set -- "$@" "$line"
    done < "$ARGS_FILE"
fi

if command -v update-ca-certificates >/dev/null 2>&1; then
    shopt -s nullglob
    custom_ca_certs=("${CUSTOM_CA_DIR}"/*.crt)
    shopt -u nullglob
    if [ ${#custom_ca_certs[@]} -gt 0 ]; then
        echo "[INFO] Updating CA certificates from ${CUSTOM_CA_DIR}."
        update-ca-certificates
    fi
fi

echo "[INFO] Starting gpustack prerun check."
if [ "$#" -gt 0 ]; then
    exec gpustack prerun "$@"
else
    exec gpustack prerun
fi
