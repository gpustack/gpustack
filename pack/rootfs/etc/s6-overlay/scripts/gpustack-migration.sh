#!/command/with-contenv /bin/bash
# shellcheck disable=SC1008,SC1090,SC1091
# ================================
# GPUStack migration oneshot service
# ================================

SCRIPT_ROOT=/etc/s6-overlay/scripts
source "$SCRIPT_ROOT/base.sh"
# The config should be ready before starting
source "$GPUSTACK_POSTGRES_CONFIG"
source "$SCRIPT_ROOT/default-variables.sh"


if [ "${GPUSTACK_DATA_MIGRATION}" = "true" ]; then
    if [ -f "$STATE_MIGRATION_DONE_FILE" ]; then
        echo "[INFO] Migration already completed previously. Skipping."
        exit 0
    fi

    echo "[INFO] Using GPUSTACK_MIGRATION_DATA_DIR: ${DATA_DIR}."
    if gpustack migrate --migration-data-dir "${DATA_DIR}" \
        --database-url "postgresql://root@localhost:${EMBEDDED_DATABASE_PORT}/gpustack"; then
        # shellcheck disable=SC2086
        mkdir -p "$(dirname ${STATE_MIGRATION_DONE_FILE})"
        touch "$STATE_MIGRATION_DONE_FILE"
        echo "[INFO] Migration completed successfully."
    else
        echo "[ERROR] Migration failed."
        exit 1
    fi
else
    echo "[INFO] No migration data dir specified, skipping."
fi
