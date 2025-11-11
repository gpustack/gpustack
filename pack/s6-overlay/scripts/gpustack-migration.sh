#!/command/with-contenv /bin/bash
# shellcheck disable=SC1008
# ================================
# GPUStack migration oneshot service
# ================================

STATE_MIGRATION_DONE_FILE="/var/lib/gpustack/run/state_migration_done"
FLAG_EMBEDDED_DATABASE_PORT_FILE="/var/lib/gpustack/run/flag_embedded_database_port"
EMBEDDED_DATABASE_PORT="5432"
if [ -f "$FLAG_EMBEDDED_DATABASE_PORT_FILE" ]; then
    EMBEDDED_DATABASE_PORT=$(cat "$FLAG_EMBEDDED_DATABASE_PORT_FILE")
fi

if [ -n "${GPUSTACK_MIGRATION_DATA_DIR}" ]; then
    if [ -f "$STATE_MIGRATION_DONE_FILE" ]; then
        echo "[INFO] Migration already completed previously. Skipping."
        exit 0
    fi

    echo "[INFO] Using GPUSTACK_MIGRATION_DATA_DIR: ${GPUSTACK_MIGRATION_DATA_DIR}."
    if gpustack migrate --migration-data-dir "${GPUSTACK_MIGRATION_DATA_DIR}" \
        --database-url "postgresql://root@localhost:${EMBEDDED_DATABASE_PORT}/gpustack"; then
        touch "$STATE_MIGRATION_DONE_FILE"
    else
        echo "[ERROR] Migration failed."
        exit 1
    fi
else
    echo "[INFO] No migration data dir specified, skipping."
fi
