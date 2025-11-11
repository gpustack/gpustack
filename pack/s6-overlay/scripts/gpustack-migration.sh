#!/command/with-contenv /bin/bash
# shellcheck disable=SC1008
# ================================
# GPUStack migration oneshot service
# ================================

FLAG_EMBEDDED_DATABASE_PORT_FILE="/var/lib/gpustack/run/flag_embedded_database_port"
EMBEDDED_DATABASE_PORT="5432"
if [ -f "$FLAG_EMBEDDED_DATABASE_PORT_FILE" ]; then
    EMBEDDED_DATABASE_PORT=$(cat "$FLAG_EMBEDDED_DATABASE_PORT_FILE")
fi

if [ -n "${GPUSTACK_MIGRATION_DATA_DIR}" ]; then
    echo "[INFO] Using GPUSTACK_MIGRATION_DATA_DIR: ${GPUSTACK_MIGRATION_DATA_DIR}."
    exec gpustack migrate --migration-data-dir "${GPUSTACK_MIGRATION_DATA_DIR}" --database-url "postgresql://root@localhost:${EMBEDDED_DATABASE_PORT}/gpustack"
else
    echo "[INFO] No migration data dir specified, skipping."
fi
