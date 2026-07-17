#!/command/with-contenv /bin/bash
# shellcheck shell=bash
# shellcheck disable=SC1091,SC1090
# ================================
# Embedded Postgres log cleanup (invoked by supercronic; see cron.txt)
# ================================
# Postgres' logging_collector rotates logs but never deletes them, so without
# this the logs grow unbounded (issue #5774). Resolve LOG_DIR exactly the way
# the postgres longrun service does — sourcing base.sh, the postgres .env, then
# default-variables.sh — so a custom --data-dir / log dir is honoured rather
# than a hardcoded /var/lib/gpustack path.

SCRIPT_ROOT=/etc/s6-overlay/scripts
source "$SCRIPT_ROOT/base.sh"
# base.sh exports GPUSTACK_POSTGRES_CONFIG; fall back to its default path so a
# custom --data-dir written into that file is still honoured even if the var is
# somehow unset. The file is written by prerun only when the embedded database
# is enabled; it is absent for external databases, where there are no
# embedded-Postgres logs to clean.
GPUSTACK_POSTGRES_CONFIG="${GPUSTACK_POSTGRES_CONFIG:-${GPUSTACK_RUN_DIR:-/run/gpustack}/postgresql/.env}"
if [ -f "$GPUSTACK_POSTGRES_CONFIG" ]; then
	source "$GPUSTACK_POSTGRES_CONFIG"
fi
source "$SCRIPT_ROOT/default-variables.sh"

# Number of days of Postgres logs to retain (override via env). Anything last
# modified more than this many days ago is deleted.
RETENTION_DAYS="${POSTGRES_LOG_RETENTION_DAYS:-30}"
if [[ ! "$RETENTION_DAYS" =~ ^[0-9]+$ ]]; then
	echo "[ERROR] POSTGRES_LOG_RETENTION_DAYS must be a non-negative integer, got: '$RETENTION_DAYS'" >&2
	exit 1
fi
PGLOG="${LOG_DIR}/postgresql"

if [ -d "$PGLOG" ]; then
	find "$PGLOG" -name 'postgresql-*.log' -type f -mtime "+${RETENTION_DAYS}" -delete
fi
