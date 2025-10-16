#!/bin/bash
set -e

# Create necessary directories for s6-overlay

mkdir -p /etc/s6-overlay/s6-rc.d
mkdir -p /etc/s6-overlay/s6-rc.d/user/contents.d


########################################
# GPUStack s6 Overlay Service
########################################

mkdir -p /etc/s6-overlay/s6-rc.d/gpustack
mkdir -p /etc/s6-overlay/s6-rc.d/gpustack/dependencies.d

# Define service type
cat > /etc/s6-overlay/s6-rc.d/gpustack/type <<EOF
longrun
EOF

# Define service run script
cat > /etc/s6-overlay/s6-rc.d/gpustack/run <<EOF
#!/command/with-contenv /bin/bash
# ================================
# GPUStack longrun service
# ================================

source /etc/profile
exec gpustack start $@
EOF
chmod +x /etc/s6-overlay/s6-rc.d/gpustack/run

# Define user contents
touch /etc/s6-overlay/s6-rc.d/user/contents.d/gpustack

########################################
# Postgres s6 Overlay Service
########################################

if [[ " $* " == *" --database-url "* ]] || [ -n "${DATABASE_URL}" ]; then
    echo "[entrypoint] DATABASE_URL detected or --database-url in arguments, skipping embedded PostgreSQL service creation."
else
    mkdir -p /etc/s6-overlay/s6-rc.d/postgres
    mkdir -p /etc/s6-overlay/s6-rc.d/postgres/dependencies.d
    
    # Define service type
    cat > /etc/s6-overlay/s6-rc.d/postgres/type <<EOF
longrun
EOF

    # Define service run script
    cat > /etc/s6-overlay/s6-rc.d/postgres/run <<EOF
#!/command/execlineb -P
with-contenv
s6-setuidgid postgres
fdmove -c 2 1
/usr/bin/postgres \
  -D /var/lib/postgresql/data \
  -c config_file=/etc/postgresql/main/postgresql.conf \
  -c hba_file=/etc/postgresql/main/pg_hba.conf
EOF
    chmod +x /etc/s6-overlay/s6-rc.d/postgres/run

    # Define service dependencies
    touch /etc/s6-overlay/s6-rc.d/gpustack/dependencies.d/postgres

    # Define user contents
    touch /etc/s6-overlay/s6-rc.d/user/contents.d/postgres
fi

########################################
# Data Migration s6 Overlay Service
########################################

if [ -z "${SQLITE_PATH}" ]; then
  echo "Environment variable SQLITE_PATH is not set, skipping migration."
else

    mkdir -p /etc/s6-overlay/s6-rc.d/gpustack-migration
    mkdir -p /etc/s6-overlay/s6-rc.d/gpustack-migration/dependencies.d

    # Define service type
    cat > /etc/s6-overlay/s6-rc.d/gpustack-migration/type <<EOF
oneshot
EOF

    # Define service up script
    cat > /etc/s6-overlay/s6-rc.d/gpustack-migration/up <<EOF
#!/command/with-contenv /bin/bash
exec gpustack migrate --sqlite-path ${SQLITE_PATH} --database-url "postgresql://root@localhost:5432/gpustack"
EOF

    chmod +x /etc/s6-overlay/s6-rc.d/gpustack-migration/up

    # Define service dependencies
    touch /etc/s6-overlay/s6-rc.d/gpustack-migration/dependencies.d/postgres
    touch /etc/s6-overlay/s6-rc.d/gpustack/dependencies.d/gpustack-migration

    # Define user contents
    touch /etc/s6-overlay/s6-rc.d/user/contents.d/gpustack-migration
fi

exec /init
