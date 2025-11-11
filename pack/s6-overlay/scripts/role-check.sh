#!/command/with-contenv /bin/bash
# shellcheck disable=SC1008
# ================================
# Role check oneshot service
# ================================

ROLE_FILE="/var/lib/gpustack/run/role"
FLAG_START_EMBEDDED_DATABASE_FILE="/var/lib/gpustack/run/flag_start_embedded_database"
FLAG_EMBEDDED_DATABASE_PORT="/var/lib/gpustack/run/flag_embedded_database_port"


# default
echo "server" > "$ROLE_FILE"
echo "1" > "$FLAG_START_EMBEDDED_DATABASE_FILE"
echo "5432" > "$FLAG_EMBEDDED_DATABASE_PORT"


# Detect from the args

ARGS_FILE="/var/lib/gpustack/run/args/gpustack"
set --
if [ -s "$ARGS_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        [ -z "$line" ] && continue
        set -- "$@" "$line"
    done < "$ARGS_FILE"
fi

HAS_CONFIG_FILE=0
CONFIG_FILE_VALUE=""
for ((i=0; i<$#; i++)); do
	arg="${!i}"
	next_idx=$((i+1))
	next_arg="${!next_idx}"
    if [[ ("$arg" == "--server-url" || "$arg" == "-s") && -n "$next_arg" && ! "$next_arg" =~ ^-- ]]; then
        SERVER_URL_VALUE="$next_arg"
		echo "[INFO] detect --server-url/-s argument: $SERVER_URL_VALUE, setting role to worker."
        echo "worker" > "$ROLE_FILE"
        echo "0" > "$FLAG_START_EMBEDDED_DATABASE_FILE"
	fi

    if [[ "$arg" == --database-url && -n "$next_arg" && ! "$next_arg" =~ ^-- ]]; then
		DATABASE_URL_VALUE="$next_arg"
        echo "[INFO] detected --database-url argument: $DATABASE_URL_VALUE."
        echo "0" > "$FLAG_START_EMBEDDED_DATABASE_FILE"
	fi

    if [[ "$arg" == --database-port && -n "$next_arg" && ! "$next_arg" =~ ^-- ]]; then
		DATABASE_PORT_VALUE="$next_arg"
        echo "[INFO] detected --database-port argument: $DATABASE_PORT_VALUE."
        echo "$DATABASE_PORT_VALUE" > "$FLAG_EMBEDDED_DATABASE_PORT"
	fi

    if [[ "$arg" == --config-file && -n "$next_arg" && ! "$next_arg" =~ ^-- ]]; then
		HAS_CONFIG_FILE=1
		CONFIG_FILE_VALUE="$next_arg"
        echo "[INFO] detected --config-file argument: $CONFIG_FILE_VALUE."
	fi
done

# Detect from environment variables

if [ -n "$GPUSTACK_SERVER_URL" ]; then
    echo "[INFO] detect GPUSTACK_SERVER_URL environment, setting role to worker."
	echo "worker" > "$ROLE_FILE"
    echo "0" > "$FLAG_START_EMBEDDED_DATABASE_FILE"
fi

if [ -n "$GPUSTACK_DATABASE_URL" ]; then
    echo "[INFO] detect GPUSTACK_DATABASE_URL environment."
    echo "0" > "$FLAG_START_EMBEDDED_DATABASE_FILE"
fi

if [ -n "$GPUSTACK_DATABASE_PORT" ]; then
    echo "[INFO] detect GPUSTACK_DATABASE_PORT environment."
    echo "$GPUSTACK_DATABASE_PORT" > "$FLAG_EMBEDDED_DATABASE_PORT"
fi

# Detect from config file

if [[ $HAS_CONFIG_FILE -eq 1 && -n "$CONFIG_FILE_VALUE" ]]; then
    database_url_output=$(yq -r '.database_url' "$CONFIG_FILE_VALUE" 2>/dev/null)
    if [[ -n "$database_url_output" && "$database_url_output" != "null" ]]; then
        echo "[INFO] detect database_url in config file: $database_url_output."
        echo "0" > "$FLAG_START_EMBEDDED_DATABASE_FILE"
    fi

    database_port_output=$(yq -r '.database_port' "$CONFIG_FILE_VALUE" 2>/dev/null)
    if [[ -n "$database_port_output" && "$database_port_output" != "null" ]]; then
        echo "[INFO] detect database_port in config file: $database_port_output."
        echo "$database_port_output" > "$FLAG_EMBEDDED_DATABASE_PORT"
    fi

    server_url_output=$(yq -r '.server_url' "$CONFIG_FILE_VALUE" 2>/dev/null)
    if [[ -n "$server_url_output" && "$server_url_output" != "null" ]]; then
        echo "[INFO] detect server_url in config file: $server_url_output."
        echo "worker" > "$ROLE_FILE"
        echo "0" > "$FLAG_START_EMBEDDED_DATABASE_FILE"
    fi
fi
