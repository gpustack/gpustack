#!/bin/bash

function readinessCheck() {
    # $1=name
    # $2=port
    # $3=timeout
    timeout="${3:-0}"
    while true; do
        if [ "$timeout" -gt 0 ]; then
            timeout=$((timeout - 1))
            if [ "$timeout" -eq 0 ]; then
                echo "[ERROR] Timeout while waiting for $1 to be ready."
                exit 1
            fi
        fi
        echo "Checking the readiness of $1..."
        if nc -z 127.0.0.1 "$2"; then
            break
        fi
        sleep 1
    done
}

function createDir() {
    mkdir -p "$1"
    chmod 755 "$1"
}

function waitForConfig() {
    local name="$1"
    local path="$2"
    local silent_mode="${3:-false}"
    local count=0
    local sleep_time=1
    while true; do
        if [ -f "$path" ]; then
            break
        fi
        ((count++))
            if [ $count -le 10 ]; then
                if [ "$silent_mode" != "true" ]; then
                    echo "Waiting for $name configuration from GPUStack..."
                fi
                sleep_time=1
        else
            if [ "$silent_mode" != "true" ]; then
                echo "$name configuration is still missing. This component may be disabled."
            fi
            sleep_time=$((sleep_time * 2))
            [ $sleep_time -gt 30 ] && sleep_time=30
        fi
        sleep $sleep_time
    done
}

function handleServiceExit() {
  local service_name="$1"
  local exit_code_service="$2"
  local exit_code_signal="$3"

  local exit_code_file="/run/s6-linux-init-container-results/exitcode"
  local exit_code_container=0

  [[ -f "${exit_code_file}" ]] && exit_code_container=$(<"${exit_code_file}")

  echo "[INFO] Service '${service_name}' exited (code: ${exit_code_service}, signal: ${exit_code_signal})"

  # Case 1: Exit by signal
  if [[ "${exit_code_service}" -eq 256 ]]; then
    # If SIGTERM, stop the container
    if [[ "${exit_code_signal}" -eq 15 ]]; then
      echo 0 > "${exit_code_file}"
    fi

    # Write translated signal code if the container exit code isn't already set
    if [[ "${exit_code_container}" -eq 0 ]]; then
      echo $((128 + exit_code_signal)) > "${exit_code_file}"
    fi

    echo "[INFO] Service '${service_name}' exited by signal, shutting down container..."
    exec /run/s6/basedir/bin/halt
  
  # Case 2: non-zero exit → fatal → shutdown container
  elif [[ "${exit_code_service}" -ne 0 ]]; then

    # Update container exit code if not already set
    if [[ "${exit_code_container}" -eq 0 ]]; then
      echo "${exit_code_service}" > "${exit_code_file}"
    fi

    echo "[INFO] Service '${service_name}' exited with non-zero code, shutting down container..."
    exec /run/s6/basedir/bin/halt
  fi

  # Case 3: zero exit → exit normally
  echo "[INFO] Service '${service_name}' exited normally."
  exec /run/s6/basedir/bin/halt
}


export GPUSTACK_GATEWAY_DIR="${GPUSTACK_GATEWAY_DIR:-/var/lib/gpustack/higress}"
createDir "$GPUSTACK_GATEWAY_DIR"
# shellcheck disable=SC2034
export GPUSTACK_GATEWAY_CONFIG="${GPUSTACK_GATEWAY_DIR}/.env"

export GPUSTACK_POSTGRES_DIR="${GPUSTACK_POSTGRES_DIR:-/var/lib/gpustack/postgresql}"
createDir "$GPUSTACK_POSTGRES_DIR"
# shellcheck disable=SC2034
export GPUSTACK_POSTGRES_CONFIG="${GPUSTACK_POSTGRES_DIR}/.env"
