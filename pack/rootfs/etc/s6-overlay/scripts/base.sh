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

GPUSTACK_GATEWAY_DIR="${GPUSTACK_GATEWAY_DIR:-/var/lib/gpustack/higress}"
createDir "$GPUSTACK_GATEWAY_DIR"
# shellcheck disable=SC2034
GPUSTACK_GATEWAY_CONFIG="${GPUSTACK_GATEWAY_DIR}/.env"

GPUSTACK_POSTGRES_DIR="${GPUSTACK_POSTGRES_DIR:-/var/lib/gpustack/postgres}"
createDir "$GPUSTACK_POSTGRES_DIR"
# shellcheck disable=SC2034
GPUSTACK_POSTGRES_CONFIG="${GPUSTACK_POSTGRES_DIR}/.env"
