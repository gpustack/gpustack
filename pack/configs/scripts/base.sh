#!/bin/bash

function readinessCheck() {
    # $1=name
    # $2=port
    while true; do
        echo "Checking the readiness of $1..."
        if nc -z 127.0.0.1 "$2"; then
            break
        fi
        sleep 1
    done
}

function createDir() {
    sudo mkdir -p "$1"
}

function waitForConfig() {
    local name="$1"
    local path="$2"
    local count=0
    local sleep_time=1
    while true; do
        if [ -f "$path" ]; then
            break
        fi
        ((count++))
        if [ $count -le 10 ]; then
            echo "Waiting for $name configuration from GPUStack..."
            sleep_time=1
        else
            echo "$name configuration is still missing. This component may be disabled."
            sleep_time=$((sleep_time * 2))
            [ $sleep_time -gt 30 ] && sleep_time=30
        fi
        sleep $sleep_time
    done
}

GPUSTACK_GATEWAY_DIR="${GPUSTACK_GATEWAY_DIR:-/var/lib/gpustack/higress}"
mkdir -p "$GPUSTACK_GATEWAY_DIR"
# shellcheck disable=SC2034
GPUSTACK_GATEWAY_CONFIG="${GPUSTACK_GATEWAY_DIR}/variables.sh"
