#!/bin/bash
# shellcheck disable=SC1091,SC1090

cd "$(dirname -- "$0")"
ROOT=$(pwd)
cd - >/dev/null
source "$ROOT/base.sh"

waitForConfig "Higress Config" "$GPUSTACK_GATEWAY_CONFIG"
source "$GPUSTACK_GATEWAY_CONFIG"
source "$ROOT/default-variables.sh"

echo "GATEWAY_HTTP_PORT=$GATEWAY_HTTP_PORT"
echo "GATEWAY_HTTPS_PORT=$GATEWAY_HTTPS_PORT"

readinessCheck "Higress API Server" "${APISERVER_PORT}"

set -e

/usr/local/bin/higress \
    serve \
    --kubeconfig="${EMBEDDED_KUBECONFIG_PATH}" \
    --gatewaySelectorKey=higress \
    --gatewaySelectorValue=higress-system-higress-gateway \
    --gatewayHttpPort="$GATEWAY_HTTP_PORT" \
    --gatewayHttpsPort="$GATEWAY_HTTPS_PORT" \
    --ingressClass=
