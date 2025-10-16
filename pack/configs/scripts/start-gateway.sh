#!/bin/bash
# shellcheck disable=SC1091,SC1090

cd "$(dirname -- "$0")"
ROOT=$(pwd)
cd - >/dev/null
source "$ROOT/base.sh"

waitForConfig "Higress Config" "$GPUSTACK_GATEWAY_CONFIG"
source "$GPUSTACK_GATEWAY_CONFIG"
source "$ROOT/default-variables.sh"

# The pilot port is configured in mesh config, default to 15010
readinessCheck "Higress Pilot" 15010

set -e


createDir /etc/istio/proxy
createDir "${ISTIO_DATA_DIR}"
if [ -e /var/lib/istio/data ] || [ -L /var/lib/istio/data ]; then
    rm -rf /var/lib/istio/data
fi
ln -s "${ISTIO_DATA_DIR}" /var/lib/istio/data

ACCESS_LOG_PATH="${HIGRESS_LOG_DIR}/access.log"
createDir "${HIGRESS_LOG_DIR}"
touch "${ACCESS_LOG_PATH}"
sed -i -E "s;^accessLogFile: .+$;accessLogFile: ${ACCESS_LOG_PATH};" /etc/istio/config/mesh


/usr/local/bin/higress-proxy-start.sh proxy router \
    --concurrency="${GATEWAY_CONCURRENCY}" \
    --domain=higress-system.svc.cluster.local \
    --proxyLogLevel=warning \
    --proxyComponentLogLevel=misc:error \
    --log_output_level=all:info \
    --serviceCluster=higress-gateway
