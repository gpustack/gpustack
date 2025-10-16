#!/bin/bash
# shellcheck disable=SC1091,SC1090

cd "$(dirname -- "$0")"
ROOT=$(pwd)
cd - >/dev/null

source "$ROOT/base.sh"

waitForConfig "Higress Config" "$GPUSTACK_GATEWAY_CONFIG"
source "$GPUSTACK_GATEWAY_CONFIG"
source "$ROOT/default-variables.sh"

set -e

if [ -z "$EMBEDDED_KUBECONFIG_PATH" ]; then
    echo "  Missing required variable EMBEDDED_KUBECONFIG_PATH in apiserver configuration."
    exit 255
fi
mkdir -p "$APISERVER_DATA_DIR"

# prepare default data
cp -rn /opt/data/defaultConfig/* "${APISERVER_DATA_DIR}/"
# prepare kubeconfig
cat <<EOF >"$EMBEDDED_KUBECONFIG_PATH"
apiVersion: v1
kind: Config
clusters:
  - name: higress
    cluster:
      server: https://localhost:${APISERVER_PORT}
      insecure-skip-tls-verify: true
users:
  - name: higress-admin
contexts:
  - name: higress
    context:
      cluster: higress
      user: higress-admin
preferences: {}
current-context: higress
EOF

# prepare mesh config
MESH_CONFIG_DIR='/etc/istio/config'
mkdir -p "$MESH_CONFIG_DIR"
HIGRESS_CONFIG_FILE="${APISERVER_DATA_DIR}/configmaps/higress-config.yaml"
MESH_CONFIG_FILES=$(yq '.data | keys | .[]' "$HIGRESS_CONFIG_FILE")
if [ -z "$MESH_CONFIG_FILES" ]; then
    echo "  Missing required files in higress-config ConfigMap."
    exit 255
fi
IFS=$'\n'
for MESH_CONFIG_FILE in $MESH_CONFIG_FILES; do
    if [ -z "$MESH_CONFIG_FILE" ] || [ "$MESH_CONFIG_FILE" == "higress" ]; then
        continue
    fi
    yq ".data.$MESH_CONFIG_FILE" "$HIGRESS_CONFIG_FILE" > "$MESH_CONFIG_DIR/$MESH_CONFIG_FILE"
done

apiserver --bind-address 127.0.0.1 --secure-port "${APISERVER_PORT}" --storage file --file-root-dir "${APISERVER_DATA_DIR}" --cert-dir /tmp
