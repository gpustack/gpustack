#!/bin/bash
# shellcheck disable=SC1091,SC1090,SC2035

cd "$(dirname -- "$0")"
ROOT=$(pwd)
cd - >/dev/null
source "$ROOT/base.sh"

function initCerts() {
    RSA_KEY_LENGTH=4096

    createDir /etc/certs
    cd /etc/certs

    openssl req -newkey rsa:$RSA_KEY_LENGTH -nodes -keyout root-key.pem -x509 -days 36500 -out root-cert.pem >/dev/null 2>&1 <<EOF
CN
Shanghai
Shanghai
Higress
Gateway
Root CA
rootca@higress.io


EOF

    cat <<EOF >ca.cfg
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = CN
ST = Shanghai
L = Shanghai
O = Higress
CN = Higress CA

[v3_req]
keyUsage = keyCertSign
basicConstraints = CA:TRUE
subjectAltName = @alt_names

[alt_names]
DNS.1 = ca.higress.io
EOF
    openssl genrsa -out ca-key.pem $RSA_KEY_LENGTH >/dev/null &&
        openssl req -new -key ca-key.pem -out ca-cert.csr -config ca.cfg -batch -sha256 >/dev/null 2>&1 &&
        openssl x509 -req -days 36500 -in ca-cert.csr -sha256 -CA root-cert.pem -CAkey root-key.pem -CAcreateserial -out ca-cert.pem -extensions v3_req -extfile ca.cfg >/dev/null 2>&1
    cp ca-cert.pem cert-chain.pem >/dev/null
    chmod a+r ca-key.pem

    cat <<EOF >gateway.cfg
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = CN
ST = Shanghai
L = Shanghai
O = Higress
CN = Higress Gateway

[v3_req]
keyUsage = digitalSignature, keyEncipherment
subjectAltName = URI:spiffe://cluster.local/ns/higress-system/sa/higress-gateway
EOF
    openssl genrsa -out gateway-key.pem $RSA_KEY_LENGTH > /dev/null \
      && openssl req -new -key gateway-key.pem -out gateway-cert.csr -config gateway.cfg -batch -sha256 > /dev/null 2>&1 \
      && openssl x509 -req -days 36500 -in gateway-cert.csr -sha256 -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial -out gateway-cert.pem -extensions v3_req -extfile gateway.cfg > /dev/null 2>&1
    chmod a+r gateway-key.pem

    cat ca-cert.pem >> gateway-cert.pem
    mv gateway-cert.pem cert-chain.pem
    mv gateway-key.pem key.pem

    rm *.csr >/dev/null
    rm *.cfg >/dev/null

    cd -
}

waitForConfig "Higress Config" "$GPUSTACK_GATEWAY_CONFIG"
source "$GPUSTACK_GATEWAY_CONFIG"
source "$ROOT/default-variables.sh"

# higress controller port is configured in configmap higress-config
readinessCheck "Higress Controller" 15051

set -e

initCerts

PILOT_FILTER_GATEWAY_CLUSTER_CONFIG=${PILOT_FILTER_GATEWAY_CLUSTER_CONFIG:-true} \
/usr/local/bin/pilot-discovery \
    discovery \
    --kubeconfig="${EMBEDDED_KUBECONFIG_PATH}" \
    --httpAddr=:15080 \
    --monitoringAddr=:15014 \
    --log_output_level=default:info \
    --domain=cluster.local \
    --keepaliveMaxServerConnectionAge=30m \
    --caCertFile=/etc/certs/ca-cert.pem \
    --meshConfig=/etc/istio/config/mesh \
    --networksConfig=/etc/istio/config/meshNetworks
