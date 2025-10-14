#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

PACKAGE_NAMESPACE=${PACKAGE_NAMESPACE:-gpustack}
PACKAGE_REPOSITORY=${PACKAGE_REPOSITORY:-gpustack}
PACKAGE_OS=${PACKAGE_OS:-$(uname -s | tr '[:upper:]' '[:lower:]')}
PACKAGE_ARCH=${PACKAGE_ARCH:-$(uname -m | sed 's/aarch64/arm64/' | sed 's/x86_64/amd64/')}
PACKAGE_TAG=${PACKAGE_TAG:-dev}

function pack() {
    if ! command -v docker &>/dev/null; then
        gpustack::log::fatal "Docker is not installed. Please install Docker to use this target."
        exit 1
    fi

    if ! docker buildx inspect --builder "gpustack" &>/dev/null; then
        gpustack::log::info "Creating new buildx builder 'gpustack'"
        docker run --rm --privileged tonistiigi/binfmt:qemu-v9.2.2-52 --uninstall qemu-*
        docker run --rm --privileged tonistiigi/binfmt:qemu-v9.2.2 --install all
        docker buildx create \
            --name "gpustack" \
            --driver "docker-container" \
            --buildkitd-flags "--allow-insecure-entitlement security.insecure --allow-insecure-entitlement network.host" \
            --driver-opt "network=host,default-load=true,env.BUILDKIT_STEP_LOG_MAX_SIZE=-1,env.BUILDKIT_STEP_LOG_MAX_SPEED=-1" \
            --bootstrap
    fi

    TAG="${PACKAGE_NAMESPACE}/${PACKAGE_REPOSITORY}:${PACKAGE_TAG}"
    gpustack::log::info "Building '${TAG}' platform '${PACKAGE_OS}/${PACKAGE_ARCH}'"
    docker buildx build \
        --pull \
        --allow network.host \
        --allow security.insecure \
        --builder "gpustack" \
        --platform "${PACKAGE_OS}/${PACKAGE_ARCH}" \
        --tag "${TAG}" \
        --file "${ROOT_DIR}/pack/Dockerfile" \
        --progress plain \
        "${ROOT_DIR}"
}

gpustack::log::info "+++ PACKAGE +++"
pack
gpustack::log::info "--- PACKAGE ---"
