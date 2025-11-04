#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${ROOT_DIR}/hack/lib/init.sh"

PACKAGE_NAMESPACE=${PACKAGE_NAMESPACE:-gpustack}
PACKAGE_REPOSITORY=${PACKAGE_REPOSITORY:-gpustack}
PACKAGE_ARCH=${PACKAGE_ARCH:-$(uname -m | sed 's/aarch64/arm64/' | sed 's/x86_64/amd64/')}
PACKAGE_TAG=${PACKAGE_TAG:-dev}
PACKAGE_WITH_CACHE=${PACKAGE_WITH_CACHE:-true}
PACKAGE_PUSH=${PACKAGE_PUSH:-false}

function pack() {
    if ! command -v docker &>/dev/null; then
        gpustack::log::fatal "Docker is not installed. Please install Docker to use this target."
        exit 1
    fi

    if ! docker buildx inspect --builder "gpustack" &>/dev/null; then
        gpustack::log::info "Creating new buildx builder 'gpustack'"
        docker run --rm --privileged tonistiigi/binfmt:qemu-v9.2.2-52 --uninstall qemu-*
        docker run --rm --privileged tonistiigi/binfmt:qemu-v9.2.2-52 --install all
        docker buildx create \
            --name "gpustack" \
            --driver "docker-container" \
            --driver-opt "network=host,default-load=true,env.BUILDKIT_STEP_LOG_MAX_SIZE=-1,env.BUILDKIT_STEP_LOG_MAX_SPEED=-1" \
            --buildkitd-flags "--allow-insecure-entitlement=security.insecure --allow-insecure-entitlement=network.host --oci-worker-net=host --oci-worker-gc-keepstorage=204800" \
            --bootstrap
    fi

    TAG="${PACKAGE_NAMESPACE}/${PACKAGE_REPOSITORY}:${PACKAGE_TAG}"
    EXTRA_ARGS=()
	if [[ "${PACKAGE_WITH_CACHE}" == "true" ]]; then
		EXTRA_ARGS+=("--cache-from=type=registry,ref=gpustack/build-cache:gpustack-main")
	fi
	if [[ "${PACKAGE_PUSH}" == "true" ]]; then
		EXTRA_ARGS+=("--push")
	fi
    gpustack::log::info "Building '${TAG}' platform 'linux/${PACKAGE_ARCH}'"
    set -x
    docker buildx build \
        --pull \
        --allow network.host \
        --allow security.insecure \
        --builder "gpustack" \
        --platform "linux/${PACKAGE_ARCH}" \
        --tag "${TAG}" \
        --file "${ROOT_DIR}/pack/Dockerfile" \
        --ulimit nofile=65536:65536 \
        --shm-size 16G \
        --progress plain \
        "${EXTRA_ARGS[@]}" \
        "${ROOT_DIR}"
    set +x
}

gpustack::log::info "+++ PACKAGE +++"
pack
gpustack::log::info "--- PACKAGE ---"
