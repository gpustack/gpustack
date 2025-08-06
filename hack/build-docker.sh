#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

docker buildx build -t gpustack/gpustack:dev -f pack/Dockerfile --platform linux/amd64,linux/arm64 .
