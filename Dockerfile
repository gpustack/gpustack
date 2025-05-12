ARG CUDA_VERSION=12.4.1
ARG CUDA_TAG_SUFFIX=-cudnn-runtime-ubuntu22.04

FROM nvidia/cuda:${CUDA_VERSION}${CUDA_TAG_SUFFIX}

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    tzdata \
    iproute2 \
    python3 \
    python3-pip \
    python3-venv \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/gpustack
RUN cd /workspace/gpustack && \
    make build

ARG VLLM_VERSION=0.8.5.post1
RUN <<EOF
    if [ "$TARGETPLATFORM" = "linux/amd64" ]; then
        # Install vllm dependencies for x86_64
        if [ "$(echo "${CUDA_VERSION}" | cut -d. -f1,2)" = "11.8" ]; then
            pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp38-abi3-manylinux1_x86_64.whl \
            --extra-index-url https://download.pytorch.org/whl/cu118;
        fi;
        WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[all]";
    else
        WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[audio]";
    fi
    pip install pipx
    pip install $WHEEL_PACKAGE
    pip cache purge
    rm -rf /workspace/gpustack
EOF

RUN gpustack download-tools

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
