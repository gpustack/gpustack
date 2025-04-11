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

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
    # Install vllm dependencies for x86_64
    WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[all]"; \
    else  \
    WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[audio]"; \
    fi && \
    pip install pipx && \
    pip install $WHEEL_PACKAGE && \
    pip cache purge && \
    rm -rf /workspace/gpustack

RUN gpustack download-tools

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
