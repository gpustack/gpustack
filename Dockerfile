ARG CUDA_VERSION=12.4.1

FROM nvidia/cuda:$CUDA_VERSION-runtime-ubuntu22.04

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    tzdata \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/gpustack
RUN cd /workspace/gpustack && \
    make build

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
    # Install vllm dependencies for x86_64
    WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[vllm]"; \
    else  \
    WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)"; \
    fi && \
    pip install $WHEEL_PACKAGE &&\
    pip cache purge && \
    rm -rf /workspace/gpustack

RUN gpustack download-tools

ENTRYPOINT [ "gpustack", "start" ]
