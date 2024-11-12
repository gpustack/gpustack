ARG CUDA_VERSION=12.5.1

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
    pip3 install $WHEEL_PACKAGE &&\
    pip3 cache purge && \
    rm -rf /workspace/gpustack

ENTRYPOINT [ "gpustack", "start" ]
