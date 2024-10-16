ARG CUDA_VERSION=12.5.1

FROM nvidia/cuda:$CUDA_VERSION-runtime-ubuntu22.04

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    tzdata \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY dist/*.whl /tmp/

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
    echo "Installing x86_64 wheel"; \
    WHEEL_PACKAGE="$(ls /tmp/*_x86_64.whl)[vllm]"; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
    echo "Installing aarch64 wheel"; \
    WHEEL_PACKAGE="$(ls /tmp/*_aarch64.whl)"; \
    else \
    echo "Unsupported platform: $TARGETPLATFORM" && exit 1; \
    fi && \
    pip3 install $WHEEL_PACKAGE &&\
    rm /tmp/*.whl && \
    pip3 cache purge

ENTRYPOINT [ "gpustack", "start" ]
