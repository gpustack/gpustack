ARG CUDA_VERSION=12.5.1

FROM nvidia/cuda:$CUDA_VERSION-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install gpustack
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.5/compat:${LD_LIBRARY_PATH}

ENTRYPOINT [ "gpustack", "start" ]
