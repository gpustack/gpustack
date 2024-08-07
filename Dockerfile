ARG CUDA_VERSION=12.5.1

FROM nvidia/cuda:$CUDA_VERSION-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

ARG GPUSTACK_VERSION=0.1.0

RUN pip3 install gpustack==$GPUSTACK_VERSION

ENTRYPOINT [ "gpustack", "start" ]
