# Docker Installation

You can use the official Docker image to run GPUStack in a container. Installation using docker is supported on:

- Linux with Nvidia GPUs

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Run GPUStack with Docker

Run the following command to start the GPUStack server:

```shell
docker run -d --gpus all -p 80:80 --ipc=host \
    -v gpustack-data:/var/lib/gpustack gpustack/gpustack
```

!!! note

    You can either use the `--ipc=host` flag or `--shm-size` flag to allow the container to access the host’s shared memory. It is used by vLLM and pyTorch to share data between processes under the hood, particularly for tensor parallel inference.

You can set additional flags for the `gpustack start` command by appending them to the docker run command.

For example, to start a GPUStack worker:

```shell
docker run -d --gpus all --ipc=host --network=host \
    gpustack/gpustack --server-url http://myserver --token mytoken
```

!!! note

    The `--network=host` flag is used to ensure that server is accessible to the worker and inference services running on it. Alternatively, you can set `--worker-ip <host-ip> -p 10150:10150 -p 40000-41024:40000-41024` to expose relevant ports.

For configuration details, please refer to the [CLI Reference](../cli-reference/start.md).

## Run GPUStack with Docker Compose

Get the docker-compose file from GPUStack repository, run the following command to start the GPUStack server:

```shell
docker-compose up -d
```

You can update the `docker-compose.yml` file to customize the command while starting a GPUStack worker.

## Build Your Own Docker Image

The official Docker image is built with CUDA 12.5. If you want to use a different version of CUDA, you can build your own Docker image.

```dockerfile
# Example Dockerfile
ARG CUDA_VERSION=12.5.1

FROM nvidia/cuda:$CUDA_VERSION-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    tzdata \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install gpustack[vllm] && \
    pip3 cache purge

ENTRYPOINT [ "gpustack", "start" ]
```

Run the following command to build the Docker image:

```shell
docker build -t my/gpustack --build-arg CUDA_VERSION=12.0.0 .
```
