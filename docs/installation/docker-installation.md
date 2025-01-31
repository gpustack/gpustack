# Docker Installation

You can use the official Docker image to run GPUStack in a container. Installation using docker is supported on:

## Supported Platforms

- [x] Linux

## Supported Devices

- [x] NVIDIA GPUs ([Compute Capability](https://developer.nvidia.com/cuda-gpus) 6.0 and above)
- [x] AMD GPUs
- [x] Ascend NPUs
- [x] Moore Threads GPUs
- [x] CPUs (AVX2 for x86 or NEON for ARM)

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/)

## Run GPUStack with Docker

!!! note

    1. **Heterogeneous clusters are supported.**

    2. You can set additional flags for the `gpustack start` command by appending them to the docker run command.
    For configuration details, please refer to the [CLI Reference](../cli-reference/start.md).

    3. You can either use the `--ipc=host` flag or `--shm-size` flag to allow the container to access the host’s shared memory. It is used by vLLM and pyTorch to share data between processes under the hood, particularly for tensor parallel inference.

    4. The  `-p 10150:10150 -p 40000-41024:40000-41024 -p 50000-51024:50000-51024` and `--worker-ip your_host_ip` flags are used to ensure that server is accessible to the worker and inference services running on it.  Alternatively, you can set the `--network=host` and  `--worker-ip your_host_ip` flags to instead.

### NVIDIA CUDA

#### Prerequisites

- [NVIDIA Drvier 550+](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

#### Run GPUStack

Run the following command to start the GPUStack server **and built-in worker**:

```shell
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    -p 80:80 \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

or

```shell
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    -p 80:80 \
    -p 10150:10150 \
    -p 40000-41024:40000-41024 \
    -p 50000-51024:50000-51024 \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack --worker-ip your_host_ip
```

(**Optional**) Run the following command to start the GPUStack server **without** built-in worker:

```shell
docker run -d --name gpustack-server \
    --restart=unless-stopped \
    -p 80:80 \
    -v gpustack-server-data:/var/lib/gpustack \
    gpustack/gpustack:latest-cpu \
    --disable-worker
```

#### (Optional) Add Worker

To retrieve the token, run the following command on the GPUStack server host:

```shell
docker exec -it gpustack-server cat /var/lib/gpustack/token
```

To start a GPUStack worker and **register it with the GPUStack server**, run the following command on the current host or another host. Replace your specific URL, token, and IP address accordingly:

```shell
docker run -d --name gpustack-worker \
    --restart=unless-stopped \
    --gpus all \
    -p 10150:10150 \
    -p 40000-41024:40000-41024 \
    -p 50000-51024:50000-51024 \
    --ipc=host \
    -v gpustack-worker-data:/var/lib/gpustack \
    gpustack/gpustack \
    --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_host_ip
```

### AMD ROCm

#### Prerequisites

- [AMDGPU driver and ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.4/install/install-overview.html#package-manager-versus-amdgpu)

Refer to this [Tutorial](../tutorials/running-inference-with-amd-gpus.md).

#### Run GPUStack

Run the following command to start the GPUStack server **and built-in worker**:

```shell
docker run -d --name gpustack \
    --restart=unless-stopped \
    -p 80:80 \
    --ipc=host \
    --group-add=video \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-rocm
```

(**Optional**) Run the following command to start the GPUStack server **without** built-in worker:

```shell
docker run -d --name gpustack-server \
    --restart=unless-stopped \
    -p 80:80 \
    -v gpustack-server-data:/var/lib/gpustack \
    gpustack/gpustack:latest-cpu \
    --disable-worker
```

#### (Optional) Add Worker

To retrieve the token, run the following command on the GPUStack server host:

```shell
docker exec -it gpustack-server cat /var/lib/gpustack/token
```

To start a GPUStack worker and **register it with the GPUStack server**, run the following command on the current host or another host. Replace your specific URL, token, and IP address accordingly:

```shell
docker run -d --name gpustack-worker \
    --restart=unless-stopped \
    -p 10150:10150 \
    -p 40000-41024:40000-41024 \
    -p 50000-51024:50000-51024 \
    --ipc=host \
    --group-add=video \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v gpustack-worker-data:/var/lib/gpustack \
    gpustack/gpustack:latest-rocm \
    --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_host_ip
```

### Ascend CANN

#### Prerequisites

- [NPU driver and firmware for Ascend 910B](https://www.hiascend.com/hardware/firmware-drivers/community?product=2&model=28&cann=8.0.RC2.beta1&driver=1.0.25.alpha)
- [Ascend Docker Runtime](https://gitee.com/ascend/ascend-docker-runtime/releases)

Refer to this [Tutorial](../tutorials/running-inference-with-ascend-npus.md).

#### Run GPUStack

Run the following command to start the GPUStack server **and built-in worker** ( Set `ASCEND_VISIBLE_DEVICES` to the required GPU indices ):

```shell
docker run -d --name gpustack \
    --restart=unless-stopped \
    -e ASCEND_VISIBLE_DEVICES=0 \
    -p 80:80 \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-npu
```

(**Optional**) Run the following command to start the GPUStack server **without** built-in worker:

```shell
docker run -d --name gpustack-server \
    --restart=unless-stopped \
    -p 80:80 \
    -v gpustack-server-data:/var/lib/gpustack \
    gpustack/gpustack:latest-cpu \
    --disable-worker
```

#### (Optional) Add Worker

To retrieve the token, run the following command on the GPUStack server host:

```shell
docker exec -it gpustack-server cat /var/lib/gpustack/token
```

To start a GPUStack worker and **register it with the GPUStack server**, run the following command on the current host or another host. Replace your specific URL, token, and IP address accordingly:

```shell
docker run -d --name gpustack-worker \
    --restart=unless-stopped \
    -e ASCEND_VISIBLE_DEVICES=0 \
    -p 10150:10150 \
    -p 40000-41024:40000-41024 \
    -p 50000-51024:50000-51024 \
    --ipc=host \
    -v gpustack-worker-data:/var/lib/gpustack \
    gpustack/gpustack:latest-npu \
    --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_host_ip
```

### Moore Threads MUSA

#### Prerequisites

- [Latest drivers for MTT S80/S3000/S4000](https://developer.mthreads.com/sdk/download/musa)
- [MT Container Toolkits](https://developer.mthreads.com/sdk/download/CloudNative)

Refer to this [Tutorial](../tutorials/running-inference-with-moorethreads-gpus.md).

#### Run GPUStack

Run the following command to start the GPUStack server **and built-in worker**:

```shell
docker run -d --name gpustack \
    --restart=unless-stopped \
    -p 80:80 \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-musa
```

(**Optional**) Run the following command to start the GPUStack server **without** built-in worker:

```shell
docker run -d --name gpustack-server \
    --restart=unless-stopped \
    -p 80:80 \
    -v gpustack-server-data:/var/lib/gpustack \
    gpustack/gpustack:latest-cpu \
    --disable-worker
```

#### (Optional) Add Worker

To retrieve the token, run the following command on the GPUStack server host:

```shell
docker exec -it gpustack-server cat /var/lib/gpustack/token
```

To start a GPUStack worker and **register it with the GPUStack server**, run the following command on the current host or another host. Replace your specific URL, token, and IP address accordingly:

```shell
docker run -d --name gpustack-worker \
    --restart=unless-stopped \
    -p 10150:10150 \
    -p 40000-41024:40000-41024 \
    -p 50000-51024:50000-51024 \
    --ipc=host \
    -v gpustack-worker-data:/var/lib/gpustack \
    gpustack/gpustack:latest-musa \
    --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_host_ip
```

## Build Your Own Docker Image

For example, the official GPUStack NVIDIA CUDA image is built with CUDA 12.4. If you want to use a different version of CUDA, you can build your own Docker image.

```dockerfile
# Example Dockerfile
ARG CUDA_VERSION=12.4.1

FROM nvidia/cuda:$CUDA_VERSION-cudnn-runtime-ubuntu22.04

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

ENTRYPOINT [ "gpustack", "start" ]
```

Run the following command to build the Docker image:

```shell
docker build -t my/gpustack --build-arg CUDA_VERSION=12.0.0 .
```

For other accelerators, refer to the corresponding Dockerfile in the [GPUStack repository](https://github.com/gpustack/gpustack).
