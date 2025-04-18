# Air-Gapped Installation

You can install GPUStack in an air-gapped environment. An air-gapped environment refers to a setup where GPUStack will be installed offline.

The following methods are available for installing GPUStack in an air-gapped environment:

| OS    | Arch  | Supported methods                           |
| ----- | ----- | ------------------------------------------- |
| Linux | AMD64 | [Docker Installation](#docker-installation) |

## Docker Installation

### Prerequisites

- [Port Requirements](../installation-requirements.md#port-requirements)
- CPU support for llama-box backend: AMD64 with AVX2

Check if the CPU is supported:

```bash
lscpu | grep avx2
```

- [Docker](https://docs.docker.com/engine/install/)
- [DCU Driver rock-6.3](https://developer.sourcefind.cn/tool/)

Check if the driver is installed:

```bash
lsmod | grep dcu
```

### Run GPUStack

When running GPUStack with Docker, it works out of the box in an air-gapped environment as long as the Docker images are available. To do this, follow these steps:

1. Pull GPUStack docker image in an online environment:

```bash
docker pull gpustack/gpustack:latest-dcu
```

If your online environment differs from the air-gapped environment in terms of OS or arch, specify the OS and arch of the air-gapped environment when pulling the image:

```bash
docker pull --platform linux/amd64 gpustack/gpustack:latest-dcu
```

2. Publish docker image to a private registry or load it directly in the air-gapped environment.
3. Refer to the [Docker Installation](./online-installation.md#docker-installation) guide to run GPUStack using Docker.
