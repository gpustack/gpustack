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

- [Driver for MTT S80/S3000/S4000](https://developer.mthreads.com/sdk/download/musa)

Check if the driver is installed:

```bash
mthreads-gmi
```

- [Docker](https://docs.docker.com/engine/install/)
- [MT Container Toolkits](https://developer.mthreads.com/sdk/download/CloudNative)

Check if the MT Container Toolkits are installed and set as the default runtime:

```bash
# cd /usr/bin/musa && sudo ./docker setup $PWD
docker info | grep Runtimes | grep mthreads
```

### Run GPUStack

When running GPUStack with Docker, it works out of the box in an air-gapped environment as long as the Docker images are available. To do this, follow these steps:

1. Pull GPUStack docker image in an online environment:

```bash
docker pull gpustack/gpustack:latest-musa
```

If your online environment differs from the air-gapped environment in terms of OS or arch, specify the OS and arch of the air-gapped environment when pulling the image:

```bash
docker pull --platform linux/amd64 gpustack/gpustack:latest-musa
```

2. Publish docker image to a private registry or load it directly in the air-gapped environment.
3. Refer to the [Docker Installation](./online-installation.md#docker-installation) guide to run GPUStack using Docker.
