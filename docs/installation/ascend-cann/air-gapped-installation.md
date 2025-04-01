# Air-Gapped Installation

You can install GPUStack in an air-gapped environment. An air-gapped environment refers to a setup where GPUStack will be installed offline.

The following methods are available for installing GPUStack in an air-gapped environment:

| OS    | Arch  | Supported methods                           |
| ----- | ----- | ------------------------------------------- |
| Linux | ARM64 | [Docker Installation](#docker-installation) |

## Docker Installation

### Prerequisites

- [Port Requirements](../installation-requirements.md#port-requirements)
- [NPU Driver and Firmware](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.0.0.beta1&driver=1.0.28.alpha) (Must supports CANN 8.0.0.beta1)

Check if the NPU driver is installed:

```bash
npu-smi info
```

- [Docker](https://docs.docker.com/engine/install/)
- [Ascend Docker Runtime](https://gitee.com/ascend/ascend-docker-runtime/releases)

Check if Docker and Ascend Docker Runtime are installed:

```bash
docker info | grep Runtimes | grep ascend
```

### Run GPUStack

When running GPUStack with Docker, it works out of the box in an air-gapped environment as long as the Docker images are available. To do this, follow these steps:

1. Pull GPUStack docker image in an online environment:

```bash
docker pull gpustack/gpustack:latest-npu
```

If your online environment differs from the air-gapped environment in terms of OS or arch, specify the OS and arch of the air-gapped environment when pulling the image:

```bash
docker pull --platform linux/arm64 gpustack/gpustack:latest-npu
```

2. Publish docker image to a private registry or load it directly in the air-gapped environment.
3. Refer to the [Docker Installation](./online-installation.md#docker-installation) guide to run GPUStack using Docker.
