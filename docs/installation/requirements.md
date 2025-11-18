# Installation Requirements

This page outlines the software and networking requirements for nodes running GPUStack.

## Operating System Requirements

GPUStack supports most modern Linux distributions on **AMD64** and **ARM64** architectures.

!!! note

    - GPUStack is not recommended for direct installation via PyPi. For best compatibility, use the provided Docker images.
    - The Network Time Protocol (NTP) package must be installed to ensure consistent state synchronization between nodes.

GPUStack has been tested and verified on the following operating systems:

| OS        | Versions        |
|-----------|-----------------|
| Ubuntu    | \>= 20.04       |
| Debian    | \>= 11          |
| RHEL      | \>= 8           |
| Rocky     | \>= 8           |
| Fedora    | \>= 36          |
| OpenSUSE  | \>= 15.3 (Leap) |
| OpenEuler | \>= 22.03       |

## Accelerator Runtime Requirements

GPUStack supports a variety of General-Purpose Accelerators as inference backends, including:

- [x] NVIDIA GPU
- [x] AMD GPU
- [x] Ascend NPU
- [x] Hygon DCU (Experimental)
- [x] MThreads GPU (Experimental)
- [x] Iluvatar GPU (Experimental)
- [x] MetaX GPU (Experimental)
- [x] Cambricon MLU (Experimental)

Ensure all required drivers and toolkits are installed before running GPUStack.

### NVIDIA GPU

To use NVIDIA GPU, install:

- [NVIDIA GPU Driver](https://www.nvidia.com/en-us/drivers/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit)

### AMD GPU

To use AMD GPU, install:

- [AMD GPU Driver](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
- [AMD Container Runtime](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/overview.html)

### Ascend NPU

For Ascend NPU, install:

- [Ascend NPU Driver](https://www.hiascend.com/hardware/firmware-drivers/community)
- [Ascend Docker Runtime](https://www.hiascend.com/document/detail/zh/mindcluster/72rc1/clustersched/dlug/dlug_installation_017.html)

### Hygon DCU

To use Hygon DCU, install:

- [Hygon DCU Driver](https://developer.sourcefind.cn/tool/)
- [Hygon DTK Toolkit](https://developer.sourcefind.cn/tool/)

### MThreads GPU

To use MThreads GPU, install:

- [MThreads GPU Driver](https://docs.mthreads.com/)
- [MThreads Container Toolkit](https://developer.mthreads.com/sdk/download/CloudNative)

### Iluvatar GPU

To use Iluvatar GPU, install:

- [Iluvatar GPU Driver](https://support.iluvatar.com/#/login)
- [Iluvatar Container Toolkit](https://github.com/Deep-Spark/ix-container-toolkit)

### MetaX GPU

To use MetaX GPU, install:

- [MetaX GPU Driver](https://developer.metax-tech.com/softnova/download?package_kind=Driver)
- [MetaX MACA SDK](https://developer.metax-tech.com/softnova/download?package_kind=SDK)

### Cambricon MLU

To use Cambricon MLU, install:

- Cambricon MLU Driver
- Cambricon NeuWare Toolkit

## Networking Requirements

### Network Architecture

The following diagram illustrates the GPUStack network architecture:

![gpustack-network-architecture](../assets/gpustack-network-architecture.png)

### Connectivity Requirements

The following network connectivity is required for GPUStack to function properly:

**Server-to-Worker:** The server must be able to reach workers to proxy inference requests.

**Worker-to-Server:** Workers must be able to reach the server to register and send updates.

**Worker-to-Worker:** Required for distributed inference across multiple workers.

### Port Requirements

GPUStack uses these ports for communication:

#### Server Ports

| Port      | Description                                                              |
|-----------|--------------------------------------------------------------------------|
| TCP 80    | Default port for GPUStack UI and API endpoints                           |
| TCP 443   | Default port for GPUStack UI and API endpoints (TLS enabled)             |
| TCP 10161 | Default port for GPUStack server metrics                                 |

#### Worker Ports

| Port            | Description                                  |
|-----------------|----------------------------------------------|
| TCP 10150       | Default port for GPUStack worker             |
| TCP 10151       | Default port for exposing metrics            |
| TCP 40000-40063 | Port range for inference services            |
| TCP 40064-40095 | Port range for distributed serving           |

##### Distributed vLLM with Ray Ports

When distributed vLLM is enabled, GPUStack uses the following Ray ports:

| Ray Port        | Description                                      |
|-----------------|--------------------------------------------------|
| TCP 6379        | Default port for Ray (GCS server)                |
| TCP 10001       | Default port for Ray Client Server               |
| TCP 8265        | Default port for Ray dashboard                   |
| TCP <random>    | Default port for Ray node manager                |
| TCP <random>    | Default port for Ray object manager              |
| TCP <random>    | Default port for Ray dashboard agent gRPC listen |
| TCP 52365       | Default port for Ray dashboard agent HTTP listen |
| TCP 10002-19999 | Port range for Ray worker processes              |

For more details on Ray ports, see the [Ray documentation](https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations).
