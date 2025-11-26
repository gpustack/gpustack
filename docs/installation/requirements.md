# Installation Requirements

This page outlines the software and networking requirements for nodes running GPUStack.

## Operating System Requirements

GPUStack supports most modern Linux distributions on **AMD64** and **ARM64** architectures.

!!! note

    - GPUStack is not recommended for direct installation via PyPi. For best compatibility, use the provided Docker images.
    - The Network Time Protocol (NTP) package must be installed to ensure consistent state synchronization between nodes.

GPUStack has been tested and verified on the following operating systems:

| OS        | Versions        |
| --------- | --------------- |
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

### Connectivity Requirements

The following network connectivity is required for GPUStack to function properly:

**Server-to-Worker:** The server must be able to reach workers to proxy inference requests.

**Worker-to-Server:** Workers must be able to reach the server to register and send updates.

**Worker-to-Worker:** Required for distributed inference across multiple workers.

### Port Requirements

GPUStack uses these ports for communication:

#### Server Ports

| Port      | Description                                                  |
| --------- | ------------------------------------------------------------ |
| TCP 80    | Default port for GPUStack UI and API endpoints               |
| TCP 443   | Default port for GPUStack UI and API endpoints (TLS enabled) |
| TCP 10161 | Default port for server metrics endpoint                     |
| TCP 8080  | Default port for GPUStack server internal API                |
| TCP 5432  | Default port for embedded Postgres Database                  |

#### Worker Ports

| Port            | Description                                   |
| --------------- | --------------------------------------------- |
| TCP 10150       | Default port for GPUStack worker              |
| TCP 10151       | Default port for worker metrics endpoint      |
| TCP 8080        | Default port for GPUStack worker internal API |
| TCP 40000-40063 | Port range for inference services             |
| TCP 41000-41999 | Port range for Ray services(vLLM distributed deployment using) |

#### Embedded Gateway Ports

The embedded gateway for both server and worker uses the following ports for internal communications.

| Port      | Host      | Description                                          |
| --------- | --------- | ---------------------------------------------------- |
| TCP 18443 | 127.0.0.1 | Port for the file-based APIServer serving via HTTPS  |
| TCP 15000 | 127.0.0.1 | Management port for the Envoy gateway                |
| TCP 15021 | 0.0.0.0   | Health check port for the Envoy gateway              |
| TCP 15090 | 0.0.0.0   | Metrics port for the Envoy gateway                   |
| TCP 9876  | 127.0.0.1 | Introspection port for the Pilot-discovery           |
| TCP 15010 | 127.0.0.1 | Port for Pilot-discovery serving XDS via HTTP/gRPC   |
| TCP 15012 | 127.0.0.1 | Port for Pilot-discovery serving XDS via secure gRPC |
| TCP 15020 | 0.0.0.0   | Metrics port for Pilot-agent                         |
| TCP 8888  | 127.0.0.1 | Port for Controller serving XDS via HTTP             |
| TCP 15051 | 127.0.0.1 | Port for Controller serving XDS via gRPC             |

##### Distributed vLLM with Ray Ports

When using distributed vLLM, GPUStack will parse the above port range for Ray services,
and assign them in order as below:

1. GCS server port (the first port of the range)
2. Client Server port
3. Dashboard port
4. Dashboard gRPC port (no longer used since Ray 2.45.0, kept for backward compatibility)
5. Dashboard agent gRPC port
6. Dashboard agent listen port
7. Metrics export port
8. Node Manager port
9. Object Manager port
10. Raylet runtime env agent port
11. Minimum port number for the worker
12. Maximum port number for the worker (the last port of the range)

For more details on Ray ports, see the [Ray documentation](https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations).
