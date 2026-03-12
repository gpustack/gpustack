# Installation Requirements

This page outlines the software and networking requirements for nodes running GPUStack.

## Operating System Requirements

GPUStack supports most modern Linux distributions on **AMD64** and **ARM64** architectures.

!!! note

    - GPUStack is not supported for direct installation via PyPi. For best compatibility, use the provided Docker images.
    - The Network Time Protocol (NTP) package must be installed to ensure consistent state synchronization between nodes.

## Accelerator Runtime Requirements

GPUStack supports a variety of General-Purpose Accelerators as inference backends, including:

- [x] NVIDIA GPU
- [x] AMD GPU
- [x] Ascend NPU
- [x] Hygon DCU
- [x] MThreads GPU (Experimental)
- [x] Iluvatar GPU (Experimental)
- [x] MetaX GPU (Experimental)
- [x] Cambricon MLU (Experimental)
- [x] T-Head PPU (Experimental)

Ensure all required drivers and toolkits are installed before running GPUStack.

### NVIDIA GPU

#### Requirements

- [NVIDIA GPU Driver](https://www.nvidia.com/en-us/drivers/) that supports NVIDIA CUDA 12.6 or higher.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit)

Run the following commands to verify:

```bash
sudo nvidia-smi

# If using Docker
sudo docker info 2>/dev/null | grep -q "nvidia" \
    && echo "NVIDIA Container Toolkit OK" \
    || (echo "NVIDIA Container Toolkit not configured"; exit 1)
```

#### Supported Inference Backends

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] [VoxBox](https://github.com/gpustack/vox-box)
- [x] Custom

### AMD GPU

#### Requirements

- [AMD GPU Driver](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) that supports AMD ROCm 6.4 or higher.
- [AMD Container Runtime](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/overview.html)

Run the following commands to verify:

```bash
sudo amd-smi static

# If using Docker
sudo docker info 2>/dev/null | grep -q "amd" \
    && echo "AMD Container Toolkit OK" \
    || (echo "AMD Container Toolkit not configured"; exit 1)
```

#### Supported Inference Backends

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang) (requires gfx9 series GPUs)
- [x] Custom

### Ascend NPU

#### Requirements

- [Ascend NPU Driver](https://www.hiascend.com/hardware/firmware-drivers/community)
- [Ascend Docker Runtime](https://www.hiascend.com/document/detail/zh/mindcluster/730/clustersched/dlug/dlug_installation_017.html)

Run the following commands to verify:

```bash
sudo npu-smi info

# If using Docker
sudo docker info 2>/dev/null | grep -q "ascend" \
    && echo "Ascend Container Toolkit OK" \
    || (echo "Ascend Container Toolkit not configured"; exit 1)
```

#### Supported Devices

- [x] Ascend NPU 910C series
- [x] Ascend NPU 910B series (910B1 ~ 910B4)
- [x] Ascend NPU 310P3

#### Supported Inference Backends

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] [MindIE](https://www.hiascend.com/document/detail/zh/mindie/21RC2/index/index.html)
- [x] Custom

### Hygon DCU

#### Requirements

- [Hygon DCU Driver](https://developer.sourcefind.cn/tool/)
- [Hygon DTK Toolkit](https://developer.sourcefind.cn/tool/)

Run the following commands to verify:

```bash
sudo hy-smi
```

#### Supported Devices

- [x] Hygon DCUs

#### Supported Inference Backends

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] Custom

### MThreads GPU

#### Requirements

- [MThreads GPU Driver](https://docs.mthreads.com/)
- [MThreads Container Toolkit](https://developer.mthreads.com/sdk/download/CloudNative)

Run the following commands to verify:

```bash
sudo mthreads-gmi

# If using Docker
sudo docker info 2>/dev/null | grep -q "mthreads" \
    && echo "MThreads Container Toolkit OK" \
    || (echo "MThreads Container Toolkit not configured"; exit 1)
```

#### Supported Inference Backends

- [x] Custom

### Iluvatar GPU

#### Requirements

- [Iluvatar GPU Driver](https://support.iluvatar.com/#/login)
- [Iluvatar Container Toolkit](https://github.com/Deep-Spark/ix-container-toolkit)

Run the following commands to verify:

```bash
sudo ixsmi

# If using Docker
sudo docker info 2>/dev/null | grep -q "iluvatar" \
    && echo "Iluvatar Container Toolkit OK" \
    || (echo "Iluvatar Container Toolkit not configured"; exit 1)
```

#### Supported Inference Backends

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] Custom

### MetaX GPU

#### Requirements

- [MetaX GPU Driver](https://developer.metax-tech.com/softnova/download?package_kind=Driver)
- [MetaX MACA SDK](https://developer.metax-tech.com/softnova/download?package_kind=SDK)

Run the following commands to verify:

```bash
sudo mx-smi
```

#### Supported Inference Backends

- [x] Custom

### Cambricon MLU

#### Requirements

- Cambricon MLU Driver
- Cambricon NeuWare Toolkit

Run the following commands to verify:

```bash
sudo cnmon
```

#### Supported Inference Backends

- [x] Custom

### T-Head PPU

#### Requirements

- T-Head PPU Driver
- T-Head PPU SDK

Run the following commands to verify:

```bash
sudo ppu-smi
```

#### Supported Inference Backends

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] Custom

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
| TCP 30080 | Default port for GPUStack server internal API                |
| TCP 5432  | Default port for embedded Postgres Database                  |

#### Worker Ports

| Port            | Description                                                    |
| --------------- | -------------------------------------------------------------- |
| TCP 10150       | Default port for GPUStack worker                               |
| TCP 10151       | Default port for worker metrics endpoint                       |
| TCP 40000-40063 | Port range for inference services                              |
| TCP 41000-41999 | Port range for Ray services(vLLM distributed deployment using) |

##### Distributed vLLM with Ray Ports

When using distributed vLLM, GPUStack will parse the above port range for Ray services,
and assign them in order as below:

1. GCS server port (the first port of the range)
2. Client Server port (reserved for compatibility, not used anymore, see https://github.com/gpustack/gpustack/issues/4171)
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
