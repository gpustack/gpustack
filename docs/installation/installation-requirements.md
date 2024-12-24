# Installation Requirements

This page describes the software and networking requirements for the nodes where GPUStack will be installed.

## Python Requirements

GPUStack requires Python version 3.10 to 3.12.

## Operating System Requirements

GPUStack is supported on the following operating systems:

- [x] macOS
- [x] Windows
- [x] Linux

GPUStack has been tested and verified to work on the following operating systems:

| OS        | Versions        |
| --------- | --------------- |
| Windows   | 10, 11          |
| Ubuntu    | \>= 20.04       |
| Debian    | \>= 11          |
| RHEL      | \>= 8           |
| Rocky     | \>= 8           |
| Fedora    | \>= 36          |
| OpenSUSE  | \>= 15.3 (leap) |
| OpenEuler | \>= 22.03       |

!!! note

    The installation of GPUStack worker on a Linux system requires that the GLIBC version be 2.29 or higher. If your system uses a lower GLIBC version, consider using the [Docker Installation](./docker-installation.md) method as an alternative.

### Supported Architectures

GPUStack supports both **AMD64** and **ARM64** architectures, with the following notes:

- On Linux and macOS, when using Python versions below 3.12, ensure that the installed Python distribution corresponds to your system architecture.
- On Windows, please use the AMD64 distribution of Python, as wheel packages for certain dependencies are unavailable for ARM64. If you use tools like `conda`, this will be handled automatically, as conda installs the AMD64 distribution by default.

## Accelerator Runtime Requirements

GPUStack supports the following accelerators:

- [x] Apple Metal (M-series chips)
- [x] NVIDIA CUDA ([Compute Capability](https://developer.nvidia.com/cuda-gpus) 6.0 and above)
- [x] Ascend CANN
- [x] Moore Threads MUSA
- [x] AMD ROCm

Ensure all necessary drivers and libraries are installed on the system prior to installing GPUStack.

### NVIDIA CUDA

To use NVIDIA CUDA as an accelerator, ensure the following components are installed:

- [NVIDIA CUDA Toolkit 12](https://developer.nvidia.com/cuda-toolkit) (Including CUDA Runtime, cuBLAS, NVIDIA driver)
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) (Optional, required for audio models)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit) (Optional, required for docker installation)

### Ascend CANN

For Ascend CANN as an accelerator, ensure the following components are installed:

- [Ascend NPU driver & firmware](https://www.hiascend.com/en/hardware/firmware-drivers/community)
- [Ascend CANN Toolkit & kernels](https://www.hiascend.com/zh/developer/download/community/result?module=cann)

### MUSA

To use Moore Threads MUSA as an accelerator, ensure the following components are installed:

- [MUSA SDK](https://developer.mthreads.com/sdk/download/musa)
- [MT Container Toolkits](https://developer.mthreads.com/sdk/download/CloudNative)(Optional, required for docker installation)

### AMD ROCm

To use AMD ROCm as an accelerator, ensure the following components are installed:

- [ROCm](https://rocm.docs.amd.com/en/docs-6.1.0/)

## Networking Requirements

### Connectivity Requirements

The following network connectivity is required to ensure GPUStack functions properly:

**Server-to-Worker:** The server must be able to reach the workers for proxying inference requests.

**Worker-to-Server:** Workers must be able to reach the server to register themselves and send updates.

**Worker-to-Worker:** Necessary for distributed inference across multiple workers

### Port Requirements

GPUStack uses the following ports for communication:

**Server Ports**

| Port    | Description                                                              |
| ------- | ------------------------------------------------------------------------ |
| TCP 80  | Default port for the GPUStack UI and API endpoints                       |
| TCP 443 | Default port for the GPUStack UI and API endpoints (when TLS is enabled) |

**Worker Ports**

| Port            | Description                                    |
| --------------- | ---------------------------------------------- |
| TCP 10150       | Default port for the GPUStack worker           |
| TCP 10151       | Default port for exposing metrics              |
| TCP 40000-41024 | Port range allocated for inference services    |
| TCP 50000-51024 | Port range allocated for llama-box RPC servers |
