# Air-Gapped Installation

You can install GPUStack in an air-gapped environment. An air-gapped environment refers to a setup where GPUStack will be installed offline.

The following methods are available for installing GPUStack in an air-gapped environment:

| OS      | Arch           | Supported methods                                                                                  |
| ------- | -------------- | -------------------------------------------------------------------------------------------------- |
| Linux   | AMD64<br>ARM64 | [Docker Installation](#docker-installation) (Recommended)<br>[pip Installation](#pip-installation) |
| Windows | AMD64          | [pip Installation](#pip-installation)                                                              |

## Supported backends

- [x] vLLM (Compute Capability 7.0 and above, only supports Linux AMD64)
- [x] llama-box
- [x] vox-box

## Prerequisites

- [Port Requirements](../installation-requirements.md#port-requirements)
- [NVIDIA Driver](https://www.nvidia.com/en-us/drivers/)

Check if the NVIDIA driver is installed:

```bash
nvidia-smi --format=csv,noheader --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu
```

And ensure the driver supports CUDA 12.4 or higher:

=== "Linux"

    ```bash
    nvidia-smi | grep "CUDA Version"
    ```

=== "Windows"

    ```powershell
    nvidia-smi | findstr "CUDA Version"
    ```

## Docker Installation

### Prerequisites

- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Check if Docker and NVIDIA Container Toolkit are installed:

```bash
docker info | grep Runtimes | grep nvidia
```

- [Disabling Systemd Cgroup Management in Docker](https://github.com/NVIDIA/nvidia-container-toolkit/issues/48)

!!! note

    When systemd is used to manage the cgroups of the container and it is triggered to reload any Unit files that have references to NVIDIA GPUs (e.g. systemctl daemon-reload), containerized GPU workloads may suddenly lose access to their GPUs.

    In GPUStack, GPUs may be lost in the Resources menu, and running `nvidia-smi` inside the GPUStack container may result in the error: `Failed to initialize NVML: Unknown Error`

    To prevent [this issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/48), disabling systemd cgroup management in Docker is required.

Set the parameter "exec-opts": ["native.cgroupdriver=cgroupfs"] in the `/etc/docker/daemon.json` file and restart docker, such as:

```bash
vim /etc/docker/daemon.json
```

```json
{
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  },
  "exec-opts": ["native.cgroupdriver=cgroupfs"]
}
```

```bash
systemctl daemon-reload && systemctl restart docker
```

### Run GPUStack

When running GPUStack with Docker, it works out of the box in an air-gapped environment as long as the Docker images are available. To do this, follow these steps:

1. Pull GPUStack docker image in an online environment:

```bash
docker pull gpustack/gpustack
```

If your online environment differs from the air-gapped environment in terms of OS or arch, specify the OS and arch of the air-gapped environment when pulling the image:

```bash
docker pull --platform linux/amd64 gpustack/gpustack
```

2. Publish docker image to a private registry or load it directly in the air-gapped environment.
3. Refer to the [Docker Installation](./online-installation.md#docker-installation) guide to run GPUStack using Docker.

## pip Installation

### Prerequisites

- Python 3.10 ~ 3.12

Check the Python version:

```bash
python -V
```

- [NVIDIA CUDA Toolkit 12](https://developer.nvidia.com/cuda-toolkit)

Check if CUDA is installed and verify that its version is at least 12.4:

```bash
nvcc -V
```

- [NVIDIA cuDNN 9](https://developer.nvidia.com/cudnn) (Optional, required for audio models)

Check if cuDNN 9 is installed:

=== "Linux"

    ```bash
    ldconfig -p | grep libcudnn
    ```

=== "Windows"

    ```powershell
    Get-ChildItem -Path C:\ -Recurse -Filter "cudnn*.dll" -ErrorAction SilentlyContinue
    ```

### Install GPUStack

For manually pip installation, you need to prepare the required packages and tools in an online environment and then transfer them to the air-gapped environment.

Set up an online environment identical to the air-gapped environment, including **OS**, **architecture**, and **Python version**.

#### Step 1: Download the Required Packages

=== "Linux"

    Run the following commands in an online environment:

    === "AMD64"

        ```bash
        # Extra dependencies options are "vllm", "audio" and "all"
        # "vllm" is only available for Linux AMD64
        PACKAGE_SPEC="gpustack[all]"
        # To install a specific version
        # PACKAGE_SPEC="gpustack[all]==0.6.0"
        ```

    === "ARM64"

        ```bash
        PACKAGE_SPEC="gpustack[audio]"
        # To install a specific version
        # PACKAGE_SPEC="gpustack[audio]==0.6.0"
        ```

    If you don’t need the vLLM backend and support for audio models, just set:

    ```bash
    PACKAGE_SPEC="gpustack"
    ```

=== "Windows"

    Run the following commands in an online environment:

    ```powershell
    $PACKAGE_SPEC = "gpustack[audio]"
    # To install a specific version
    # $PACKAGE_SPEC = "gpustack[audio]==0.6.0"
    ```

    If you don’t need support for audio models, just set:

    ```powershell
    $PACKAGE_SPEC = "gpustack"
    ```

Download all required packages:

```bash
pip wheel $PACKAGE_SPEC -w gpustack_offline_packages
```

Install GPUStack to use its CLI:

```bash
pip install gpustack
```

Download dependency tools and save them as an archive:

```bash
gpustack download-tools --save-archive gpustack_offline_tools.tar.gz
```

If your online environment differs from the air-gapped environment, specify the **OS**, **architecture**, and **device** explicitly:

```bash
gpustack download-tools --save-archive gpustack_offline_tools.tar.gz --system linux --arch amd64 --device cuda
```

!!!note

    This instruction assumes that the online environment uses the same GPU type as the air-gapped environment. If the GPU types differ, use the `--device` flag to specify the device type for the air-gapped environment. Refer to the [download-tools](../../cli-reference/download-tools.md) command for more information.

#### Step 2: Transfer the Packages

Transfer the following files from the online environment to the air-gapped environment.

- `gpustack_offline_packages` directory.
- `gpustack_offline_tools.tar.gz` file.

#### Step 3: Install GPUStack

In the air-gapped environment, run the following commands.

Install GPUStack from the downloaded packages:

```bash
pip install --no-index --find-links=gpustack_offline_packages gpustack
```

Load and apply the pre-downloaded tools archive:

```bash
gpustack download-tools --load-archive gpustack_offline_tools.tar.gz
```

Now you can run GPUStack by following the instructions in the [pip Installation](online-installation.md#pip-installation) guide.
