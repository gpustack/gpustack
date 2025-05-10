# Online Installation

## Supported Devices

- [x] NVIDIA GPUs (Compute Capability 6.0 and above, check [Your GPU Compute Capability](https://developer.nvidia.com/cuda-gpus))

## Supported Platforms

| OS      | Arch           | Supported methods                                                                                                                                 |
| ------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux   | AMD64<br>ARM64 | [Installation Script](#installation-script)<br>[Docker Installation](#docker-installation) (Recommended)<br>[pip Installation](#pip-installation) |
| Windows | AMD64          | [Installation Script](#installation-script)<br>[pip Installation](#pip-installation)                                                              |

## Supported backends

- [x] vLLM (Compute Capability 7.0 and above, only supports AMD64 Linux)
- [x] llama-box
- [x] vox-box

## Prerequisites

- [Port Requirements](../installation-requirements.md#port-requirements)
- CPU support for llama-box backend: AMD64 with AVX2, or ARM64 with NEON

=== "Linux"

    Check if the CPU is supported:

    === "AMD64"

        ```bash
        lscpu | grep avx2
        ```

    === "ARM64"

        ```bash
        grep -E -i "neon|asimd" /proc/cpuinfo
        ```

=== "Windows"

    Windows users need to manually verify support for the above instructions.

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

## Installation Script

### Prerequites

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

### Run GPUStack

GPUStack provides a script to install it as a service with default port 80.

=== "Linux"

    ```bash
    curl -sfL https://get.gpustack.ai | sh -s -
    ```

    To configure additional environment variables and startup flags when running the script, refer to the [Installation Script](../installation-script.md).

    After installed, ensure that the GPUStack startup logs are normal:

    ```bash
    tail -200f /var/log/gpustack.log
    ```

    If the startup logs are normal, open `http://your_host_ip` in the browser to access the GPUStack UI. Log in to GPUStack with username `admin` and the default password. You can run the following command to get the password for the default setup:

    ```bash
    cat /var/lib/gpustack/initial_admin_password
    ```

    If you specify the `--data-dir` parameter to set the data directory, the `initial_admin_password` file will be located in the specified directory.

=== "Windows"

    ```powershell
    Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content
    ```

    To configure additional environment variables and startup flags when running the script, refer to the [Installation Script](../installation-script.md).

    After installed, ensure that the GPUStack startup logs are normal:

    ```powershell
    Get-Content "$env:APPDATA\gpustack\log\gpustack.log" -Tail 200 -Wait
    ```

    If the startup logs are normal, open `http://your_host_ip` in the browser to access the GPUStack UI. Log in to GPUStack with username `admin` and the default password. You can run the following command to get the password for the default setup:

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\initial_admin_password" -Raw
    ```
    If you specify the `--data-dir` parameter to set the data directory, the `initial_admin_password` file will be located in the specified directory.

### (Optional) Add Worker

=== "Linux"

    To add workers to the GPUStack cluster, you need to specify the server URL and authentication token when installing GPUStack on the workers.

    To get the token used for adding workers, run the following command on the GPUStack **server node**:

    ```bash
    cat /var/lib/gpustack/token
    ```

    If you specify the `--data-dir` parameter to set the data directory, the `token` file will be located in the specified directory.

    To install GPUStack and start it as a worker, and **register it with the GPUStack server**, run the following command on the **worker node**. Be sure to replace the URL and token with your specific values:

    ```bash
    curl -sfL https://get.gpustack.ai | sh -s - --server-url http://your_gpustack_url --token your_gpustack_token
    ```

    After installed, ensure that the GPUStack startup logs are normal:

    ```bash
    tail -200f /var/log/gpustack.log
    ```

=== "Windows"

    To add workers to the GPUStack cluster, you need to specify the server URL and authentication token when installing GPUStack on the workers.

    To get the token used for adding workers, run the following command on the GPUStack **server node**:

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\token" -Raw
    ```

    If you specify the `--data-dir` parameter to set the data directory, the `token` file will be located in the specified directory.

    To install GPUStack and start it as a worker, and **register it with the GPUStack server**, run the following command on the **worker node**. Be sure to replace the URL and token with your specific values:

    ```powershell
    Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --server-url http://your_gpustack_url --token your_gpustack_token"
    ```

    After installed, ensure that the GPUStack startup logs are normal:

    ```powershell
    Get-Content "$env:APPDATA\gpustack\log\gpustack.log" -Tail 200 -Wait
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

Run the following command to start the GPUStack server **and built-in worker**:

=== "Host Network"

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --gpus all \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack
    ```

=== "Port Mapping"

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --gpus all \
        -p 80:80 \
        -p 10150:10150 \
        -p 40064-40095:40064-40095 \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack \
        --worker-ip your_host_ip
    ```

You can refer to the [CLI Reference](../../cli-reference/start.md) for available startup flags.

Check if the startup logs are normal:

```bash
docker logs -f gpustack
```

If the logs are normal, open `http://your_host_ip` in the browser to access the GPUStack UI. Log in to GPUStack with username `admin` and the default password. You can run the following command to get the password for the default setup:

```bash
docker exec -it gpustack cat /var/lib/gpustack/initial_admin_password
```

### (Optional) Add Worker

You can add more GPU nodes to GPUStack to form a GPU cluster. You need to add workers on other GPU nodes and specify the `--server-url` and `--token` parameters to join GPUStack.

To get the token used for adding workers, run the following command on the GPUStack **server node**:

```bash
docker exec -it gpustack cat /var/lib/gpustack/token
```

To start GPUStack as a worker, and **register it with the GPUStack server**, run the following command on the **worker node**. Be sure to replace the URL, token and node IP with your specific values:

=== "Host Network"

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --gpus all \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack \
        --server-url http://your_gpustack_url --token your_gpustack_token
    ```

=== "Port Mapping"

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --gpus all \
        -p 10150:10150 \
        -p 40064-40095:40064-40095 \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack \
        --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_worker_host_ip
    ```

!!! note

    1. **Heterogeneous cluster is supported.** No matter what type of device it is, you can add it to the current GPUStack as a worker by specifying the `--server-url` and `--token` parameters.

    2. You can set additional flags for the `gpustack start` command by appending them to the docker run command.
    For configuration details, please refer to the [CLI Reference](../../cli-reference/start.md).

    3. You can either use the `--ipc=host` flag or `--shm-size` flag to allow the container to access the host’s shared memory. It is used by vLLM and pyTorch to share data between processes under the hood, particularly for tensor parallel inference.

    4. The  `-p 40064-40095:40064-40095` flag is used to ensure connectivity for distributed inference across workers running llama-box RPC servers. For more details, please refer to the [Port Requirements](../installation-requirements.md#port-requirements). You can omit this flag if you don't need distributed inference across workers.

### Build Your Own Docker Image

For example, the official GPUStack NVIDIA CUDA image is built with CUDA 12.4. If you want to use a different CUDA version, you can build your own Docker image.

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

```bash
docker build -t gpustack:cuda-12.8 --build-arg CUDA_VERSION=12.8.1 .
```

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

Run the following to install GPUStack.

=== "Linux AMD64"

    ```bash
    # Extra dependencies options are "vllm", "audio" and "all"
    # "vllm" is only available for Linux AMD64
    pip install "gpustack[all]"
    ```

=== "Linux ARM64 or Windows"

    ```bash
    pip install "gpustack[audio]"
    ```

If you don’t need the vLLM backend and support for audio models, just run:

```bash
pip install gpustack
```

To verify, run:

```bash
gpustack version
```

### Run GPUStack

Run the following command to start the GPUStack server **and built-in worker**:

```bash
gpustack start
```

If the startup logs are normal, open `http://your_host_ip` in the browser to access the GPUStack UI. Log in to GPUStack with username `admin` and the default password. You can run the following command to get the password for the default setup:

=== "Linux"

    ```bash
    cat /var/lib/gpustack/initial_admin_password
    ```

=== "Windows"

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\initial_admin_password" -Raw
    ```

By default, GPUStack uses `/var/lib/gpustack` as the data directory so you need `sudo` or proper permission for that. You can also set a custom data directory by running:

```bash
gpustack start --data-dir mypath
```

You can refer to the [CLI Reference](../../cli-reference/start.md) for available CLI Flags.

### (Optional) Add Worker

To add a worker to the GPUStack cluster, you need to specify the server URL and the authentication token.

To get the token used for adding workers, run the following command on the GPUStack **server node**:

=== "Linux"

    ```bash
    cat /var/lib/gpustack/token
    ```

=== "Windows"

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\token" -Raw
    ```

To start GPUStack as a worker, and **register it with the GPUStack server**, run the following command on the **worker node**. Be sure to replace the URL, token and node IP with your specific values:

```bash
gpustack start --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_worker_host_ip
```

### Run GPUStack as a System Service

A recommended way is to run GPUStack as a startup service. For example, using systemd:

Create a service file in `/etc/systemd/system/gpustack.service`:

```bash
tee /etc/systemd/system/gpustack.service > /dev/null <<EOF
[Unit]
Description=GPUStack Service
Wants=network-online.target
After=network-online.target

[Service]
EnvironmentFile=-/etc/default/%N
ExecStart=$(command -v gpustack) start
Restart=always
StandardOutput=append:/var/log/gpustack.log
StandardError=append:/var/log/gpustack.log

[Install]
WantedBy=multi-user.target
EOF
```

Then start GPUStack:

```bash
systemctl daemon-reload && systemctl enable gpustack --now
```

Check the service status:

```bash
systemctl status gpustack
```

And ensure that the GPUStack startup logs are normal:

```bash
tail -200f /var/log/gpustack.log
```
