# Online Installation

## Supported Devices

- [x] AMD GPUs

## Supported Platforms

| OS      | Version                      | Arch  | Supported methods                                                                                        |
| ------- | ---------------------------- | ----- | -------------------------------------------------------------------------------------------------------- |
| Linux   | Ubuntu 22.04<br>Ubuntu 24.04 | AMD64 | [Docker Installation](#docker-installation) (Recommended)<br>[Installation Script](#installation-script) |
| Windows | 10<br>11<br>Server 2022      | AMD64 | [Installation Script](#installation-script)                                                              |

## Prerequisites

- [Port Requirements](../installation-requirements.md#port-requirements)
- CPU support for llama-box backend: AMD64 with AVX2

=== "Linux"

    Check if the CPU is supported:

    ```bash
    lscpu | grep avx2
    ```

=== "Windows"

    Windows users need to manually verify support for the above instructions.

## Docker Installation

### Supported Devices

| Devices                                             | Supported Backends |
| --------------------------------------------------- | ------------------ |
| gfx1100: AMD Radeon RX 7900 XTX/7900 XT/7900 GRE    | llama-box, vLLM    |
| gfx1101: AMD Radeon RX 7800 XT/7700 XT              | llama-box, vLLM    |
| gfx1102: AMD Radeon RX 7600 XT/7600                 | llama-box, vLLM    |
| gfx942: AMD Instinct MI300X/MI300A                  | llama-box, vLLM    |
| gfx90a: AMD Instinct MI250X/MI250/MI210             | llama-box, vLLM    |
| gfx1030: AMD Radeon RX 6950 XT/6900 XT/6800 XT/6800 | llama-box          |
| gfx1031: AMD Radeon RX 6750 XT/6700 XT/6700         | llama-box          |
| gfx1032: AMD Radeon RX 6650 XT/6600 XT/6600         | llama-box          |
| gfx908: AMD Instinct MI100                          | llama-box          |
| gfx906: AMD Instinct MI50                           | llama-box          |

### Prerequisites

- [Docker](https://docs.docker.com/engine/install/)
- [ROCm 6.2.4](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.4/install/install-overview.html)

Select the appropriate installation method for your system. Here, we provide steps for Linux (Ubuntu 22.04). For other systems, refer to the ROCm documentation.

1. Install ROCm:

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/jammy/amdgpu-install_6.2.60204-1_all.deb
sudo apt install ./amdgpu-install_6.2.60204-1_all.deb
amdgpu-install -y --usecase=graphics,rocm
sudo reboot
```

2. Set Groups permissions:

```bash
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

3. Verify Installation:

Verify that the current user is added to the render and video groups:

```bash
groups
```

Check if amdgpu kernel driver is installed:

```bash
dkms status
```

Check if the GPU is listed as an agent:

```bash
rocminfo
```

Check `rocm-smi`:

```bash
rocm-smi -i --showmeminfo vram --showpower --showserial --showuse --showtemp --showproductname --showuniqueid --json
```

### Run GPUStack

Run the following command to start the GPUStack server **and built-in worker**:

=== "Host Network"

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device=/dev/kfd \
        --device=/dev/dri \
        --network=host \
        --ipc=host \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-rocm
    ```

=== "Port Mapping"

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device=/dev/kfd \
        --device=/dev/dri \
        -p 80:80 \
        -p 10150:10150 \
        -p 40064-40131:40064-40131 \
        --ipc=host \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-rocm \
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
        --device=/dev/kfd \
        --device=/dev/dri \
        --network=host \
        --ipc=host \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-rocm \
        --server-url http://your_gpustack_url --token your_gpustack_token
    ```

=== "Port Mapping"

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device=/dev/kfd \
        --device=/dev/dri \
        -p 10150:10150 \
        -p 40064-40131:40064-40131 \
        --ipc=host \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-rocm \
        --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_worker_host_ip
    ```

!!! note

    1. **Heterogeneous cluster is supported.** No matter what type of device it is, you can add it to the current GPUStack as a worker by specifying the `--server-url` and `--token` parameters.

    2. You can set additional flags for the `gpustack start` command by appending them to the docker run command.
    For configuration details, please refer to the [CLI Reference](../../cli-reference/start.md).

    3. You can either use the `--ipc=host` flag or `--shm-size` flag to allow the container to access the host’s shared memory. It is used by vLLM and pyTorch to share data between processes under the hood, particularly for tensor parallel inference.

    4. The  `-p 40064-40131:40064-40131` flag is used to ensure connectivity for distributed inference across workers. For more details, please refer to the [Port Requirements](../installation-requirements.md#port-requirements). You can omit this flag if you don't need distributed inference across workers.

## Installation Script

#### Supported Devices

=== "Linux"

    | Devices                                             | Supported Backends |
    | --------------------------------------------------- | ------------------ |
    | gfx1100: AMD Radeon RX 7900 XTX/7900 XT/7900 GRE    | llama-box          |
    | gfx1101: AMD Radeon RX 7800 XT/7700 XT              | llama-box          |
    | gfx1102: AMD Radeon RX 7600 XT/7600                 | llama-box          |
    | gfx1030: AMD Radeon RX 6950 XT/6900 XT/6800 XT/6800 | llama-box          |
    | gfx1031: AMD Radeon RX 6750 XT/6700 XT/6700         | llama-box          |
    | gfx1032: AMD Radeon RX 6650 XT/6600 XT/6600         | llama-box          |
    | gfx942: AMD Instinct MI300X/MI300A                  | llama-box          |
    | gfx90a: AMD Instinct MI250X/MI250/MI210             | llama-box          |
    | gfx908: AMD Instinct MI100                          | llama-box          |
    | gfx906: AMD Instinct MI50                           | llama-box          |

    View more details [here](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.4/reference/system-requirements.html).

=== "Windows"

    | Devices                                             | Supported Backends |
    | --------------------------------------------------- | ------------------ |
    | gfx1100: AMD Radeon RX 7900 XTX/7900 XT             | llama-box          |
    | gfx1101: AMD Radeon RX 7800 XT/7700 XT              | llama-box          |
    | gfx1102: AMD Radeon RX 7600 XT/7600                 | llama-box          |
    | gfx1030: AMD Radeon RX 6950 XT/6900 XT/6800 XT/6800 | llama-box          |
    | gfx1031: AMD Radeon RX 6750 XT/6700 XT/6700         | llama-box          |
    | gfx1032: AMD Radeon RX 6650 XT/6600 XT/6600         | llama-box          |

    View more details [here](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-6.2.4/reference/system-requirements.html).

#### Prerequisites

=== "Linux"

    - [ROCm 6.2.4](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.4/install/install-overview.html)

    1. Install ROCm 6.2.4:

    ```bash
    sudo apt update
    wget https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/jammy/amdgpu-install_6.2.60204-1_all.deb
    sudo apt install ./amdgpu-install_6.2.60204-1_all.deb
    amdgpu-install -y --usecase=graphics,rocm
    sudo reboot
    ```

    2. Set Groups permissions:

    ```bash
    sudo usermod -a -G render,video $LOGNAME
    sudo reboot
    ```

    3. Verify Installation:

    Verify that the current user is added to the render and video groups:

    ```bash
    groups
    ```

    Check if amdgpu kernel driver is installed:

    ```bash
    dkms status
    ```

    Check if the GPU is listed as an agent:

    ```bash
    rocminfo
    ```

    Check `rocm-smi`:

    ```bash
    rocm-smi -i --showmeminfo vram --showpower --showserial --showuse --showtemp --showproductname --showuniqueid --json
    ```

=== "Windows"

    - [HIP SDK 6.2.4](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-6.2.4/how-to/install.html)

    1. Type the following command on your system from PowerShell to confirm that the obtained information matches that listed in [Supported SKUs](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-6.2.4/reference/system-requirements.html#supported-skus-win):

    ```powershell
    Get-ComputerInfo | Format-Table CsSystemType,OSName,OSDisplayVersion
    ```

    2. Download the installer from the [HIP-SDK download page](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html).

    3. Launch the installer.

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
