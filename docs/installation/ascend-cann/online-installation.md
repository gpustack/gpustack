# Online Installation

## Supported Devices

- [x] Ascend 910B series (910B1 ~ 910B4)
- [x] Ascend 310P3

## Supported Platforms

| OS    | Arch  | Supported methods                                                                                        |
| ----- | ----- | -------------------------------------------------------------------------------------------------------- |
| Linux | ARM64 | [Docker Installation](#docker-installation) (Recommended)<br>[Installation Script](#installation-script) |

## Prerequisites

- [Port Requirements](../installation-requirements.md#port-requirements)
- CPU support for llama-box backend: ARM64 with NEON

Check if the CPU is supported:

```bash
grep -E -i "neon|asimd" /proc/cpuinfo
```

- [NPU Driver and Firmware](https://www.hiascend.com/hardware/firmware-drivers/community) (Must support CANN 8.1.RC1.beta1)

Check if the NPU driver is installed:

```bash
npu-smi info
```

## Docker Installation

### Supported backends

- [x] llama-box (Only supports FP16 precision)
- [x] MindIE
- [x] vLLM (Only supports Ascend 910B series)

### Prerequisites

- [Docker](https://docs.docker.com/engine/install/)

### Run GPUStack

Run the following command to start the GPUStack server **and built-in worker** (host network mode is recommended). Set `--device /dev/davinci{index}` to the required GPU indices:

=== "Ascend 910B"

    Follow the steps below to install GPUStack on the Ascend 910B:

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-npu
    ```

    If you need to change the default server port 80, please use the `--port` parameter:

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-npu \
        --port 9090
    ```

=== "Ascend 310P"

    Follow the steps below to install GPUStack on the Ascend 310P:

    === "Host Network"

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-npu-310p
    ```

    If you need to change the default server port 80, please use the `--port` parameter:

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-npu-310p \
        --port 9090
    ```

If other ports are in conflict, or if you want to customize startup options, refer to the [CLI Reference](../../cli-reference/start.md) for available flags and configuration instructions.

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

To start GPUStack as a worker, and **register it with the GPUStack server** (Set `--device /dev/davinci{index}` to the required GPU indices), run the following command on the **worker node**. Be sure to replace the URL and token with your specific values:

=== "Ascend 910B"

    Follow the steps below to add workers on the Ascend 910B:

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-npu \
        --server-url http://your_gpustack_url --token your_gpustack_token
    ```

=== "Ascend 310P"

    Follow the steps below to add workers on the Ascend 310P:

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-npu-310p \
        --server-url http://your_gpustack_url --token your_gpustack_token
    ```

!!! note

    1. **Heterogeneous cluster is supported.** No matter what type of device it is, you can add it to the current GPUStack as a worker by specifying the `--server-url` and `--token` parameters.

    2. You can set additional flags for the `gpustack start` command by appending them to the docker run command.
    For configuration details, please refer to the [CLI Reference](../../cli-reference/start.md).

    3. You can either use the `--ipc=host` flag or `--shm-size` flag to allow the container to access the host’s shared memory. It is used by vLLM and pyTorch to share data between processes under the hood, particularly for tensor parallel inference.

## Installation Script(Deprecated)

!!! note
      The installation script method is deprecated as of version 0.7.

### Supported backends

- [x] llama-box (Only supports Ascend 910B and FP16 precision)

### Prerequites

- [Ascend CANN Toolkit 8.1.RC1.beta1 & Kernels](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.0.beta1)

Check if CANN is installed and verify that its version is 8.1.RC1:

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

Check if CANN kernels are installed and verify that its version is 8.1.RC1:

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg | grep opp
```

### Run GPUStack

GPUStack provides a script to install it as a service with default port 80.

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

### (Optional) Add Worker

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
