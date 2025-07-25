# Online Installation

In GPUStack, `llama-box` and `vox-box` backends support CPU inference. However, compared to GPUs, CPU performance is significantly lower, so it is only recommended for testing or small-scale use cases.

## Supported Devices

- [x] CPUs (AMD64 with AVX or ARM64 with NEON)

## Supported Platforms

| OS      | Arch           | Supported methods                                                                                                                                                          |
| ------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux   | AMD64<br>ARM64 | [Docker Installation](#docker-installation) (Recommended)<br>[pip Installation](#pip-installation)<br>[Installation Script](#installation-script-deprecated) (Deprecated)  |
| Windows | AMD64<br>ARM64 | [Desktop Installer](../desktop-installer.md) (Recommended)<br>[pip Installation](#pip-installation)<br>[Installation Script](#installation-script-deprecated) (Deprecated) |

## Supported backends

- [x] llama-box
- [x] vox-box

## Prerequisites

- [Port Requirements](../installation-requirements.md#port-requirements)
- CPUs (AMD64 with AVX or ARM64 with NEON)

=== "Linux"

    Check if the CPU is supported:

    === "AMD64"

        ```bash
        lscpu | grep avx
        ```

    === "ARM64"

        ```bash
        grep -E -i "neon|asimd" /proc/cpuinfo
        ```

=== "Windows"

    Windows users need to manually verify support for the above instructions.

## Docker Installation

### Prerequisites

- [Docker](https://docs.docker.com/engine/install/)

### Run GPUStack

Run the following command to start the GPUStack server **and built-in worker** (host network mode is recommended):

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --network=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-cpu
```

If you need to change the default server port 80, please use the `--port` parameter:

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --network=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-cpu \
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

You can add more CPU nodes to GPUStack. You need to add workers on other CPU nodes and specify the `--server-url` and `--token` parameters to join GPUStack.

To get the token used for adding workers, run the following command on the GPUStack **server node**:

```bash
docker exec -it gpustack cat /var/lib/gpustack/token
```

To start GPUStack as a worker, and **register it with the GPUStack server**, run the following command on the worker node. Be sure to replace the URL and token with your specific values:

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --network=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-cpu \
    --server-url http://your_gpustack_url --token your_gpustack_token
```

!!! note

    1. **Heterogeneous cluster is supported.** No matter what type of device it is, you can add it to the current GPUStack as a worker by specifying the `--server-url` and `--token` parameters.

    2. You can set additional flags for the `gpustack start` command by appending them to the docker run command.
    For configuration details, please refer to the [CLI Reference](../../cli-reference/start.md).

## pip Installation

### Prerequisites

- Python 3.10 ~ 3.12

Check the Python version:

```bash
python -V
```

### Install GPUStack

Run the following to install GPUStack.

```bash
pip install "gpustack[audio]"
```

If you don’t need support for audio models, just run:

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
sudo tee /etc/systemd/system/gpustack.service > /dev/null <<EOF
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

## Installation Script (Deprecated)

!!! warning

    The installation script method is deprecated as of version 0.7. We recommend using Docker on Linux, and the [desktop installer](https://gpustack.ai/) on macOS or Windows.

GPUStack provides a script to install it as a service with default port 80.

=== "Linux"

    - Install Server

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

    - (Optional) Add Worker

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

    - Install Server

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

    - (Optional) Add Worker

    To add workers to the GPUStack cluster, you need to specify the server URL and authentication token when installing GPUStack on the workers.

    To get the token used for adding workers, run the following command on the GPUStack **server node**:

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\token" -Raw
    ```

    If you specify the `--data-dir` parameter to set the data directory, the `token` file will be located in the specified directory.

    To install GPUStack and start it as a worker, and **register it with the GPUStack server**, run the following command on the worker node. Be sure to replace the URL and token with your specific values:

    ```powershell
    Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --server-url http://your_gpustack_url --token your_gpustack_token"
    ```

    After installed, ensure that the GPUStack startup logs are normal:

    ```powershell
    Get-Content "$env:APPDATA\gpustack\log\gpustack.log" -Tail 200 -Wait
    ```
