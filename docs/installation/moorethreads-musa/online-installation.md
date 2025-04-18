# Online Installation

## Supported Devices

- [x] Moore Threads GPUs (MTT S80, MTT S3000, MTT S4000)

## Supported Platforms

| OS    | Arch  | Supported methods                                                                                        |
| ----- | ----- | -------------------------------------------------------------------------------------------------------- |
| Linux | AMD64 | [Docker Installation](#docker-installation) (Recommended)<br>[Installation Script](#installation-script) |

## Supported backends

- [x] llama-box

## Prerequisites

- [Port Requirements](../installation-requirements.md#port-requirements)
- CPU support for llama-box backend: AMD64 with AVX2

Check if the CPU is supported:

```bash
lscpu | grep avx2
```

- [Driver for MTT S80/S3000/S4000](https://developer.mthreads.com/sdk/download/musa)

Check if the driver is installed:

```bash
mthreads-gmi
```

## Docker Installation

- [Docker](https://docs.docker.com/engine/install/)
- [MT Container Toolkits](https://developer.mthreads.com/sdk/download/CloudNative)

Check if the MT Container Toolkits are installed and set as the default runtime:

```bash
# cd /usr/bin/musa && sudo ./docker setup $PWD
docker info | grep Runtimes | grep mthreads
```

### Run GPUStack

Run the following command to start the GPUStack server **and built-in worker**:

=== "Host Network"

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-musa
    ```

=== "Port Mapping"

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        -p 80:80 \
        -p 10150:10150 \
        -p 40064-40131:40064-40131 \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-musa \
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
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-musa \
        --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_worker_host_ip
    ```

=== "Port Mapping"

    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        -p 10150:10150 \
        -p 40064-40131:40064-40131 \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-musa \
        --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_worker_host_ip
    ```

!!! note

    1. **Heterogeneous cluster is supported.** No matter what type of device it is, you can add it to the current GPUStack as a worker by specifying the `--server-url` and `--token` parameters.

    2. You can set additional flags for the `gpustack start` command by appending them to the docker run command.
    For configuration details, please refer to the [CLI Reference](../../cli-reference/start.md).

    3. The  `-p 40064-40131:40064-40131` flag is used to ensure connectivity for distributed inference across workers. For more details, please refer to the [Port Requirements](../installation-requirements.md#port-requirements). You can omit this flag if you don't need distributed inference across workers.

## Installation Script

### Prerequites

- [MUSA SDK](https://developer.mthreads.com/sdk/download/musa)

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
