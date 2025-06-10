# Online Installation

## Supported Devices

- [x] Iluvatar GPUs (MR-V100 MR-V50 BI-V100 BI-V150)

## Supported Platforms

| OS    | Arch  | Supported methods                           |
| ----- | ----- | ------------------------------------------- |
| Linux | AMD64 | [Docker Installation](#docker-installation) |

## Supported backends

- [x] vllm

## Prerequisites

- [Driver for MR-V100 MR-V50 BI-V100 BI-V150](https://support.iluvatar.com/#/ProductLine?id=2)

Check if the driver is installed:

```bash
ixsmi
```

## Docker Installation

- [Docker](https://support.iluvatar.com/#/ProductLine?id=2)
- [Corex Container Toolkits](https://support.iluvatar.com/#/ProductLine?id=2)

### Run GPUStack

Run the following command to start the GPUStack server **and built-in worker** (host network mode is recommended):

```bash
docker run -d --name gpustack \
    -v /lib/modules:/lib/modules \
    -v /dev:/dev \
    --privileged \
    --cap-add=ALL \
    --pid=host \
    --restart=unless-stopped \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-corex
```

If you need to change the default server port 80, please use the `--port` parameter:

```bash
docker run -d --name gpustack \
    -v /lib/modules:/lib/modules \
    -v /dev:/dev \
    --privileged \
    --cap-add=ALL \
    --pid=host \
    --restart=unless-stopped \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-corex \
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

To start GPUStack as a worker, and **register it with the GPUStack server**, run the following command on the **worker node**. Be sure to replace the URL and token with your specific values:

```bash
docker run -d --name gpustack \
    -v /lib/modules:/lib/modules \
    -v /dev:/dev \
    --privileged \
    --cap-add=ALL \
    --pid=host \
    --restart=unless-stopped \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-corex \
    --server-url http://your_gpustack_url --token your_gpustack_token
```

!!! note

    1. **Heterogeneous cluster is supported.** No matter what type of device it is, you can add it to the current GPUStack as a worker by specifying the `--server-url` and `--token` parameters.

    2. You can set additional flags for the `gpustack start` command by appending them to the docker run command.
    For configuration details, please refer to the [CLI Reference](../../cli-reference/start.md).
