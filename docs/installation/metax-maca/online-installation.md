# Online Installation

## Supported Devices

- [x] Metax GPUs (MetaX C500、 MetaX C550、 MetaX N260、etc.)

## Supported Platforms

| OS    | Arch  | Supported methods                           |
| ----- | ----- | ------------------------------------------- |
| Linux | AMD64 | [Docker Installation](#docker-installation) |

## Supported backends

- [x] vLLM

## Prerequisites

Install driver and sdk

- [metax-driver-3.0.0.5-deb-x86_64.run](https://sw-download.metax-tech.com/)
- [maca-sdk-3.0.0.8-deb-x86_64.tar.xz](https://sw-download.metax-tech.com/)

Check if the driver is installed:

```bash
mx-smi
```

## Docker Installation

- [Docker](https://docs.docker.com/engine/install/)

### Run GPUStack

Run the following command to start the GPUStack server **and built-in worker** (host network mode is recommended):

```bash
docker run -d --name gpustack \
    --device=/dev/dri \
    --device=/dev/mxcd \
    --privileged=true \
    --group-add video \
    --network=host \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --shm-size '100gb' --ulimit memlock=-1 \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-maca
```

If you need to change the default server port 80, please use the `--port` parameter:

```bash
docker run -d --name gpustack \
    --device=/dev/dri \
    --device=/dev/mxcd \
    --privileged=true \
    --group-add video \
    --network=host \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --shm-size '100gb' --ulimit memlock=-1 \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-maca
    --port 10000
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
    --device=/dev/dri \
    --device=/dev/mxcd \
    --privileged=true \
    --group-add video \
    --network=host \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --shm-size '100gb' --ulimit memlock=-1 \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-maca \
    --server-url http://your_gpustack_url --token your_gpustack_token
```

!!! note

    1. **Heterogeneous cluster is supported.** No matter what type of device it is, you can add it to the current GPUStack as a worker by specifying the `--server-url` and `--token` parameters.

    2. You can set additional flags for the `gpustack start` command by appending them to the docker run command.
    For configuration details, please refer to the [CLI Reference](../../cli-reference/start.md).