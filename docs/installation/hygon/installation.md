# Hygon Installation (Experimental)

## Supported

- **Target Devices**
  - [x] Hygon DCUs (K100_AI (Verified), Z100/Z100L/K100(Not Verified))
- **Operating Systems**
  - [x] Linux AMD64
- **Available Inference Backends**
  - [x] [vLLM](https://github.com/vllm-project/vllm)
  - [x] Custom Engines

!!! note

    1. Whether a target device can run a specific inference backend depends on whether the corresponding version of the inference backend (container image) provides support for that device.
       Please verify compatibility with your target devices.

    2. Default container images, such as vLLM, are provided by the [GPUStack runner](https://github.com/gpustack/runner?tab=readme-ov-file#hygon-dtk).

## Prerequisites

### Hygon DCU Driver and DTK Toolkit

Ensure your system has an [Hygon DCU Driver](https://developer.sourcefind.cn/tool/) that supports Hygon DTK 25.04 or higher. Verify installation with:

```bash
sudo hy-smi

```

### Container Running Environment

It is recommended to use [Docker](https://docs.docker.com/engine/install/).

### Port Requirements

Ensure that each node meets [the port requirements](../requirements.md#port-requirements).

## Installation

Run the following command to start the GPUStack server **with the built-in worker**:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --volume /opt/hyhal:/opt/hyhal:ro \
    --volume /opt/dtk:/opt/dtk:ro \
    --env ROCM_PATH=/opt/dtk \
    --env ROCM_SMI_LIB_PATH=/opt/hyhal/lib \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

- To restrict DCU access, remove `--privileged` flag and set `--device` options accordingly.
- If the `/opt/hyhal` directory does not exist, please create a symbolic link pointing to the Hygon installed path: `ln -s /path/to/hyhal /opt/hyhal`.
  Same as `/opt/dtk` directory.
- If failed to detect devices, please try to remove `--env ROCM_SMI_LIB_PATH=/opt/hyhal/lib`.
- The `--network=host` option is necessary for port awareness.
- Mounting `/var/run/docker.sock` allows GPUStack to manage Docker containers for inference engines.

### Reusing Model Files

You can reuse model files stored on the host in two ways.

#### Bind Mount (Recommended)

Mount pre-downloaded model files into the container so they can be deployed from a local path without re-downloading:

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/model_files:/path/to/model_files:ro \
     gpustack/gpustack

```

#### Override Cache Directory

Mount a dedicated directory for storing downloaded models rather than relying on the default Docker volume:

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/model_files:/var/lib/gpustack/cache \
     gpustack/gpustack

```

### Customizing Serving Port

By default, GPUStack listens on port 80. You can change this with the `--port` parameter:

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
     gpustack/gpustack \
+    --port 9090

```

For more options or to resolve port conflicts, refer to the [CLI Reference](../../cli-reference/start.md).

## Startup

Check the GPUStack container logs:

```bash
sudo docker logs -f gpustack
```

If everything is normal, open `http://your_host_ip` in a browser to access the GPUStack UI.

Log in with username `admin` and the default password. Retrieve the initial password with:

```bash
sudo docker exec -it gpustack \
    cat /var/lib/gpustack/initial_admin_password
```

## (Optional) Add Worker

You can add more nodes to GPUStack to form a cluster.

Please navigate to the **Workers** page in the GPUStack UI to get the command for adding workers.
