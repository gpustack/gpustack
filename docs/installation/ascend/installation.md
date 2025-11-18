# Ascend Installation

## Supported

- **Target Devices**
    + [x] Ascend NPU 910C series
    + [x] Ascend NPU 910B series (910B1 ~ 910B4)
    + [x] Ascend NPU 310P3
- **Operating Systems**
    + [x] Linux AMD64
    + [x] Linux ARM64
- **Available Inference Backends**
    + [x] vLLM
    + [x] SGLang
    + [x] MindIE
    + [x] Custom Engines

!!! note

    Whether a target device can run a specific inference engine depends on whether the corresponding version of the inference engine (container image) provides support for that device.

## Prerequisites

### Ascend NPU Driver

Ensure your system has an [Ascend NPU Driver](https://www.hiascend.com/hardware/firmware-drivers/community) that supports Ascend CANN 8.2 or higher. Verify installation with:

```bash
sudo npu-smi info

```

### Container Running Environment

It is recommended to use [Docker](https://docs.docker.com/engine/install/) with the [Ascend Docker Runtime](https://www.hiascend.com/document/detail/zh/mindcluster/72rc1/clustersched/dlug/dlug_installation_017.html):

```bash
sudo docker info 2>/dev/null | grep -q "ascend" \
    && echo "Ascend Container Toolkit OK" \
    || (echo "Ascend Container Toolkit not configured"; exit 1)

```

### Port Requirements

Ensure that each node meets [the port requirements](../requirements.md#port-requirements).

## Installation

Run the following command to start the GPUStack server **with the built-in worker**:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --env "ASCEND_VISIBLE_DEVICES=$(sudo ls /dev/davinci* | head -1 | grep -o '[0-9]\+' || echo "0")" \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --runtime ascend \
    gpustack/gpustack
```

- To restrict NPU access, remove `--privileged` flag and set the `ASCEND_VISIBLE_DEVICES` environment variable. 
  See [MindCluster - Docker Client Usage](https://www.hiascend.com/document/detail/zh/mindcluster/72rc1/clustersched/dlug/dlruntime_ug_004.html).
- The `--network=host` option is necessary for port awareness.
- Mounting `/var/run/docker.sock` allows GPUStack to manage Docker containers for inference engines.

### Reusing Model Files

You can reuse model files stored on the host in two ways.

#### Bind Mount (Recommended)

Avoid re-downloading model files inside the container:

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/model_files:/path/to/model_files:ro \
     --runtime ascend \
     ...

```

#### Override Cache Directory

Mount your model directory to the container’s cache path:

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/model_files:/var/lib/gpustack/cache \
     --runtime ascend \
     ...

```

### Customizing Serving Port

By default, GPUStack listens on port 80. You can change this with the `--port` parameter:

```diff
 sudo docker run -d --name gpustack \
     ...
     --runtime ascend \
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

### (Optional) Add Worker

You can add more nodes to GPUStack to form a cluster. 

Please navigate to the **Workers** page in the GPUStack UI to get the command for adding workers.
