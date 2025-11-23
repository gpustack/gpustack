# MThreads Installation (Experimental)

## Supported

- **Target Devices**
    + [x] MThreads GPUs
- **Operating Systems**
    + [x] Linux AMD64
- **Available Inference Backends**
    + [x] Custom Engines

!!! note

    Whether a target device can run a specific inference backend depends on whether the corresponding version of the inference backend (container image) provides support for that device.
    Please verify compatibility with your target devices.

## Prerequisites

### MThreads GPU Driver

Ensure your system has an [MThreads GPU Driver](https://docs.mthreads.com/). Verify installation with:

```bash
sudo mthreads-gmi

```

### Container Running Environment

It is recommended to use [Docker](https://docs.docker.com/engine/install/) with the [KUAE Cloud Native Toolkits](https://developer.mthreads.com/sdk/download/CloudNative):

```bash
sudo docker info 2>/dev/null | grep -q "mthreads" \
    && echo "MThreads Container Toolkit OK" \
    || (echo "MThreads Container Toolkit not configured"; exit 1)

```

### Port Requirements

Ensure that each node meets [the port requirements](../requirements.md#port-requirements).

## Installation

Run the following command to start the GPUStack server **with the built-in worker** (host network mode is recommended):

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --runtime mthreads \
    gpustack/gpustack
```

- To restrict GPU access, remove `--privileged` flag and set the `MTHREADS_VISIBLE_DEVICES` environment variable. 
  See [MThreads Container Toolkit - GPU Enumeration](https://docs.mthreads.com/cloud-native/cloud-native-doc-online/start).
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
     --runtime mthreads \
     ..

```

#### Override Cache Directory

Mount your model directory to the container’s cache path:

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/model_files:/var/lib/gpustack/cache \
     --runtime mthreads \
     ...

```

### Customizing Serving Port

By default, GPUStack listens on port 80. You can change this with the `--port` parameter:

```diff
 sudo docker run -d --name gpustack \
     ...
     --runtime mthreads \
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
