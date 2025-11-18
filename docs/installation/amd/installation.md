# AMD Installation

## Supported

- **Target Devices**
    + [x] AMD GPUs
- **Operating Systems**
    + [x] Linux AMD64
    + [x] Linux ARM64
- **Available Inference Backends**
    + [x] vLLM
    + [x] SGLang
    + [x] Custom Engines

!!! note

    Whether a target device can run a specific inference engine depends on whether the corresponding version of the inference engine (container image) provides support for that device.

## Prerequisites

### AMD GPU Driver

Ensure your system has an [AMD GPU Driver](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) that supports AMD ROCm 6.4 or higher. Verify installation with:

```bash
sudo amd-smi static

```

### Container Running Environment

It is recommended to use [Docker](https://docs.docker.com/engine/install/) with the [AMD Container Runtime](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/overview.html):

```bash
sudo docker info 2>/dev/null | grep -q "amd" \
    && echo "AMD Container Toolkit OK" \
    || (echo "AMD Container Toolkit not configured"; exit 1)

```

### Port Requirements

Ensure that each node meets [the port requirements](../requirements.md#port-requirements).

## Installation

Run the following command to start the GPUStack server **with the built-in worker**:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --runtime amd \
    gpustack/gpustack
```

- The `--privileged` flag is required for device vendor-agnostic access.
- To restrict GPU access, remove `--privileged` flag and set the `AMD_VISIBLE_DEVICES` environment variable. 
  See [AMD Container Runtime - Migration Guide](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/migration-guide.html).
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
     --runtime amd \
     ...

```

#### Override Cache Directory

Mount your model directory to the container’s cache path:

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/model_files:/var/lib/gpustack/cache \
     --runtime amd \
     ...

```

### Customizing Serving Port

By default, GPUStack listens on port 80. You can change this with the `--port` parameter:

```diff
 sudo docker run -d --name gpustack \
     ...
     --runtime amd \
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
