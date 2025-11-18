# NVIDIA Installation

## Supported

- **Target Devices**
    + [x] NVIDIA GPUs (Compute Capability 7.5 and above, check [Your GPU Compute Capability](https://developer.nvidia.com/cuda-gpus))
- **Operating Systems**
    + [x] Linux AMD64
    + [x] Linux ARM64
- **Available Inference Backends**
    + [x] vLLM
    + [x] SGLang
    + [x] Vox-Box
    + [x] Custom Engines

!!! note

    Whether a target device can run a specific inference engine depends on whether the corresponding version of the inference engine (container image) provides support for that device.

## Prerequisites

### NVIDIA GPU Driver

Ensure your system has an [NVIDIA GPU Driver](https://www.nvidia.com/en-us/drivers/) that supports NVIDIA CUDA 12.4 or higher. Verify installation with:

```bash
sudo nvidia-smi

```

### Container Running Environment

It is recommended to use [Docker](https://docs.docker.com/engine/install/) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit):

```bash
sudo docker info 2>/dev/null | grep -q "nvidia" \
    && echo "NVIDIA Container Toolkit OK" \
    || (echo "NVIDIA Container Toolkit not configured"; exit 1)

```

### Configure CGroup Driver

Set Docker to use the native CGroup driver (cgroupfs):

```diff
vim /etc/docker/daemon.json
 {
   ...,
+  "exec-opts": [
+    "native.cgroupdriver=cgroupfs"
+  ],
   ...
 }
```

If this step is skipped, GPU devices may become inaccessible within containers after certain operations (e.g.,
`systemctl daemon-reload`), see [NVIDIA Container Toolkit #48](https://github.com/NVIDIA/nvidia-container-toolkit/issues/48) for details.

Remember to restart the Docker service to apply the above changes:

```bash
sudo systemctl daemon-reload \
    && sudo systemctl restart docker
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
    --runtime nvidia \
    gpustack/gpustack
```

- To restrict GPU access, remove `--privileged` flag and set the `NVIDIA_VISIBLE_DEVICES` environment variable. 
  See [NVIDIA Container Toolkit - GPU Enumeration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#gpu-enumeration).
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
     --runtime nvidia \
     ...

```

#### Override Cache Directory

Mount your model directory to the container’s cache path:

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/model_files:/var/lib/gpustack/cache \
     --runtime nvidia \
     ...

```

### Customizing Serving Port

By default, GPUStack listens on port 80. You can change this with the `--port` parameter:

```diff
 sudo docker run -d --name gpustack \
     ...
     --runtime nvidia \
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
