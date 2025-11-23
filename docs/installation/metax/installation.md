# MetaX Installation (Experimental)

## Supported

- **Target Devices**
    + [x] MetaX GPUs
- **Operating Systems**
    + [x] Linux AMD64
- **Available Inference Backends**
    + [x] [vLLM](https://github.com/vllm-project/vllm)
    + [x] Custom Engines

!!! note

    1. Whether a target device can run a specific inference backend depends on whether the corresponding version of the inference backend (container image) provides support for that device. 
       Please verify compatibility with your target devices.
    2. Default container images, such as vLLM, are provided by the [GPUStack runner](https://github.com/gpustack/runner?tab=readme-ov-file#metax-maca).

## Prerequisites

### MetaX GPU Driver and MACA Toolkit

Ensure your system has an [MetaX GPU Driver](https://developer.metax-tech.com/softnova/download?package_kind=Driver) that supports MetaX MACA 3.0 or higher. Verify installation with:

```bash
sudo mx-smi

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
    --volume /opt/mxdriver:/opt/mxdriver:ro \
    --volume /opt/maca:/opt/maca:ro \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

- To restrict GPU access, remove `--privileged` flag and set `--device` options accordingly.
- If the `/opt/mxdriver` directory does not exist, please create a symbolic link pointing to the MetaX installed path: `ln -s /path/to/metax /opt/mxdriver`. 
  Same as `/opt/maca` directory.
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
     gpustack/gpustack

```

#### Override Cache Directory

Mount your model directory to the container’s cache path:

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

### (Optional) Add Worker

You can add more nodes to GPUStack to form a cluster. 

Please navigate to the **Workers** page in the GPUStack UI to get the command for adding workers.
