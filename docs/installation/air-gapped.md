# Air-Gapped Installation

GPUStack can be installed in an air-gapped (offline) environment with no internet access.

## Prerequisites

### Driver

Ensure your system has the appropriate GPU drivers installed for your hardware.

See the [Installation Requirements](../installation/requirements.md) for details on driver compatibility.

### Container Running Environment

[Docker](https://docs.docker.com/engine/install/) must be installed.

If your system supports a container toolkit, install and configure it as needed (e.g., NVIDIA Container Toolkit, AMD ROCm Container Toolkit, etc.).

### Container Images

- **Copy Images**

GPUStack provides various container images for different inference backends, available on [Docker Hub](https://hub.docker.com/r/gpustack/runner).

To transfer the required container images to your internal registry from a machine with internet access, use the GPUStack `copy-images` command:

```bash
sudo docker run --rm -it --entrypoint "" gpustack/gpustack \
    gpustack copy-images \
    --destination <your_internal_registry> \
    --destination-username <your_username> \
    --destination-password <your_password>
```

If you cannot pull images from `Docker Hub` or the download is very slow, you can use our `Quay.io` mirror by pointing the source registry to `quay.io`:

```bash
sudo docker run --rm -it --entrypoint "" gpustack/gpustack \
    gpustack copy-images \
    --source quay.io \
    --destination <your_internal_registry> \
    --destination-username <your_username> \
    --destination-password <your_password>
```

For more details on `copy-images`, refer to the [CLI Reference](../cli-reference/copy-images.md).

- **List Images**

If you cannot access your internal registry directly, you can first pull the `gpustack/gpustack` image and then use `list-images` command to see which images need to be downloaded:

```bash
sudo docker run --rm -it --entrypoint "" gpustack/gpustack \
    gpustack list-images
```

!!! note

    This uses the latest version by default. To target a specific version, use the full image tag, e.g., gpustack/gpustack:vx.y.z.

The displayed image list includes all supported accelerators, inference backends, versions, and architectures. If you only need a subset, see the [CLI Reference](../cli-reference/list-images.md) for filtering options.

## Installation

After preparing the internal container registry with the required images, you can install GPUStack in the air-gapped environment. Port 80 is the primary server endpoint, while port 10161 is used to expose metrics for observability.

```diff
 sudo docker run -d --name gpustack \
     --restart unless-stopped \
     -p 80:80 \
     -p 10161:10161 \
     --volume gpustack-data:/var/lib/gpustack \
-    gpustack/gpustack
+    <your_internal_registry>/gpustack/gpustack \
+    --system-default-container-registry <your_internal_registry>

```

### Pulling Inference Backend Images from a Secure Registry

If your internal container registry requires authentication,  
set the following environment variables when starting the GPUStack worker to allow it to pull the runner image.

```diff
 sudo docker run -d --name gpustack \
     ...
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME=<your_internal_registry_username> \
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD=<your_internal_registry_password> \
     <your_internal_registry>/gpustack/gpustack \
     --system-default-container-registry <your_internal_registry>

```

### Pulling Inference Backend Images from none default Namespace

If your internal container registry uses a different namespace than the default `gpustack`,  
set the following environment variable when starting the GPUStack worker to allow it to pull the runner image.

```diff
 sudo docker run -d --name gpustack \
     ...
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE=<your_namespace> \
     <your_internal_registry>/gpustack/gpustack \
     --system-default-container-registry <your_internal_registry>
```
