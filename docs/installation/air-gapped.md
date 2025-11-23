# Air-Gapped Installation

You can install GPUStack in an air-gapped environment, which means setting up GPUStack offline without internet access.

## Prerequisites

### Driver

Ensure your system has the appropriate GPU drivers installed for your hardware.

See the [Installation Requirements](../installation/requirements.md) for details on driver compatibility.

### Container Running Environment

It is recommended to use [Docker](https://docs.docker.com/engine/install/).

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

- **Save Images**

## Installation

After preparing the internal container registry with the required images, you can install GPUStack in the air-gapped environment.

For example, to install with NVIDIA and start the GPUStack server **with the built-in worker**, run:

```diff
 sudo docker run -d --name gpustack \
     --restart unless-stopped \
     --privileged \
     --network host \
     --volume /var/run/docker.sock:/var/run/docker.sock \
     --volume gpustack-data:/var/lib/gpustack \
     --runtime nvidia \
-    gpustack/gpustack
+    <your_internal_registry>/gpustack/gpustack \
+    --system-default-container-registry <your_internal_registry>

```

If your accelerator is not NVIDIA, adjust the startup command accordingly.
