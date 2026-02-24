# FAQ

## Support Matrix

### Hybird Cluster Support

GPUStack supports heterogeneous clusters spanning NVIDIA, AMD, Ascend, Hygon, Moore Threads, Iluvatar, MetaX, and Cambricon GPUs, and works across both AMD64 and ARM64 architectures.

### Distributed Inference Support

**Single-Node Multi-GPU**

- [x] vLLM
- [x] SGLang
- [x] MindIE

**Multi-Node Multi-GPU**

- [x] vLLM
- [x] SGLang
- [x] MindIE

!!! tip

    Related documentations:

    **vLLM**：[Distributed Inference and Serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)

    **SGLang**：[Multi-Node Deployment](https://docs.sglang.io/references/multi_node_deployment/multi_node_index.html)

    **MindIE**: [Multi-Node Inference](https://www.hiascend.com/document/detail/zh/mindie/20RC2/envdeployment/instg/mindie_instg_0027.html)

## Installation

### How can I change the registered worker name?

You can set it to a custom name using the `--worker-name` flag when running GPUStack:

```diff
sudo docker run -d --name gpustack \
    ...
    gpustack/gpustack \
+    --worker-name New-Name
```

### How can I change the registered worker IP?

You can set it to a custom IP using the `--worker-ip` flag when running GPUStack:

```diff
sudo docker run -d --name gpustack \
    ...
    gpustack/gpustack \
+    --worker-ip xx.xx.xx.xx
```

### Where are GPUStack's data stored?

When running the GPUStack container, the Docker volume is mounted using `--volume/-v` parameter. The default data path is under the Docker data directory, specifically in the volumes subdirectory, and the default path is:

```bash
/var/lib/docker/volumes/gpustack-data/_data
```

You can check it by the following method:

```bash
docker volume ls
docker volume inspect gpustack-data
```

If you need to change it to a custom path, modify the mount configuration when running container. For example, to mount the host directory `/data/gpustack`:

```diff
sudo docker run -d --name gpustack \
    ...
    --volume /var/run/docker.sock:/var/run/docker.sock \
-    --volume gpustack-data:/var/lib/gpustack \
+    --volume /data/gpustack:/var/lib/gpustack \
    ...
    gpustack/gpustack
```

### Where are model files stored?

When running the GPUStack container, the Docker volume is mounted using `--volume/-v` parameter. The default cache path is under the Docker data directory, specifically in the volumes subdirectory, and the default path is:

```bash
/var/lib/docker/volumes/gpustack-data/_data/cache
```

You can check it by the following method:

```bash
docker volume ls
docker volume inspect gpustack-data
```

If you need to change it to a custom path, modify the mount configuration when running container.

For example, to mount the host directory `/data/model-cache`:

```diff
sudo docker run -d --name gpustack \
    ...
    --volume gpustack-data:/var/lib/gpustack \
+    --volume /data/model-cache:/var/lib/gpustack/cache \
    ...
    gpustack/gpustack
```

---

## Managing Models

### How can I deploy the model from Hugging Face?

To deploy models from Hugging Face, the server node and the worker nodes where the model instances are scheduled must have access to Hugging Face, or you can use a mirror.

For example, configure the `hf-mirror.com` mirror:

```diff
sudo docker run -d --name gpustack \
+    -e HF_ENDPOINT=https://hf-mirror.com \
    ...
    gpustack/gpustack
```

---

### How can I deploy the model from Local Path?

When deploying models from Local Path, ensure the model path is accessible on the target workers.

If the model is stored on a specific worker, you can use the worker selector to deploy the model to that worker.

Another option is to mount a shared storage across multiple nodes.

And the model files must be mounted into the container, and the host directory and the container mount path must be identical.

When deploying Safetensors models from Local Path, the path **must point to the absolute path of the model directory which contain `*.safetensors`, `config.json`, and other files**.

When deploying GGUF models from Local Path, the path **must point to the absolute path of the `.gguf` file**. For sharded model files, use the absolute path of the first `.gguf` file (00001).

---

### What should I do if the model is stuck in `Pending` state?

`Pending` means that there are currently no workers meeting the model’s requirements, move the mouse over the `Pending` status to view the reason.

First, check the `Workers` section to ensure that the worker status is Ready.

Then, each backend has its own handling logic. For example, for vLLM:

vLLM requires that all GPUs have more than 90% of their memory available by default (controlled by the `--gpu-memory-utilization` parameter). Ensure that there is enough allocatable GPU memory exceeding 90%. Note that even if other models are in an `Error` or `Downloading` state, the GPU memory has already been allocated.

If all GPUs have more than 90% available memory but still show `Pending`, it indicates insufficient memory. For `safetensors` models, the required GPU memory (GB) can be estimated as:

```
GPU Memory (GB) = Model weight size (GB) * 1.2 + 2
```

If the allocatable GPU memory is less than 90%, but you are sure the model can run with a lower allocation, you can adjust the `--gpu-memory-utilization` parameter. For example, add `--gpu-memory-utilization=0.5` in `Edit Model` → `Advanced` → `Backend Parameters` to allocate 50% of the GPU memory.

**Note**: If the model encounters an error after running and the logs show `CUDA: out of memory`, it means the allocated GPU memory is insufficient. You will need to further adjust `--gpu-memory-utilization`, add more resources, or deploy a smaller model.

The context size for the model also affects the required GPU memory. You can adjust the `--max-model-len` parameter to set a smaller context. In GPUStack, if this parameter is not set, its default value is 8192. If it is specified in the backend parameters, the actual setting will take effect.

You can adjust it to a smaller context as needed, for example, `--max-model-len=2048`. However, keep in mind that the max tokens for each inference request cannot exceed the value of `--max-model-len`. Therefore, setting a very small context may cause inference truncation.

The `--enforce-eager` parameter also helps reduce GPU memory usage. However, this parameter in vLLM forces the model to execute in eager execution mode, meaning that operations are executed immediately as they are called, rather than being deferred for optimization in graph-based execution (like in lazy execution). This can make the execution slower but easier to debug. However, it can also reduce performance due to the lack of optimizations provided by graph execution.

**SGLang** and **MindIE** follow a similar process, differing only in their parameters. For more information, see the [Built-in Inference Backends](user-guide/built-in-inference-backends.md) section.

---

### What should I do if the model is stuck in `Scheduled` state?

Try restarting the GPUStack container where the model is scheduled. If the issue persists, check the worker logs [here](troubleshooting.md#view-gpustack-logs) to analyze the cause.

### What should I do if the model is stuck in `Error` state?

Move the mouse over the `Error` status to view the reason. If there is a `View More` button, click it to check the error messages in the model logs and analyze the cause of the error.

### Why does the model fail to start when using a custom backend version based on the official vLLM image with `PYPI_PACKAGES_INSTALL`?

When deploying a model using a custom vLLM backend version based on the official vLLM image, the model may fail to start if `PYPI_PACKAGES_INSTALL` is used to install additional Python packages.

You may see an error similar to:

```bash
/gpustack-command-xxxxxxxx: 40: /path/to/your_model: Permission denied
```

This happens because the container starts with a custom command instead of the image’s default entrypoint.

In `Inference Backends` → `vLLM` → `Edit` → `Versions Config`, edit the corresponding version and set Override Image Entrypoint to:

```bash
vllm serve
```

### Why doesn’t deleting a model free up disk space?

This is to avoid re-downloading the model when redeploying. You need to clean it up in `Model Files` manually.

---

## Using Models

### How can I resolve the error: At most 1 image(s) may be provided in one request?

This is a limitation of vLLM. You can adjust the `--limit-mm-per-prompt` parameter in `Edit Model → Advanced → Backend Parameters` as needed. For example, `--limit-mm-per-prompt={"image": 4}` means that it supports up to 4 images per inference request, see the details [here](https://docs.vllm.ai/en/latest/configuration/engine_args/#-limit-mm-per-prompt).

---

## Managing GPUStack

### How do I use GPUStack behind a proxy?

We recommend passing standard proxy environment variables when running GPUStack.

The following case demonstrates how to configure GPUStack to forward all requests to the target proxy, except for requests to addresses specified in the NO_PROXY environment variable.

```bash
docker run -d --name gpustack \
    -e HTTPS_PROXY="http://proxy-server:port" \
    -e HTTP_PROXY="http://proxy-server:port" \
    -e NO_PROXY="127.0.0.1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/16,localhost,cluster.local" \
    ...
```

!!! note

    - Replace the IP address/proxy address accordingly.
    - If your proxy requires authentication, use the format `http://username:password@proxy-server:port`. Be aware that special characters in passwords may need URL encoding.
