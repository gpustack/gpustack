---
translation_status: translated
---

# 常见问题 { #faq }

## 支持矩阵 { #support-matrix }

### 异构集群支持 { #hybrid-cluster-support }

GPUStack 支持由 NVIDIA、AMD、昇腾 NPU、海光 DCU、摩尔线程、天数智芯、沐曦、寒武纪和平头哥 PPU 组成的异构集群，同时支持 AMD64 和 ARM64 架构。

### 分布式推理支持 { #distributed-inference-support }

**单节点多 GPU**

- [x] vLLM
- [x] SGLang
- [x] MindIE

**多节点多 GPU**

- [x] vLLM
- [x] SGLang
- [x] MindIE

!!! tip

    相关文档：

    **vLLM**：[分布式推理与服务](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)

    **SGLang**：[多节点部署](https://docs.sglang.io/references/multi_node_deployment/multi_node_index.html)

    **MindIE**：[多节点推理](https://www.hiascend.com/document/detail/zh/mindie/20RC2/envdeployment/instg/mindie_instg_0027.html)

## 安装 { #installation }

### 如何修改注册的 Worker 名称？ { #how-can-i-change-the-registered-worker-name }

运行 GPUStack 时，可通过 `--worker-name` 参数指定自定义名称：
```diff
sudo docker run -d --name gpustack \
    ...
    gpustack/gpustack \
+    --worker-name New-Name
```

### 如何修改注册的 Worker IP？ { #how-can-i-change-the-registered-worker-ip }

运行 GPUStack 时，可通过 `--worker-ip` 参数指定自定义 IP：
```diff
sudo docker run -d --name gpustack \
    ...
    gpustack/gpustack \
+    --worker-ip xx.xx.xx.xx
```

### GPUStack 的数据存储在哪里？ { #where-are-gpustacks-data-stored }

运行 GPUStack 容器时，通过 `--volume/-v` 参数挂载 Docker 卷。默认数据路径位于 Docker 数据目录下的 volumes 子目录中，具体路径为：
```bash
/var/lib/docker/volumes/gpustack-data/_data
```

可通过以下方式查看：
```bash
docker volume ls
docker volume inspect gpustack-data
```

如果需要使用自定义路径，请在运行容器时修改挂载配置。例如，挂载宿主机目录 `/data/gpustack`：
```diff
sudo docker run -d --name gpustack \
    ...
    --volume /var/run/docker.sock:/var/run/docker.sock \
-    --volume gpustack-data:/var/lib/gpustack \
+    --volume /data/gpustack:/var/lib/gpustack \
    ...
    gpustack/gpustack
```

### 模型文件存储在哪里？ { #where-are-model-files-stored }

运行 GPUStack 容器时，通过 `--volume/-v` 参数挂载 Docker 卷。默认缓存路径位于 Docker 数据目录下的 volumes 子目录中，具体路径为：
```bash
/var/lib/docker/volumes/gpustack-data/_data/cache
```

可通过以下方式查看：
```bash
docker volume ls
docker volume inspect gpustack-data
```

如果需要使用自定义路径，请在运行容器时修改挂载配置。

例如，挂载宿主机目录 `/data/model-cache`：
```diff
sudo docker run -d --name gpustack \
    ...
    --volume gpustack-data:/var/lib/gpustack \
+    --volume /data/model-cache:/var/lib/gpustack/cache \
    ...
    gpustack/gpustack
```

---

## 模型管理 { #managing-models }

### 如何从 Hugging Face 部署模型？ { #how-can-i-deploy-the-model-from-hugging-face }

从 Hugging Face 部署模型时，Server 节点和模型实例调度到的 Worker 节点都必须能够访问 Hugging Face，也可以使用镜像站。

例如，配置 `hf-mirror.com` 镜像站：
```diff
sudo docker run -d --name gpustack \
+    -e HF_ENDPOINT=https://hf-mirror.com \
    ...
    gpustack/gpustack
```

---

### 如何从本地路径部署模型？ { #how-can-i-deploy-the-model-from-local-path }

从本地路径部署模型时，确保目标 Worker 能够访问模型路径。

如果模型存储在特定 Worker 上，可使用 Worker 选择器将模型部署到该 Worker。

也可以在多个节点上挂载共享存储。

模型文件也必须挂载到容器内，且宿主机目录路径与容器内挂载路径必须相同。

从本地路径部署 Safetensors 模型时，路径**必须指向模型目录的绝对路径，该目录包含 `*.safetensors`、`config.json` 等文件**。

从本地路径部署 GGUF 模型时，路径**必须指向 `.gguf` 文件的绝对路径**。对于分片模型文件，使用第一个 `.gguf` 文件（00001）的绝对路径。

---

### 模型一直处于 `Pending` 状态怎么办？ { #what-should-i-do-if-the-model-is-stuck-in-pending-state }

`Pending` 表示当前没有 Worker 满足模型要求。将鼠标悬停在 `Pending` 状态上可查看原因。

首先检查 `Workers` 页面，确保 Worker 状态为 Ready。

之后根据各后端的处理逻辑排查。以 vLLM 为例：

vLLM 默认要求所有 GPU 的可用显存超过 90%（由 `--gpu-memory-utilization` 参数控制）。请确保可分配显存超过 90%。注意，即使其他模型处于 `Error` 或 `Downloading` 状态，其显存也已被分配。

如果所有 GPU 的可用显存均超过 90%，但模型仍显示 `Pending`，则说明显存总量不足。对于 `safetensors` 模型，所需显存（GB）可按以下方式估算：
```
GPU Memory (GB) = Model weight size (GB) * 1.2 + 2
```

如果可分配显存低于 90%，但确定模型可以使用更少显存运行，可调整 `--gpu-memory-utilization` 参数。例如，在 `Edit Model` → `Advanced` → `Backend Parameters` 中添加 `--gpu-memory-utilization=0.5`，分配 50% 的 GPU 显存。

**注意**：如果模型运行后报错，且日志显示 `CUDA: out of memory`，则说明分配的显存不足。需要进一步调整 `--gpu-memory-utilization`、增加资源，或部署更小的模型。

模型的上下文长度也会影响显存需求。可通过 `--max-model-len` 参数设置更短的上下文。在 GPUStack 中，未设置此参数时，默认值为 8192；如果在后端参数中指定，则使用实际设置值。

可根据需要调小上下文，例如 `--max-model-len=2048`。但每次推理请求的最大 tokens 数不能超过 `--max-model-len` 的值，因此上下文设置过小可能导致推理截断。

`--enforce-eager` 参数也有助于减少显存占用。不过，在 vLLM 中，该参数会强制模型使用即时执行模式：操作在调用时立即执行，而不是像基于计算图的执行方式（例如延迟执行）那样推迟以便优化。这可能使执行变慢，但更容易调试；由于缺少计算图执行提供的优化，也可能降低性能。

**SGLang** 和 **MindIE** 的处理流程类似，仅参数不同。更多信息请参阅[内置推理后端](user-guide/built-in-inference-backends.md)。

---

### 模型一直处于 `Scheduled` 状态怎么办？ { #what-should-i-do-if-the-model-is-stuck-in-scheduled-state }

尝试重启模型调度到的 GPUStack 容器。如果问题仍然存在，请查看 [Worker 日志](troubleshooting.md#view-gpustack-logs)并分析原因。

### 模型一直处于 `Error` 状态怎么办？ { #what-should-i-do-if-the-model-is-stuck-in-error-state }

将鼠标悬停在 `Error` 状态上查看原因。如果有 `View More` 按钮，点击后查看模型日志中的错误信息并分析原因。

### 基于官方 vLLM 镜像的自定义后端版本使用 `PYPI_PACKAGES_INSTALL` 后，为什么模型无法启动？ { #why-does-the-model-fail-to-start-when-using-a-custom-backend-version-based-on-the-official-vllm-image-with-pypi_packages_install }

使用基于官方 vLLM 镜像的自定义后端版本部署模型时，如果通过 `PYPI_PACKAGES_INSTALL` 安装额外 Python 包，模型可能无法启动。

可能出现类似以下错误：
```bash
/gpustack-command-xxxxxxxx: 40: /path/to/your_model: Permission denied
```

原因是容器通过自定义命令启动，而不是使用镜像的默认入口点。

在 `Inference Backends` → `vLLM` → `Edit` → `Versions Config` 中，编辑相应版本，将 Override Image Entrypoint 设置为：
```bash
vllm serve
```

### 为什么删除模型后没有释放磁盘空间？ { #why-doesnt-deleting-a-model-free-up-disk-space }

这是为了避免再次部署时重新下载模型。需要在 `Model Files` 中手动清理。

---

## 使用模型 { #using-models }

### 如何解决错误：At most 1 image(s) may be provided in one request？ { #how-can-i-resolve-the-error-at-most-1-images-may-be-provided-in-one-request }

这是 vLLM 的限制。可根据需要，在 `Edit Model → Advanced → Backend Parameters` 中调整 `--limit-mm-per-prompt` 参数。例如，`--limit-mm-per-prompt={"image": 4}` 表示每次推理请求最多支持 4 张图片，详情请参阅[相关说明](https://docs.vllm.ai/en/latest/configuration/engine_args/#-limit-mm-per-prompt)。

---

## 管理 GPUStack { #managing-gpustack }

### 如何通过代理使用 GPUStack？ { #how-do-i-use-gpustack-behind-a-proxy }

建议在运行 GPUStack 时传入标准代理环境变量。

以下示例展示如何将所有请求转发至目标代理，但 NO_PROXY 环境变量中指定的地址除外。
```bash
docker run -d --name gpustack \
    -e HTTPS_PROXY="http://proxy-server:port" \
    -e HTTP_PROXY="http://proxy-server:port" \
    -e NO_PROXY="127.0.0.1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/16,localhost,cluster.local" \
    ...
```

!!! note

    - 请根据实际情况替换 IP 地址或代理地址。
    - 如果代理需要身份认证，请使用 `http://username:password@proxy-server:port` 格式。密码中的特殊字符可能需要进行 URL 编码。
