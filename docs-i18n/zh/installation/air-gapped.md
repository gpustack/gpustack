# 离线安装 { #air-gapped-installation }

GPUStack 支持在无法访问互联网的隔离（离线）环境中安装。

运行后，GPUStack 会自行提供本套文档，见下文[文档](#documentation)。

## 前提条件 { #prerequisites }

### 驱动 { #driver }

确保系统已安装适合硬件的 GPU 驱动。

驱动兼容性详情见[安装要求](./requirements.md)。

### 容器运行环境 { #container-running-environment }

必须安装 [Docker](https://docs.docker.com/engine/install/)。

如果系统支持容器工具包，请按需安装和配置，例如 NVIDIA Container Toolkit、AMD ROCm Container Toolkit 等。

### 容器镜像 { #container-images }

GPUStack 提供[镜像选择器](../image-selector/index.html)，方便用户选择要下载的镜像。它完全在浏览器中运行，使用随文档提供的数据，因此无需联网即可使用；其版本列表仅涵盖随文档发布时已有的版本。如需更高级或自动化的同步方式，GPUStack 还提供以下镜像管理命令：

- `gpustack copy-images`：将镜像从一个镜像仓库同步到另一个仓库
- `gpustack save-images`：下载镜像并保存到本地
- `gpustack load-images`：从本地镜像包导入镜像
- `gpustack list-images`：显示当前版本的镜像清单

以下介绍这些 CLI 命令的使用方式。

- **复制镜像**

GPUStack 为不同组件和推理后端提供多种容器镜像，可从 [Docker Hub](https://hub.docker.com/u/gpustack) 和 [Quay.io](https://quay.io/user/gpustack/) 获取。

在可访问互联网的机器上，使用 GPUStack 的 `copy-images` 命令将所需容器镜像转移到内部镜像仓库：

```bash
sudo docker run --rm -it --entrypoint "" gpustack/gpustack \
    gpustack copy-images \
    --destination <your_internal_registry> \
    --destination-username <your_username> \
    --destination-password <your_password>
```

如果无法从 `Docker Hub` 拉取镜像，或下载速度很慢，可以将源仓库指向 `quay.io`，使用我们的 `Quay.io` 镜像站：

```bash hl_lines="3"
sudo docker run --rm -it --entrypoint "" gpustack/gpustack \
    gpustack copy-images \
    --source quay.io \
    --destination <your_internal_registry> \
    --destination-username <your_username> \
    --destination-password <your_password>
```

!!! note

    默认使用最新版本。如需指定版本，请使用完整镜像标签，例如 `gpustack/gpustack:vx.y.z`。

`copy-images` 的详细信息见 [CLI 参考](../cli-reference/copy-images.md)。

- **列出镜像**

如果无法直接访问内部镜像仓库，可以先拉取 `gpustack/gpustack` 镜像，再使用 `list-images` 命令查看需要下载的镜像：

```bash
sudo docker run --rm -it --entrypoint "" \
    gpustack/gpustack \
    gpustack list-images
```

!!! note

    默认使用最新版本。如需指定版本，请使用完整镜像标签，例如 `gpustack/gpustack:vx.y.z`。

显示的镜像列表包含所有受支持的加速器、推理后端、版本和架构。如果只需要其中一部分，请参阅 [CLI 参考](../cli-reference/list-images.md)中的筛选选项。

- **复制和加载镜像**

如果目标环境为隔离环境或无法访问互联网，可以先在联网机器上下载所需镜像，再将其转移到离线环境并加载。

GPUStack 提供 `save-images` 和 `load-images` 命令来完成此流程。

**保存镜像**

在可访问互联网的机器上运行以下命令，下载并打包所需镜像：

```bash
sudo docker run --rm -it --entrypoint "" \
 -v ./gpustack-air-gapped:/gpustack-air-gapped \
 --workdir /gpustack-air-gapped \
 gpustack/gpustack \
 gpustack save-images \
 --platform linux/amd64 \
 --backend cuda \
 --backend-version 12.9 \
 --service vllm \
 --service-version 0.15.1 \
 --max-workers 3 \
 --max-retries 3
```

此命令根据指定的平台、后端和服务配置，下载所有所需容器镜像，并以本地镜像包形式保存到 `gpustack-air-gapped` 目录。示例命令展示了如何使用筛选选项，仅下载部分镜像。

!!! note

    默认使用最新版本。如需指定版本，请使用完整镜像标签，例如 `gpustack/gpustack:vx.y.z`。

可以调整筛选条件，只下载需要的镜像。完整可用选项见 [CLI 参考](../cli-reference/save-images.md)。

下载完成后，使用合适的方式将生成的目录转移到隔离环境，例如 USB 存储设备或内部文件传输。

**加载镜像**

在目标机器上，将已保存镜像加载到本地容器运行时：

```bash
sudo docker run --rm -it --entrypoint "" \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume ./gpustack-air-gapped:/gpustack-air-gapped \
    --workdir /gpustack-air-gapped \
    gpustack/gpustack \
    gpustack load-images \
    --platform linux/amd64 \
    --max-workers 3 \
    --max-retries 3 \
    /gpustack-air-gapped
```

此命令将指定目录中的所有镜像包导入本地 Docker daemon，供 GPUStack 使用。

!!! note

    默认使用最新版本。如需指定版本，请使用完整镜像标签，例如 `gpustack/gpustack:vx.y.z`。

`load-images` 的详细信息见 [CLI 参考](../cli-reference/load-images.md)。

## 安装 { #installation }

将所需镜像准备到内部容器镜像仓库后，即可在离线环境中安装 GPUStack。

```diff
 sudo docker run -d --name gpustack \
     --restart unless-stopped \
     -p 80:80 \
     --volume gpustack-data:/var/lib/gpustack \
-    gpustack/gpustack
+    <your_internal_registry>/gpustack/gpustack \
+    --system-default-container-registry <your_internal_registry>

```

### 从需要认证的镜像仓库拉取推理后端镜像 { #pulling-inference-backend-images-from-a-secure-registry }

如果内部容器镜像仓库需要身份认证，请在启动 GPUStack Worker 时设置以下环境变量，使其能够拉取运行镜像。

```diff
 sudo docker run -d --name gpustack \
     ...
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME=<your_internal_registry_username> \
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD=<your_internal_registry_password> \
     <your_internal_registry>/gpustack/gpustack \
     --system-default-container-registry <your_internal_registry>

```

### 从非默认命名空间拉取推理后端镜像 { #pulling-inference-backend-images-from-non-default-namespace }

如果内部容器镜像仓库使用的命名空间不是默认的 `gpustack`，请在启动 GPUStack Worker 时设置以下环境变量，使其能够拉取运行镜像。

```diff
 sudo docker run -d --name gpustack \
     ...
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE=<your_namespace> \
     <your_internal_registry>/gpustack/gpustack \
     --system-default-container-registry <your_internal_registry>
```

## 文档 { #documentation }

GPUStack Server 内置本套文档，并通过 `/help/` 提供访问，因此离线环境无需访问在线文档站点：

```
http://<your_gpustack_server>/help/
```

这些文档与在线站点的内容相同，版本对应构建 Server 镜像时使用的版本，搜索功能也可离线使用。
