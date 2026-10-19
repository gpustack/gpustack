# 快速入门 { #quickstart }

本指南介绍如何在自托管的 GPU 服务器上运行 GPUStack。如需使用[云 GPU](./tutorials/adding-gpucluster-using-digitalocean.md)或集成[现有 Kubernetes 集群](./tutorials/adding-gpucluster-using-kubernetes.md)，请参阅相应教程。

!!! info "前提条件"

    1. 一个至少配备一块 NVIDIA GPU 的节点。对于其他 GPU 类型，请查看 GPUStack UI 中添加 Worker 时的指引，或参阅[安装文档](./installation/requirements.md)了解详细信息。
    2. 确保 Worker 节点已安装 NVIDIA 驱动、[Docker](https://docs.docker.com/engine/install/) 和 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。
    3. **（可选）** 一个用于运行 GPUStack Server 的 CPU 节点。GPUStack Server 不需要 GPU，可以在仅有 CPU 的机器上运行。该节点必须安装 [Docker](https://docs.docker.com/engine/install/)，也支持 Docker Desktop（适用于 Windows 和 macOS）。如果没有专用的 CPU 节点，可以将 GPUStack Server 安装在 GPU Worker 所在的机器上。
    4. GPUStack Worker 仅支持 Linux。如果使用 Windows，请考虑使用 WSL2，并避免使用 Docker Desktop。GPUStack Worker 不支持 macOS。

## 安装 GPUStack { #install-gpustack }

运行以下命令，使用 [Docker](https://docs.docker.com/engine/install/) 安装并启动 GPUStack Server：

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

??? Note "替代方案：使用 Quay 容器镜像仓库"

    如果无法从 Docker Hub 拉取镜像，或者下载速度很慢，可以将镜像仓库改为 `quay.io`，使用我们提供的 Quay 容器镜像仓库：

    ```bash
    sudo docker run -d --name gpustack \
        --restart unless-stopped \
        -p 80:80 \
        --volume gpustack-data:/var/lib/gpustack \
        quay.io/gpustack/gpustack \
        --system-default-container-registry quay.io
    ```

查看 GPUStack 启动日志：

```bash
sudo docker logs -f gpustack
```

GPUStack 启动后，运行以下命令获取默认管理员密码：

```bash
sudo docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

在浏览器中访问 `http://your_host_ip`，打开 GPUStack UI。使用默认用户名 `admin` 和刚获取的密码登录。

## 设置 GPU 集群 { #set-up-a-gpu-cluster }

1. 在 GPUStack UI 中，进入 `Clusters` 页面。

2. 单击 `Add Cluster` 按钮。

3. 选择 `Docker` 作为集群提供方。

4. 填写新集群的 `Name` 和 `Description` 字段，然后单击 `Save` 按钮。

5. 按照界面指引配置新的 Worker。您需要在 Worker 上运行 Docker 命令，将其连接到 GPUStack Server。命令类似如下：

```bash
sudo docker run -d --name gpustack-worker \
      --restart=unless-stopped \
      --privileged \
      --network=host \
      --volume /var/run/docker.sock:/var/run/docker.sock \
      --volume gpustack-data:/var/lib/gpustack \
      --runtime nvidia \
      gpustack/gpustack \
      --server-url http://your_gpustack_server_url \
      --token your_worker_token \
      --advertise-address 192.168.1.2
```

6. 在 Worker 上执行该命令，将其连接到 GPUStack Server。

7. Worker 连接成功后，会出现在 GPUStack UI 的 `Workers` 页面中。

## 部署模型 { #deploy-a-model }

1. 在 GPUStack UI 中，进入 `Catalog` 页面。

2. 从可用模型列表中选择 `Qwen3.5-0.8B` 模型。

3. 部署兼容性检查通过后，单击 `Save` 按钮部署模型。

![从模型目录部署 Qwen3](assets/quick-start/quick-start-qwen3.png)

4. GPUStack 将开始下载模型文件并部署模型。当部署状态显示为 `Running` 时，表示模型已成功部署。

!!! note

    GPUStack 使用容器运行模型。首次部署模型时，下载模型文件和容器镜像可能需要一些时间。可以在界面中单击 `View Logs`，查看部署进度。

![模型正在运行](assets/quick-start/model-running.png)

5. 在导航菜单中单击 `Playground - Chat`，确认右上角的 `Model` 下拉框已选中模型 `qwen3.5-0.8b`。现在可以在界面的 Playground 中与模型对话。

![快速对话](assets/quick-start/quick-chat.png)

## 通过 API 使用模型 { #use-the-model-via-api }

1. 进入 `Access Control` > `API Keys` 页面，然后单击 `New API Key` 按钮。

2. 填写 `Name`，然后单击 `Save` 按钮。

3. 复制生成的 API 密钥，并将其保存在安全的位置。请注意，密钥只会在创建时显示一次。

4. 现在可以使用 API 密钥访问 GPUStack 提供的 OpenAI 兼容 API 端点。例如，使用以下 curl 命令：

```bash
# 将 `your_api_key` 和 `your_gpustack_server_url`
# 替换为实际的 API 密钥和 GPUStack Server URL。
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Tell me a joke."
      }
    ],
    "stream": true
  }'
```

## 清理 { #cleanup }

使用完已部署的模型后，可以进入 GPUStack UI 的 `Deployments` 页面删除模型，以释放资源。
