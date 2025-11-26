<br>

<p align="center">
    <img alt="GPUStack" src="https://raw.githubusercontent.com/gpustack/gpustack/main/docs/assets/gpustack-logo.png" width="300px"/>
</p>
<br>

<p align="center">
    <a href="https://docs.gpustack.ai" target="_blank">
        <img alt="Documentation" src="https://img.shields.io/badge/文档-GPUStack-blue?logo=readthedocs&logoColor=white"></a>
    <a href="./LICENSE" target="_blank">
        <img alt="License" src="https://img.shields.io/github/license/gpustack/gpustack?logo=github&logoColor=white&label=License&color=blue"></a>
    <a href="./docs/assets/wechat-group-qrcode.jpg" target="_blank">
        <img alt="WeChat" src="https://img.shields.io/badge/微信群-GPUStack-blue?logo=wechat&logoColor=white"></a>
    <a href="https://discord.gg/VXYJzuaqwD" target="_blank">
        <img alt="Discord" src="https://img.shields.io/badge/Discord-GPUStack-blue?logo=discord&logoColor=white"></a>
    <a href="https://twitter.com/intent/follow?screen_name=gpustack_ai" target="_blank">
        <img alt="Follow on X(Twitter)" src="https://img.shields.io/twitter/follow/gpustack_ai?logo=X"></a>
</p>
<br>

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a> |
  <a href="./README_JP.md">日本語</a>
</p>

<br>

## 概述

GPUStack 是一个开源的 GPU 集群管理器，专为高效的 AI 模型部署而设计。它允许您在自己的 GPU 硬件上高效运行模型，通过选择最佳推理引擎、调度 GPU 资源、分析模型架构以及自动配置部署参数来实现。

下图展示了 GPUStack 相较于未优化的 vLLM 基线如何提供更高的推理吞吐量：

![a100-throughput-comparison](docs/assets/a100-throughput-comparison.png)

有关详细的基准测试方法和结果，请访问我们的 [推理性能实验室](https://docs.gpustack.ai/latest/performance-lab/overview/)。

## 经过测试的推理引擎、GPU 和模型

GPUStack 采用插件式架构，可以轻松添加新的 AI 模型、推理引擎和 GPU 硬件。我们与合作伙伴和开源社区紧密合作，在不同的推理引擎和 GPU 上测试和优化新兴模型。以下是当前支持的推理引擎、GPU 和模型列表，该列表将随着时间的推移继续扩展。

**经过测试的推理引擎：**
- vLLM
- SGLang
- TensorRT-LLM
- MindIE

**经过测试的 GPU：**
- NVIDIA A100
- NVIDIA H100/H200
- Ascend 910B

**经过调优的模型：**
- Qwen3
- gpt-oss
- GLM-4.5-Air
- GLM-4.5/4.6
- DeepSeek-R1

## 架构

GPUStack 使开发团队、IT 组织和服务提供商能够大规模地提供模型即服务。它支持用于 LLM、语音、图像和视频模型的行业标准 API。该平台内置用户认证和访问控制、GPU 性能和利用率的实时监控，以及令牌使用量和 API 请求率的详细计量。

下图展示了单个 GPUStack 服务器如何管理跨本地和云环境的多个 GPU 集群。GPUStack 调度器分配 GPU 以最大化资源利用率，并选择合适的推理引擎以实现最佳性能。管理员还可以通过集成的 Grafana 和 Prometheus 仪表板全面了解系统运行状况和指标。

![gpustack-v2-architecture](docs/assets/gpustack-v2-architecture.png)

GPUStack 为部署 AI 模型提供了一个强大的框架。其核心功能包括：
- **多集群 GPU 管理。** 跨多个环境管理 GPU 集群。这包括本地服务器、Kubernetes 集群和云提供商。
- **可插拔推理引擎。** 自动配置高性能推理引擎，如 vLLM、SGLang 和 TensorRT-LLM。您也可以根据需要添加自定义推理引擎。
- **性能优化配置。** 提供预调优模式，用于低延迟或高吞吐量。GPUStack 支持扩展的 KV 缓存系统，如 LMCache 和 HiCache，以减少 TTFT。它还包括对推测性解码方法（如 EAGLE3、MTP 和 N-grams）的内置支持。
- **企业级运维能力。** 支持自动故障恢复、负载均衡、监控、认证和访问控制。

## 安装

> GPUStack 目前仅支持 Linux。

如果您使用 NVIDIA GPU，请确保已安装 NVIDIA 驱动程序、[Docker](https://docs.docker.com/engine/install/) 和 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。然后使用以下命令启动 GPUStack：

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

如果您无法从 `Docker Hub` 拉取镜像或下载速度很慢，可以使用我们的 `Quay.io` 镜像，将仓库指向 `quay.io`：

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --runtime nvidia \
    quay.io/gpustack/gpustack \
    --system-default-container-registry quay.io
```

有关安装或其他 GPU 硬件平台的更多详细信息，请参阅 [安装要求](https://docs.gpustack.ai/latest/installation/requirements/)。

检查 GPUStack 启动日志：

```bash
sudo docker logs -f gpustack
```

GPUStack 启动后，运行以下命令获取默认管理员密码：

```bash
sudo docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

打开浏览器并访问 `http://您的服务器IP` 以进入 GPUStack 用户界面。使用默认用户名 `admin` 和上面获取的密码登录。

## 部署模型

1.  在 GPUStack 用户界面中导航到 `Catalog` 页面。
2.  从可用模型列表中选择 `Qwen3 0.6B` 模型。
3.  部署兼容性检查通过后，点击 `Save` 按钮部署模型。

![从目录部署 qwen3](docs/assets/quick-start/quick-start-qwen3.png)

4.  GPUStack 将开始下载模型文件并部署模型。当部署状态显示为 `Running` 时，表示模型已成功部署。

![模型运行中](docs/assets/quick-start/model-running.png)

5.  点击导航菜单中的 `Playground - Chat`，检查右上角 `Model` 下拉菜单中是否选中了 `qwen3-0.6b` 模型。现在您可以在 UI  playground 中与模型聊天了。

![快速聊天](docs/assets/quick-start/quick-chat.png)

## 通过 API 使用模型

1.  将鼠标悬停在用户头像上，导航到 `API Keys` 页面，然后点击 `New API Key` 按钮。
2.  填写 `Name` 并点击 `Save` 按钮。
3.  复制生成的 API 密钥并将其保存在安全的地方。请注意，该密钥仅在创建时可见一次。
4.  您现在可以使用该 API 密钥访问 GPUStack 提供的 OpenAI 兼容 API 端点。例如，使用 curl 如下所示：

```bash
# 将 `your_api_key` 和 `your_gpustack_server_url`
# 替换为您实际的 API 密钥和 GPUStack 服务器 URL。
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3-0.6b",
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

## 文档

请参阅 [官方文档站点](https://docs.gpustack.ai) 获取完整文档。

## 构建

1.  安装 Python（版本 3.10 到 3.12）。
2.  运行 `make build`。

您可以在 `dist` 目录中找到构建好的 wheel 包。

## 贡献

如果您有兴趣为 GPUStack 做贡献，请阅读 [贡献指南](./docs/contributing.md)。

## 加入社区

如有任何问题或建议，欢迎加入我们的 [社区](https://discord.gg/VXYJzuaqwD) 寻求支持。

## 许可证

版权所有 (c) 2024-2025 GPUStack 作者

根据 Apache License, Version 2.0（"许可证"）授权；
除非符合许可证，否则您不得使用此文件。
您可以在 [LICENSE](./LICENSE) 文件中获取许可证副本。

除非适用法律要求或书面同意，根据许可证分发的软件按"原样"分发，无任何明示或暗示的担保或条件。
请参阅许可证中规定的特定语言管理权限及许可证下的限制。
