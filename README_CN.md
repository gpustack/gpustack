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

![demo](https://raw.githubusercontent.com/gpustack/gpustack/main/docs/assets/gpustack-demo.gif)

GPUStack 是一个用于运行 AI 模型的开源 GPU 集群管理器。

### 核心特性

- **广泛的 GPU 兼容性**：无缝支持 Apple Mac、Windows PC 和 Linux 服务器上各种供应商的 GPU。
- **广泛的模型支持**：支持各种模型，包括 LLM、多模态 VLM、图像模型、语音模型、文本嵌入模型和重排序模型。
- **灵活的推理后端**：支持与 vLLM 、 Ascend MindIE、llama-box（llama.cpp 和 stable-diffusion.cpp）和 vox-box 等多种推理后端的灵活集成。
- **多版本后端支持**：同时运行推理后端的多个版本，以满足不同模型的不同运行依赖。
- **分布式推理**：支持单机和多机多卡并行推理，包括跨供应商和运行环境的异构 GPU。
- **可扩展的 GPU 架构**：通过向基础设施添加更多 GPU 或节点轻松进行扩展。
- **强大的模型稳定性**：通过自动故障恢复、多实例冗余和推理请求的负载平衡确保高可用性。
- **智能部署评估**：自动评估模型资源需求、后端和架构兼容性、操作系统兼容性以及其他与部署相关的因素。
- **自动调度**：根据可用资源动态分配模型。
- **轻量级 Python 包**：最小依赖性和低操作开销。
- **OpenAI 兼容 API**：完全兼容 OpenAI 的 API 规范，实现无缝集成。
- **用户和 API 密钥管理**：简化用户和 API 密钥的管理。
- **实时 GPU 监控**：实时跟踪 GPU 性能和利用率。
- **令牌和速率指标**：监控 Token 使用情况和 API 请求速率。

## 安装

### Linux

如果你是 NVIDIA GPU 环境，请确保 [Docker](https://docs.docker.com/engine/install/) 和 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 都已经在系统中安装。 然后，执行如下命令启动 GPUStack：

```bash
docker run -d --name gpustack \
      --restart=unless-stopped \
      --gpus all \
      --network=host \
      --ipc=host \
      -v gpustack-data:/var/lib/gpustack \
      gpustack/gpustack
```

有关其它平台的安装或详细配置选项，请参考[安装文档](docs/installation/requirements)。

容器正常运行后，执行以下命令获取默认密码：

```bash
docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

在浏览器中打开 `http://your_host_ip`，访问 GPUStack 界面。使用 `admin` 用户名和默认密码登录 GPUStack。

### macOS & Windows

对于 macOS 和 Windows，我们提供了桌面安装程序。请参阅[文档](https://docs.gpustack.ai/latest/installation/desktop-installer/)了解安装细节。

## 部署模型

1. 在 GPUStack 界面，在菜单中点击“模型库”。

2. 从模型列表中选择 `Qwen3` 模型。

3. 在部署兼容性检查通过之后，选择保存部署模型。

![deploy qwen3 from catalog](docs/assets/quick-start/quick-start-qwen3.png)

4. GPUStack 将开始下载模型文件并部署模型。当部署状态显示为 `Running` 时，表示模型已成功部署。

![model is running](docs/assets/quick-start/model-running.png)

5. 点击菜单中的“试验场 - 对话”，在右上方模型菜单中选择模型 `qwen3`。现在你可以在试验场中与 LLM 进行对话。

![quick chat](docs/assets/quick-start/quick-chat.png)

## 通过 API 使用模型

1. 将鼠标移动到右下角的用户头像上，选择“API 密钥”，然后点击“新建 API 秘钥”按钮。

2. 填写“名称”，然后点击“保存”按钮。

3. 复制生成的 API 密钥并将其保存。请注意，秘钥只在创建时可见。

4. 现在你可以使用 API 密钥访问 OpenAI 兼容 API。例如，curl 的用法如下：

```bash
# Replace `your_api_key` and `your_gpustack_server_url`
# with your actual API key and GPUStack server URL.
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3",
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

## 平台支持

- [x] Linux
- [x] macOS
- [x] Windows

## 加速框架支持

- [x] NVIDIA CUDA ([Compute Capability](https://developer.nvidia.com/cuda-gpus) 6.0 以上)
- [x] Apple Metal (M 系列芯片)
- [x] AMD ROCm
- [x] 昇腾 CANN
- [x] 海光 DTK
- [x] 摩尔线程 MUSA
- [x] 天数智芯 Corex
- [x] 寒武纪 MLU

## 模型支持

GPUStack 使用 [vLLM](https://github.com/vllm-project/vllm)、 [Ascend MindIE](https://www.hiascend.com/en/software/mindie)、[llama-box](https://github.com/gpustack/llama-box)（基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 和 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)）和 [vox-box](https://github.com/gpustack/vox-box) 作为后端并提供广泛的模型支持。支持从以下来源部署模型：

1. [Hugging Face](https://huggingface.co/)

2. [ModelScope](https://modelscope.cn/)

3. 本地文件路径

### 示例模型

| **类别**               | **模型**                                                                                                                                                                                                                                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **大语言模型（LLM）**  | [Qwen](https://huggingface.co/models?search=Qwen/Qwen), [LLaMA](https://huggingface.co/meta-llama), [Mistral](https://huggingface.co/mistralai), [DeepSeek](https://huggingface.co/models?search=deepseek-ai/deepseek), [Phi](https://huggingface.co/models?search=microsoft/phi), [Gemma](https://huggingface.co/models?search=Google/gemma)    |
| **多模态模型（VLM）**  | [Llama3.2-Vision](https://huggingface.co/models?pipeline_tag=image-text-to-text&search=llama3.2), [Pixtral](https://huggingface.co/models?search=pixtral) , [Qwen2.5-VL](https://huggingface.co/models?search=Qwen/Qwen2.5-VL), [LLaVA](https://huggingface.co/models?search=llava), [InternVL3](https://huggingface.co/models?search=internvl3) |
| **Diffusion 扩散模型** | [Stable Diffusion](https://huggingface.co/models?search=gpustack/stable-diffusion), [FLUX](https://huggingface.co/models?search=gpustack/flux)                                                                                                                                                                                                   |
| **Embedding 模型**     | [BGE](https://huggingface.co/gpustack/bge-m3-GGUF), [BCE](https://huggingface.co/gpustack/bce-embedding-base_v1-GGUF), [Jina](https://huggingface.co/models?search=gpustack/jina-embeddings), [Qwen3-Embedding](https://huggingface.co/models?search=qwen/qwen3-embedding)                                                                       |
| **Reranker 模型**      | [BGE](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF), [BCE](https://huggingface.co/gpustack/bce-reranker-base_v1-GGUF), [Jina](https://huggingface.co/models?search=gpustack/jina-reranker), [Qwen3-Reranker](https://huggingface.co/models?search=qwen/qwen3-reranker)                                                                |
| **语音模型**           | [Whisper](https://huggingface.co/models?search=Systran/faster) (Speech-to-Text), [CosyVoice](https://huggingface.co/models?search=FunAudioLLM/CosyVoice) (Text-to-Speech)                                                                                                                                                                        |

有关支持模型的完整列表，请参阅 [inference backends](https://docs.gpustack.ai/latest/user-guide/inference-backends/) 文档中的 Supported Models 部分。

## OpenAI 兼容 API

GPUStack 在 `/v1-openai` 路径提供以下 OpenAI 兼容 API：

- [x] [List Models](https://platform.openai.com/docs/api-reference/models/list)
- [x] [Create Completion](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [Create Chat Completion](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [Create Embeddings](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [Create Image](https://platform.openai.com/docs/api-reference/images/create)
- [x] [Create Image Edit](https://platform.openai.com/docs/api-reference/images/createEdit)
- [x] [Create Speech](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [x] [Create Transcription](https://platform.openai.com/docs/api-reference/audio/createTranscription)

例如，你可以使用官方的 [OpenAI Python API 库](https://github.com/openai/openai-python)来调用 API：

```python
from openai import OpenAI
client = OpenAI(base_url="http://your_gpustack_server_url/v1-openai", api_key="your_api_key")

completion = client.chat.completions.create(
  model="llama3.2",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```

GPUStack 用户可以在 UI 中生成自己的 API 密钥。

## 文档

完整文档请参见[官方文档](https://docs.gpustack.ai)。

## 构建

1. 安装 Python（版本 3.10 ~ 3.12）。

2. 运行 `make build`。

你可以在 `dist` 目录下找到构建的 wheel 包。

## Contributing

如果你有兴趣参与 GPUStack 贡献代码，请阅读[贡献指南](./docs/contributing.md)。

## 加入社区

扫码加入社区群：

<p align="left">
    <img alt="Wechat-group" src="./docs/assets/wechat-group-qrcode.jpg" width="300px"/>
</p>

## License

版权所有 (c) 2024 GPUStack 作者

本项目基于 Apache-2.0 许可证（以下简称“许可证”）授权。
您只能在遵守许可证条款的前提下使用本项目。
许可证的完整内容请参阅 [LICENSE](./LICENSE) 文件。

除非适用法律另有规定或双方另有书面约定，依据许可证分发的软件按“原样”提供，
不附带任何明示或暗示的保证或条件。
有关许可证规定的具体权利和限制，请参阅许可证了解更多详细信息。
