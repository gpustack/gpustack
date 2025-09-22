<br>

<p align="center">
    <img alt="GPUStack" src="https://raw.githubusercontent.com/gpustack/gpustack/main/docs/assets/gpustack-logo.png" width="300px"/>
</p>

<br>

<p align="center">
  <a href="https://github.com/gpustack/gpustack/blob/main/LICENSE" target="_blank">
    <img alt="许可证" src="https://img.shields.io/github/license/gpustack/gpustack?logo=github&logoColor=white&label=License&color=blue">
  </a>
  <a href="https://discord.gg/VXYJzuaqwD" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-GPUStack-blue?logo=discord&logoColor=white">
  </a>
  <a href="../../assets/wechat-group-qrcode.jpg" target="_blank">
    <img alt="微信" src="https://img.shields.io/badge/微信群-GPUStack-blue?logo=wechat&logoColor=white">
  </a>
</p>

<p align="center">
  <script async defer src="https://buttons.github.io/buttons.js"></script>
  <a class="github-button" href="https://github.com/gpustack/gpustack" data-show-count="true" data-size="large" aria-label="标星">标星</a>
  <a class="github-button" href="https://github.com/gpustack/gpustack/subscription" data-icon="octicon-eye" data-size="large" aria-label="关注">关注</a>
  <a class="github-button" href="https://github.com/gpustack/gpustack/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="派生">派生</a>
</p>

GPUStack 是一个用于运行 AI 模型的开源 GPU 集群管理器。

### 关键特性

- **广泛的 GPU 兼容性：** 无缝支持来自不同厂商的 GPU，覆盖 Apple Mac、Windows PC 和 Linux 服务器。
- **广泛的模型支持：** 支持包括 LLM、VLM、图像模型、音频模型、嵌入模型和重排模型在内的广泛模型。
- **灵活的推理后端：** 灵活集成多个推理后端，包括 vLLM、Ascend MindIE、llama-box（llama.cpp 与 stable-diffusion.cpp）以及 vox-box。
- **多版本后端支持：** 可并行运行多个版本的推理后端，以满足不同模型的多样化运行时需求。
- **分布式推理：** 支持单机和多机多 GPU 推理，包括跨厂商与异构运行环境的 GPU。
- **可扩展的 GPU 架构：** 通过添加更多 GPU 或节点轻松扩展规模。
- **稳健的模型稳定性：** 通过自动故障恢复、多实例冗余和推理请求负载均衡，确保高可用性。
- **智能部署评估：** 自动评估模型资源需求、后端与架构兼容性、操作系统兼容性及其他与部署相关的因素。
- **自动化调度：** 基于可用资源动态分配模型。
- **轻量级 Python 包：** 依赖最小、运维开销低。
- **OpenAI 兼容 API：** 完全兼容 OpenAI 的 API 规范，便于无缝集成。
- **用户与 API Key 管理：** 简化用户与 API Key 的管理。
- **GPU 实时监控：** 实时跟踪 GPU 性能与利用率。
- **令牌与速率指标：** 监控令牌用量与 API 请求速率。

## 支持的平台

- [x] Linux
- [x] macOS
- [x] Windows

## 支持的加速器

- [x] NVIDIA CUDA（[计算能力](https://developer.nvidia.com/cuda-gpus) 6.0 及以上）
- [x] Apple Metal（M 系列芯片）
- [x] AMD ROCm
- [x] Ascend CANN
- [x] Hygon DTK
- [x] Moore Threads MUSA
- [x] Iluvatar Corex
- [x] Cambricon MLU

## 支持的模型

GPUStack 使用 [vLLM](https://github.com/vllm-project/vllm)、[Ascend MindIE](https://www.hiascend.com/en/software/mindie)、[llama-box](https://github.com/gpustack/llama-box)（内置 [llama.cpp](https://github.com/ggml-org/llama.cpp) 与 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) 服务器）以及 [vox-box](https://github.com/gpustack/vox-box) 作为后端，支持范围广泛的模型。支持来自以下来源的模型：

1. [Hugging Face](https://huggingface.co/)

2. [ModelScope](https://modelscope.cn/)

3. 本地文件路径

### 示例模型

| **类别**                        | **模型**                                                                                                                                                                                                                                                                                                                                 |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **大型语言模型（LLMs）**        | [Qwen](https://huggingface.co/models?search=Qwen/Qwen), [LLaMA](https://huggingface.co/meta-llama), [Mistral](https://huggingface.co/mistralai), [DeepSeek](https://huggingface.co/models?search=deepseek-ai/deepseek), [Phi](https://huggingface.co/models?search=microsoft/phi), [Gemma](https://huggingface.co/models?search=Google/gemma) |
| **视觉语言模型（VLMs）**        | [Llama3.2-Vision](https://huggingface.co/models?pipeline_tag=image-text-to-text&search=llama3.2), [Pixtral](https://huggingface.co/models?search=pixtral) , [Qwen2.5-VL](https://huggingface.co/models?search=Qwen/Qwen2.5-VL), [LLaVA](https://huggingface.co/models?search=llava), [InternVL3](https://huggingface.co/models?search=internvl3) |
| **扩散模型**                    | [Stable Diffusion](https://huggingface.co/models?search=gpustack/stable-diffusion), [FLUX](https://huggingface.co/models?search=gpustack/flux)                                                                                                                                                                                           |
| **嵌入模型**                    | [BGE](https://huggingface.co/gpustack/bge-m3-GGUF), [BCE](https://huggingface.co/gpustack/bce-embedding-base_v1-GGUF), [Jina](https://huggingface.co/models?search=gpustack/jina-embeddings), [Qwen3-Embedding](https://huggingface.co/models?search=qwen/qwen3-embedding)                                                           |
| **重排模型**                    | [BGE](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF), [BCE](https://huggingface.co/gpustack/bce-reranker-base_v1-GGUF), [Jina](https://huggingface.co/models?search=gpustack/jina-reranker), [Qwen3-Reranker](https://huggingface.co/models?search=qwen/qwen3-reranker)                                              |
| **音频模型**                    | [Whisper](https://huggingface.co/models?search=Systran/faster)（语音转文本），[CosyVoice](https://huggingface.co/models?search=FunAudioLLM/CosyVoice)（文本转语音）                                                                                                                                                                     |

有关完整的支持模型列表，请参阅[推理后端](user-guide/inference-backends.md)文档中的“支持的模型”章节。

## OpenAI 兼容 API

GPUStack 提供与 OpenAI 兼容的 API。详情请参阅 [OpenAI 兼容 API](user-guide/openai-compatible-apis.md)