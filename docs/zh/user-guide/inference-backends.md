# 推理后端

GPUStack 支持以下推理后端：

- [llama-box](#llama-box)
- [vLLM](#vllm)
- [vox-box](#vox-box)
- [Ascend MindIE](#ascend-mindie-experimental)

当用户部署模型时，后端会根据以下规则自动选择：

- 如果模型为 [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) 格式，则使用 `llama-box`。
- 如果模型是已知的 `Text-to-Speech` 或 `Speech-to-Text` 模型，则使用 `vox-box`。
- 否则，使用 `vLLM`。

## llama-box

[llama-box](https://github.com/gpustack/llama-box) 是基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 和 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) 的大模型推理服务器。

### 支持的平台

llama-box 后端支持 Linux、macOS 和 Windows（Windows 仅在 ARM 架构上支持 CPU 旁路）平台。

### 支持的模型

- LLM：支持的 LLM 请参考 llama.cpp 的 [README](https://github.com/ggml-org/llama.cpp#description)。
- 扩散模型：支持的模型列在此 [Hugging Face 集合](https://huggingface.co/collections/gpustack/image-672dafeb2fa0d02dbe2539a9) 或此 [ModelScope 集合](https://modelscope.cn/collections/Image-fab3d241f8a641)。
- Reranker 模型：支持的模型见此 [Hugging Face 集合](https://huggingface.co/collections/gpustack/reranker-6721a234527f6fcd90deedc4) 或此 [ModelScope 集合](https://modelscope.cn/collections/Reranker-7576210e79de4a)。

### 支持的特性

#### 允许 CPU 旁路

启用 CPU 旁路后，GPUStack 将优先尽可能多地把模型层加载到 GPU 上以优化性能。如果 GPU 资源有限，则会将部分层旁路到 CPU；仅当没有可用 GPU 时，才会完全使用 CPU 推理。

#### 允许跨 Worker 的分布式推理

启用在多个 worker 间的分布式推理。主模型实例将与一个或多个其他 worker 上的后端实例通信，并将计算任务卸载给它们。

#### 多模态语言模型

Llama-box 支持以下视觉语言模型。使用视觉语言模型时，聊天补全 API 支持传入图像输入。

- LLaVA 系列
- MiniCPM VL 系列
- Qwen2 VL 系列
- GLM-Edge-V 系列
- Granite VL 系列
- Gemma3 VL 系列
- SmolVLM 系列
- Pixtral 系列
- MobileVLM 系列
- Mistral Small 3.1
- Qwen2.5 VL 系列

!!! Note

    部署视觉语言模型时，GPUStack 会默认下载并使用符合 `*mmproj*.gguf` 模式的多模态投影器文件。如果有多个文件匹配，GPUStack 会选择精度更高的文件（例如优先选择 `f32` 而非 `f16`）。如果默认模式未能匹配到投影器文件，或你希望指定特定文件，可在模型配置中通过 `--mmproj` 参数自定义多模态投影器文件。你可以在模型来源中指定该文件的相对路径。该语法相当于简写，GPUStack 会在使用时从来源下载该文件并规范化路径。

<a id="parameters-reference_1"></a>

### 参数参考

完整的 llama-box 支持参数见[此处](https://github.com/gpustack/llama-box#usage)。

## vLLM

[vLLM](https://github.com/vllm-project/vllm) 是一个高吞吐、低内存占用的 LLM 推理引擎，是生产环境运行 LLM 的常用选择。vLLM 无缝支持大多数最新的开源模型，包括：类 Transformer 的 LLM（如 Llama）、Mixture-of-Experts LLM（如 Mixtral）、嵌入模型（如 E5-Mistral）、多模态 LLM（如 LLaVA）。

默认情况下，GPUStack 会基于模型元数据估算该模型实例的显存需求。你可以按需自定义参数。以下 vLLM 参数可能有用：

- `--gpu-memory-utilization`（默认：0.9）：模型实例使用的 GPU 内存占比。
- `--max-model-len`：模型上下文长度。对于长上下文模型，GPUStack 会自动将该参数设置为 `8192`，以简化在资源受限环境中的部署。你也可以按需自定义该参数。
- `--tensor-parallel-size`：张量并行副本数量。默认情况下，GPUStack 会根据可用 GPU 资源与模型内存需求估算设置该参数。你也可以按需自定义该参数。

更多细节请参考 [vLLM 文档](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#command-line-arguments-for-the-server)。

### 支持的平台

vLLM 后端支持在 Linux 上运行。

!!! Note

    1. 当用户在 amd64 Linux 上通过安装脚本安装 GPUStack 时，会自动安装 vLLM。
    2. 当用户使用 vLLM 后端部署模型时，GPUStack 默认将 worker 标签选择器设置为 `{"os": "linux"}`，以确保模型实例被调度到合适的 worker。你可以在模型配置中自定义 worker 标签选择器。

### 支持的模型

支持的模型请参阅 vLLM 的[文档](https://docs.vllm.ai/en/stable/models/supported_models.html)。

### 支持的特性

#### 多模态语言模型

vLLM 支持的多模态语言模型列表见[此处](https://docs.vllm.ai/en/stable/models/supported_models.html#multimodal-language-models)。当用户使用 vLLM 后端部署视觉语言模型时，聊天补全 API 支持图像输入。

#### 跨 Worker 的分布式推理（实验性）

vLLM 通过 [Ray](https://ray.io) 支持跨多个 worker 的分布式推理。你可以在启动 GPUStack 时使用 `--enable-ray` 参数启用 Ray 集群，使 vLLM 能够在多个 worker 之间进行分布式推理。

!!! warning "已知限制"

    1. GPUStack 服务器和所有参与的 worker 必须运行在 Linux 上，并使用相同版本的 Python（Ray 的要求）。
    2. 模型文件必须在所有参与的 worker 上以相同路径可访问。你必须使用共享文件系统，或在所有参与的 worker 上将模型文件下载到相同路径。
    3. 每个 worker 同一时间只能被分配给一个分布式 vLLM 模型实例。
    4. 如果自定义的 vLLM 版本中 Ray 的分布式执行器实现与内置 vLLM 版本不兼容，则可能无法正常工作。
    5. 若通过 Docker 安装 GPUStack，必须使用 host 网络模式以利用 RDMA/InfiniBand 并确保节点间连通性。

在以下条件下支持自动调度：

- 参与的 worker 拥有相同数量的 GPU。
- 每个 worker 中的所有 GPU 都满足 gpu_memory_utilization（默认 0.9）的要求。
- GPU 总数可以被注意力头数整除。
- 申请的显存总量大于估算的显存需求。

若不满足上述条件，将不会自动调度模型实例。不过，你可以在模型配置中手动选择目标 worker/GPU 进行调度。

### 参数参考

完整的 vLLM 支持参数见[此处](https://docs.vllm.ai/en/stable/cli/serve.html)。

## vox-box

[vox-box](https://github.com/gpustack/vox-box) 是为部署文本转语音（Text-to-Speech）与语音转文本（Speech-to-Text）模型而设计的推理引擎，同时提供完全兼容 OpenAI Audio API 的接口。

### 支持的平台

vox-box 后端支持 Linux、macOS 和 Windows 平台。

!!! note

    1. 若需使用 Nvidia GPU，请确保 worker 上安装以下 NVIDIA 库：
        - [适用于 CUDA 12 的 cuBLAS](https://developer.nvidia.com/cublas)
        - [适用于 CUDA 12 的 cuDNN 9](https://developer.nvidia.com/cudnn)
    2. 当用户在 Linux、macOS 和 Windows 上通过安装脚本安装 GPUStack 时，会自动安装 vox-box。
    3. CosyVoice 模型在 Linux 的 AMD 架构与 macOS 上原生支持。但在 Linux ARM 或 Windows 架构上不支持这些模型。

### 支持的模型

| 模型                              | 类型            | 链接                                                                                                                                                                 | 支持的平台                                              |
| --------------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Faster-whisper-large-v3           | speech-to-text  | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v3)                 | Linux、macOS、Windows                                   |
| Faster-whisper-large-v2           | speech-to-text  | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v2), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v2)                 | Linux、macOS、Windows                                   |
| Faster-whisper-large-v1           | speech-to-text  | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v1), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v1)                 | Linux、macOS、Windows                                   |
| Faster-whisper-medium             | speech-to-text  | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-medium)                     | Linux、macOS、Windows                                   |
| Faster-whisper-medium.en          | speech-to-text  | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-medium.en)               | Linux、macOS、Windows                                   |
| Faster-whisper-small              | speech-to-text  | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-small)                       | Linux、macOS、Windows                                   |
| Faster-whisper-small.en           | speech-to-text  | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-small.en)                 | Linux、macOS、Windows                                   |
| Faster-distil-whisper-large-v3    | speech-to-text  | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v3), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-large-v3)   | Linux、macOS、Windows                                   |
| Faster-distil-whisper-large-v2    | speech-to-text  | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v2), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-large-v2)   | Linux、macOS、Windows                                   |
| Faster-distil-whisper-medium.en   | speech-to-text  | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-medium.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-medium.en) | Linux、macOS、Windows                                   |
| Faster-whisper-tiny               | speech-to-text  | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-tiny)                         | Linux、macOS、Windows                                   |
| Faster-whisper-tiny.en            | speech-to-text  | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-tiny.en)                   | Linux、macOS、Windows                                   |
| CosyVoice-300M-Instruct           | text-to-speech  | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M-Instruct), [ModelScope](https://modelscope.cn/models/gpustack/CosyVoice-300M-Instruct)                | Linux（不支持 ARM）、macOS、Windows（不支持）           |
| CosyVoice-300M-SFT                | text-to-speech  | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M-SFT), [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-SFT)                               | Linux（不支持 ARM）、macOS、Windows（不支持）           |
| CosyVoice-300M                    | text-to-speech  | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M), [ModelScope](https://modelscope.cn/models/gpustack/CosyVoice-300M)                                  | Linux（不支持 ARM）、macOS、Windows（不支持）           |
| CosyVoice-300M-25Hz               | text-to-speech  | [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-25Hz)                                                                                                  | Linux（不支持 ARM）、macOS、Windows（不支持）           |
| CosyVoice2-0.5B                   | text-to-speech  | [Hugging Face](https://huggingface.co/gpustack/CosyVoice2-0.5B), [ModelScope](https://modelscope.cn/models/iic/CosyVoice2-0.5B)                                     | Linux（不支持 ARM）、macOS、Windows（不支持）           |
| Dia-1.6B                          | text-to-speech  | [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B), [ModelScope](https://modelscope.cn/models/nari-labs/Dia-1.6B)                                            | Linux（不支持 ARM）、macOS、Windows（不支持）           |

### 支持的特性

#### 允许 GPU/CPU 旁路

vox-box 支持将模型部署到 NVIDIA GPU。当 GPU 资源不足时，会自动将模型部署到 CPU。

## Ascend MindIE (Experimental)

[Ascend MindIE](https://www.hiascend.com/en/software/mindie) 是基于 [Ascend 硬件](https://www.hiascend.com/en/hardware/product) 的高性能推理服务。

### 支持的平台

Ascend MindIE 后端默认使用 Ascend MindIE 2.0.RC2，仅兼容 Linux 平台，支持 ARM64 与 x86_64 架构。

### 支持的模型

Ascend MindIE 支持的各类模型列在[这里](https://www.hiascend.com/software/mindie/modellist)。

在 GPUStack 中，支持
[大语言模型（LLMs）](https://www.hiascend.com/software/mindie/modellist)
和
[多模态语言模型（VLMs）](https://www.hiascend.com/software/mindie/modellist)
。但目前尚不支持“嵌入模型”和“多模态生成模型”。

### 支持的特性

Ascend MindIE 拥有多种特性，概述见[这里](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0001.html)。

目前，GPUStack 支持其中的部分能力，包括
[量化](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0288.html)、
[扩展上下文长度](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0295.html)、
[分布式推理](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0296.html)、
[多专家混合（MoE）](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0297.html)、
[Split Fuse](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0300.html)、
[推测解码](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0301.html)、
[多 Token 预测](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0500.html)、
[前缀缓存](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0302.html)、
[函数调用](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0303.html)、
[多模态理解](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0304.html)、
[多头潜在注意力（MLA）](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0305.html)、
[（数据）并行](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0424.html)、
[缓冲响应（自 Ascend MindIE 2.0.RC1 起）](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0425.html)。

!!! Note

    1. 量化需要特定权重，并且必须调整模型的 `config.json`。请按照[参考（指南）](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0288.html)准备正确的权重。
    2. 某些特性互斥，使用时请注意。例如，启用前缀缓存时，不能使用扩展上下文长度特性。

### 参数参考

Ascend MindIE 拥有可配置的[参数](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0004.html)和[环境变量](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0416.html)。

为避免直接编辑 JSON，GPUStack 提供了以下一组命令行参数。

| 参数                                   | 默认值 | 取值范围                         | 作用域                                  | 说明                                                                                                                                                                                                                                                                             |
| -------------------------------------- | ------ | -------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--log-level`                          | Info   |                                  | Log Config                              | MindIE 的日志级别。可选：`Verbose`、`Info`、`Warning`、`Warn`、`Error`、`Debug`。                                                                                                                                                                                                |
| `--max-link-num`                       | 1000   | [1, 1000]                        | Server Config                           | 最大并行请求数。                                                                                                                                                                                                                                                                |
| `--token-timeout`                      | 60     | [1, 3600]                        | Server Config                           | 单个 token 生成超时时间（秒）。                                                                                                                                                                                                                                                 |
| `--e2e-timeout`                        | 60     | [1, 3600]                        | Server Config                           | 端到端（从请求被接受到推理结束）超时时间（秒）。                                                                                                                                                                                                                                 |
| `--max-seq-len`                        | 8192   | (0, \*)                          | Model Deploy Config                     | 模型上下文长度。如未指定，将从模型配置推导。                                                                                                                                                                                                                                     |
| `--max-input-token-len`                |        | (0, `--max-seq-len`]             | Model Deploy Config                     | 最大输入 token 长度。如未指定，将从 `--max-seq-len` 推导。                                                                                                                                                                                                                        |
| `--truncation`                         |        |                                  | Model Deploy Config                     | 当输入 token 长度超过 `--max-input-token-len` 与 `--max-seq-len` - 1 的较小值时进行截断。                                                                                                                                                                                        |
| `--cpu-mem-size`                       | 0      | [0, \*)                          | Model Config                            | CPU 交换空间大小（GiB）。在指定了 `--max-preempt-count` 时生效。                                                                                                                                                                                                                 |
| `--npu-memory-fraction`                | 0.9    | (0, 1]                           | Model Config                            | 模型执行器可使用的 NPU 内存占比（0 到 1）。例如：`0.5` 表示 50% 内存利用率。                                                                                                                                                                                                     |
| `--trust-remote-code`                  |        |                                  | Model Config                            | 信任远程代码（用于模型加载）。                                                                                                                                                                                                                                                   |
| `--cache-block-size`                   | 128    |                                  | Schedule Config                         | KV 缓存块大小。必须为 2 的幂。                                                                                                                                                                                                                                                   |
| `--max-prefill-batch-size`             | 50     | [1, `--max-batch-size`]          | Schedule Config                         | 预填阶段的最大批处理请求数。必须小于 `--max-batch-size`。                                                                                                                                                                                                                        |
| `--prefill-time-ms-per-req`            | 150    | [0, 1000]                        | Schedule Config                         | 预填阶段的单请求预估时延（毫秒）。用于在预填与解码阶段之间进行决策。                                                                                                                                                                                                              |
| `--prefill-policy-type`                | 0      |                                  | Schedule Config                         | 预填阶段策略：<br> `0`：FCFS（先来先服务）。<br> `1`：STATE（与 FCFS 相同）。<br> `2`：PRIORITY（优先级队列）。<br> `3`：MLFQ（多级反馈队列）。                                                                                                                                    |
| `--max-batch-size`                     | 200    | [1, 5000]                        | Schedule Config                         | 解码阶段的最大批处理请求数。                                                                                                                                                                                                                                                     |
| `--decode-time-ms-per-req`             | 50     | [0, 1000]                        | Schedule Config                         | 解码阶段的单请求预估时延（毫秒）。与 `--prefill-time-ms-per-req` 配合用于批次选择。                                                                                                                                                                                               |
| `--decode-policy-type`                 | 0      |                                  | Schedule Config                         | 解码阶段策略：<br> `0`：FCFS <br> `1`：STATE（优先被抢占或已换出的请求）<br> `2`：PRIORITY <br> `3`：MLFQ                                                                                                                                                                          |
| `--max-preempt-count`                  | 0      | [0, `--max-batch-size`]          | Schedule Config                         | 解码阶段允许的最大抢占请求数。必须小于 `--max-batch-size`。                                                                                                                                                                                                                       |
| `--support-select-batch`               |        |                                  | Schedule Config                         | 启用批次选择。根据 `--prefill-time-ms-per-req` 与 `--decode-time-ms-per-req` 决定执行优先级。若需显式禁用，请使用 `--no-support-select-batch`。                                                                                                                                    |
| `--max-queue-delay-microseconds`       | 5000   | [500, 1000000]                   | Schedule Config                         | 最大排队等待时间（微秒）。                                                                                                                                                                                                                                                       |
| `--override-generation-config`         |        |                                  |                                         | 以 JSON 格式覆盖或设置生成配置。例如：`{"temperature": 0.5}`。该配置会合并进模型结构的 `generation_config.json`。                                                                                                                                                                 |
| `--enforce-eager`                      |        |                                  |                                         | 以 Eager 模式执行算子。                                                                                                                                                                                                                                                          |
| `--metrics`                            |        |                                  |                                         | 在 `/metrics` 端点暴露指标。                                                                                                                                                                                                                                                     |
| `--dtype`                              | auto   |                                  |                                         | 模型权重与激活的数据类型。<br> `auto`：使用模型配置的默认数据类型。<br> `half`/`float16`：FP16。<br> `bfloat16`：BF16。<br> `float`/`float32`：FP32。                                                                                                                              |
| `--rope-scaling`                       |        |                                  | Extending Context Size                  | RoPE 缩放的 JSON 配置。例如：`{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}`。该配置会合并进模型结构的 `config.json`。                                                                                                                                     |
| `--rope-theta`                         |        |                                  | Extending Context Size                  | RoPE theta 配置。该配置会合并进模型结构的 `config.json`。                                                                                                                                                                                                                         |
| `--enable-split`                       |        |                                  | Split Fuse                              | 启用 Split Fuse，类似分块预填。若需显式禁用，请使用 `--no-enable-split`。                                                                                                                                                                                                          |
| `--policy-type`                        | 0      |                                  | Split Fuse                              | Split Fuse 的策略。<br> `0`：FCFS，先来先服务。<br> `4`：SJF，最短作业优先。<br> `5`：LJF，最长作业优先。<br> `6`：Skip-Join MLFQ，跳转式多级反馈队列。<br> `7`：SJF-MLFQ，最短作业优先 + 多级反馈队列。                                                                             |
| `--split-chunk-tokens`                 | 512    | [512, `--max-input-token-len`]   | Split Fuse                              | Split Fuse 的分块 token 大小。                                                                                                                                                                                                                                                   |
| `--split-start-batch-size`             | 16     | [0, `--max-batch-size`]          | Split Fuse                              | 启动 Split Fuse 时的批大小阈值。                                                                                                                                                                                                                                                 |
| `--enable-memory-decoding`             |        |                                  | Speculative Decoding / Memory Decoding  | 启用内存解码推测。若需显式禁用，请使用 `--no-enable-memory-decoding`。                                                                                                                                                                                                            |
| `--memory-decoding-length`             | 16     | [1, 16]                          | Speculative Decoding / Memory Decoding  | 内存解码推测的长度。                                                                                                                                                                                                                                                             |
| `--memory-decoding-dynamic-algo`       |        |                                  | Speculative Decoding / Memory Decoding  | 启用内存解码推测的动态算法。                                                                                                                                                                                                                                                     |
| `--enable-lookahead`                   |        |                                  | Speculative Decoding / Lookahead        | 启用前瞻推测。若需显式禁用，请使用 `--no-enable-lookahead`。                                                                                                                                                                                                                      |
| `--lookahead-level`                    | 4      | [3, 16]                          | Speculative Decoding / Lookahead        | 前瞻推测的层级。                                                                                                                                                                                                                                                                 |
| `--lookahead-window`                   | 5      | [1, 16]                          | Speculative Decoding / Lookahead        | 前瞻推测的窗口大小。                                                                                                                                                                                                                                                             |
| `--lookahead-guess-set-size`           | 5      | [1, 16]                          | Speculative Decoding / Lookahead        | 前瞻推测的猜测集合大小。                                                                                                                                                                                                                                                         |
| `--enable-multi-token-prediction`      |        |                                  | Multi-Token Prediction                  | 启用多 Token 预测。若需显式禁用，请使用 `--no-enable-multi-token-prediction`。                                                                                                                                                                                                   |
| `--multi-token-prediction-tokens`      | 1      | (0, \*)                          | Multi-Token Prediction                  | 多 Token 预测的 token 数。仅在启用了 `--enable-multi-token-prediction` 时生效。                                                                                                                                                                                                  |
| `--enable-prefix-caching`              |        |                                  | Prefix Caching                          | 启用前缀缓存。若需显式禁用，请使用 `--no-enable-prefix-caching`。                                                                                                                                                                                                                |
| `--pipeline-parallel-size`, `-pp`      | 1      | (0, \*)                          | Parallelism                             | 流水线并行组的数量。                                                                                                                                                                                                                                                             |
| `--data-parallel-size`, `-dp`          | -1     |                                  | Parallelism                             | Attention 层的数据并行组数量。`-1` 表示禁用数据并行，否则必须为 2 的幂。                                                                                                                                                                                                          |
| `--tensor-parallel-size`, `-tp`        | -1     |                                  | Parallelism                             | Attention 层的张量并行组数量。`-1` 表示使用全局规模作为张量并行大小，否则必须为 2 的幂。                                                                                                                                                                                          |
| `--sequence-parallel-size`, `-sp`      | -1     |                                  | Parallelism                             | MLP 层的序列并行组数量。`-1` 表示禁用序列并行，否则必须为 2 的幂。                                                                                                                                                                                                                |
| `--moe-expert-parallel-size`, `-moe-ep`| -1     |                                  | Parallelism                             | 专家并行组数量。`-1` 表示禁用 MoE 专家并行，否则必须为 2 的幂。                                                                                                                                                                                                                   |
| `--moe-tensor-parallel-size`, `-moe-tp`| -1     |                                  | Parallelism                             | MoE MLP 层的张量并行组数量。`-1` 表示使用全局规模作为 MoE 张量并行大小，否则必须为 2 的幂。                                                                                                                                                                                       |
| `--enable-buffer-response`             |        |                                  | Buffer Response                         | 启用缓冲响应。若需显式禁用，请使用 `--no-enable-buffer-response`。                                                                                                                                                                                                               |
| `--prefill-expected-time-ms`           |        |                                  | Buffer Response                         | 期望的首 token 延迟（TTFT）SLO，单位毫秒。                                                                                                                                                                                                                                       |
| `--decode-expected-time-ms`            |        |                                  | Buffer Response                         | 期望的输出 token 时延（TPOT）SLO，单位毫秒。                                                                                                                                                                                                                                     |

!!! Note

    GPUStack 允许用户在模型部署时注入自定义环境变量，但某些变量可能与 GPUStack 的管理机制相冲突。因此，GPUStack 会覆盖/阻止这些变量。请对照模型实例日志的输出核对是否符合你的预期。