---
translation_status: translated
---

# 内置推理后端 { #built-in-inference-backends }

GPUStack 支持以下推理后端：

- [vLLM](#vllm)
- [SGLang](#sglang)
- [MindIE](#mindie)
- [VoxBox](#voxbox)

开始部署模型时，UI 会根据以下条件自动预选后端：

- 如果模型是已知的 `Text-to-Speech` 或 `Speech-to-Text` 模型，使用 `VoxBox`。
- 如果模型是已知的 `Image` 模型，使用 `SGLang`。
- 其他情况使用 `vLLM`。

此预选仅用于填充部署表单中的默认值。实际部署使用的后端，以用户提交部署请求时明确选择的后端为准。用户可以从上方支持的后端列表中选择。

!!! Note

    对于所有支持的推理后端，我们都在 [DockerHub](https://hub.docker.com/r/gpustack/runner) 上提供预构建 Docker 镜像。用户部署模型时，系统会自动拉取并运行对应镜像。

## vLLM { #vllm }

[vLLM](https://github.com/vllm-project/vllm) 是具有高吞吐量和高内存效率的 LLM 推理后端，是生产环境运行 LLM 的常见选择。

vLLM 无缝支持大多数先进开源模型，包括：

- 类 Transformer 的 LLM（例如 `Qwen3`）
- 混合专家 LLM（例如 `DeepSeek V3/R1`）
- 多模态 LLM（例如 `Qwen3-VL`）
- Embedding 模型（例如 `Qwen3-Embedding`）
- Reranker 模型（例如 `Qwen3-Reranker`）

默认情况下，GPUStack 根据模型元数据估算模型实例所需的 VRAM。

可以根据需要自定义参数。以下 vLLM 参数可能有用：

- `--gpu-memory-utilization`（默认：0.9）：供模型实例使用的 GPU 内存比例。
- `--max-model-len`：模型上下文长度。对于长上下文模型，GPUStack 会自动将此参数设为 `8192`，以简化模型部署，尤其适用于资源受限环境。可以按需自定义。
- `--tensor-parallel-size`：张量并行副本数。默认情况下，GPUStack 根据可用 GPU 资源和模型内存需求估算设置此参数。可以按需自定义。

更多详情参见 [vLLM CLI 参考](https://docs.vllm.ai/en/stable/cli/serve/)。

### 支持的模型 { #supported-models }

支持的模型参见 vLLM [文档](https://docs.vllm.ai/en/stable/models/supported_models.html)。

### 支持的功能 { #supported-features }

#### 全模态支持 { #omni-modal-support }

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) 是面向全模态推理的 vLLM 扩展，支持：

- **全模态模型**：同时处理文本、音频、图像和视频（例如 `Qwen3-Omni`）。
- **图像任务**：图像生成和编辑（例如 `Z-Image-Turbo`）。
- **视频任务**：视频生成和编辑（例如 `Wan2.2`）。
- **音频任务**：语音合成、声音克隆等（例如 `Qwen3-TTS`）。

GPUStack 集成 vLLM-Omni，为全模态模型的部署和管理提供流畅体验。通过 vLLM 后端部署模型时，GPUStack 会根据元数据自动判断是否为全模态模型，并为 vLLM-Omni 设置所需参数。

#### 跨 Worker 分布式推理（实验性） { #distributed-inference-across-workers-experimental }

单个 Worker 的 VRAM 或计算资源不足以承载模型时，vLLM 可以将一个模型实例拆分到多个 Worker 上协同推理。GPUStack 将这些跨节点进程作为单个模型实例编排，对外仍只暴露一个兼容 OpenAI 的端点。

部署模型时勾选 `Allow Distributed Inference Across Workers` 即可启用。

GPUStack 为 vLLM 分布式推理提供两种执行路径，通过 vLLM 的 `--distributed-executor-backend` 参数选择。通常无需手动设置，GPUStack 会根据容器镜像和 vLLM 版本自动选择。

##### Ray 后端 { #ray-backend }

- **使用场景**：GPUStack 官方 runner 镜像上的默认路径，runner 镜像始终内置 Ray，不受 vLLM 版本影响。如果在后端参数中显式将 `--distributed-executor-backend` 设置为 `ray` 或其他非 `mp` 值（例如 `external_launcher`），也会使用此路径。
- **工作方式**：GPUStack 在每个参与节点上启动 [Ray](https://ray.io) sidecar 进程，组成集群。主节点运行 vLLM 主进程并暴露 API 服务，Ray 负责协调跨节点任务调度。

##### MultiProcessing 后端 { #multiprocessing-backend }

- **使用场景**：在后端参数中显式设置 `--distributed-executor-backend=mp` 时；或在 vLLM ≥ `0.18.0` 上运行自定义镜像（后端版本带 `-custom` 后缀）时。vLLM 在 `0.18.0` 中移除了默认依赖里的 Ray，因此自定义镜像可能不包含 Ray，GPUStack 会回退到 `mp` 以避免启动失败。
- **工作方式**：直接使用 vLLM 基于 multiprocessing 的多节点执行，无需 Ray sidecar。GPUStack 根据集群布局注入 `--nnodes`、`--node-rank`、`--master-addr` 和 `--master-port` 等拓扑参数。
- **无头从节点**：调度为从节点的实例会自动获得 `--headless`，**仅参与分布式计算，不启动 OpenAI API 服务**。只有主节点暴露 HTTP 端点；所有外部请求均经主节点进入，再分发到从节点。从节点实例日志中出现 `--headless` 属于预期行为，这也是从节点实例不占用 API 端口的原因。
- **拓扑形式**：GPUStack 根据并行配置（TP / PP / DP），自动选择 `dp_only`、`mp_only` 或 `nested` 参数布局。

!!! warning "已知限制"

    1. 所有参与 Worker 必须能通过相同路径访问模型文件。请使用共享文件系统，或将模型文件下载到所有参与 Worker 的相同路径。
    2. Ray 模式下，每个 Worker 同一时间只能分配给一个分布式 vLLM 模型实例。

满足以下条件时支持自动调度：

- 参与 Worker 的 GPU 数量相同。
- Worker 内所有 GPU 均满足 gpu_memory_utilization（默认 0.9）要求。
- 使用张量并行时，根据注意力头数量划分 GPU。
- 总 VRAM 申领量大于估算的 VRAM 申领量。

不满足上述条件时，模型实例不会自动调度。但可以在模型配置中选择期望的 Worker/GPU 进行手动调度。

### 参数参考 { #parameters-reference }

vLLM 支持的完整参数列表参见[此处](https://docs.vllm.ai/en/stable/cli/serve/)。

## SGLang { #sglang }

[SGLang](https://github.com/sgl-project/sglang) 是用于大语言模型和视觉语言模型的高性能服务框架。

它旨在从单 GPU 到大规模分布式集群的各种环境中，提供低延迟、高吞吐量的推理。

默认情况下，GPUStack 根据模型元数据估算模型实例所需的 VRAM。

需要时，GPUStack 还会为长上下文模型自动设置若干参数。常用 SGLang 参数包括：

- `--mem-fraction-static`（默认：`0.9`）：每个 GPU 可分配的 VRAM 比例。调度器使用此值进行资源匹配和候选选择。可通过模型的 `backend_parameters` 覆盖。
- `--context-length`：模型上下文长度。对于长上下文模型，如果自动估算的上下文长度超过设备能力，GPUStack 会将此参数设为 `8192`，以简化资源受限环境中的部署。可按需自定义。
- `--tp-size`：张量并行大小。未显式提供时，GPUStack 根据所选 GPU 推导并设置此参数。
- `--pp-size`：流水线并行大小。多节点部署中，GPUStack 根据模型和集群配置确定 `--tp-size` 与 `--pp-size` 的组合。
- 多节点参数：`--nnodes`、`--node-rank`、`--dist-init-addr`。启用分布式推理时，GPUStack 注入这些参数以初始化多节点通信。

更多详情参见 [SGLang 文档](https://docs.sglang.ai/index.html)。

### 支持的模型 { #supported-models_1 }

支持的模型参见 SGLang [文档](https://docs.sglang.ai/supported_models/generative_models.html)。

SGLang 也支持图像模型。我们验证过的模型包括 Qwen-Image、Flux 1 Dev。更多信息参见 [SGLang Diffusion 文档](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)。

### 支持的功能 { #supported-features_1 }

#### 跨 Worker 分布式推理（实验性） { #distributed-inference-across-workers-experimental_1 }

可以在 GPUStack 中启用跨多个 Worker 的 SGLang 分布式推理。

!!! warning "已知限制"

    1. 所有参与节点必须运行 Linux，镜像/环境的 Python 版本和通信库（例如 NCCL）应兼容。
    2. 所有节点必须能通过相同路径访问模型文件。请使用共享文件系统，或将模型下载到各节点的相同路径。

自动调度的候选选择考虑以下条件：

- 参与 GPU 满足 `--mem-fraction-static` 要求（默认 0.9）。
- 在单节点多 GPU 或多节点多 GPU 场景中，有效可分配 VRAM 总量必须满足模型需求。
- 必须满足模型并行要求（例如注意力头总数可被张量并行大小整除），否则候选会被拒绝。

不满足上述条件时，仍可通过在配置中选择 Worker/GPU 手动调度模型实例，但可能会提示资源超额分配风险。

#### 扩散模型（实验性） { #diffusion-models-experimental }

SGLang Diffusion 是专为扩散模型设计的高性能推理引擎，旨在加速图像和视频的生成过程。
通过 SGLang 的并行处理技术和优化内核，相比主流基线（例如 Hugging Face Diffusers）可实现最高 1.2 倍的生成速度。

更多详情参见 [SGLang Diffusion](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)。

#### 其他高级功能 { #other-advanced-features }

还提供以下高级功能：

- [推测解码](https://docs.sglang.ai/advanced_features/speculative_decoding.html)
- [分层 KV 缓存](https://docs.sglang.ai/advanced_features/hicache.html)

使用说明参见官方文档。

### 参数参考 { #parameters-reference_1 }

SGLang 支持的完整参数列表参见[此处](https://docs.sglang.ai/advanced_features/server_arguments.html)。

## MindIE { #mindie }

[MindIE](https://www.hiascend.com/en/software/mindie) 是运行在 [昇腾硬件](https://www.hiascend.com/en/hardware/product)上的高性能推理服务。

### 支持的模型 { #supported-models_2 }

MindIE 支持[此处](https://www.hiascend.com/software/mindie/modellist)列出的多种模型。

在 GPUStack 中，支持[大语言模型（LLM）](https://www.hiascend.com/software/mindie/modellist)和[多模态语言模型（VLM）](https://www.hiascend.com/software/mindie/modellist)。

目前尚不支持 _Embedding 模型_ 和 _多模态生成模型_。

### 支持的功能 { #supported-features_2 }

MindIE 提供多种功能，参见[此处](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0001.html)。

目前，GPUStack 支持其中一部分能力，包括
[量化](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0279.html)、
[扩展上下文长度](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0295.html)、
[分布式推理](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0296.html)、
[混合专家（MoE）](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0297.html)、
[Split Fuse](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0300.html)、
[推测解码](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0301.html)、
[多 token 预测](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0500.html)、
[前缀缓存](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0302.html)、
[函数调用](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0303.html)、
[多模态理解](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0304.html)、
[多头潜在注意力（MLA）](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0305.html)、
[张量并行](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0511.html)、
[上下文并行](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0508.html)、
[序列并行](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0517.html)、
[专家并行](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0509.html)、
[数据并行](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0424.html)、
[缓冲响应（自昇腾 MindIE 2.0.RC1 起）](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0425.html)。
[KV Pool（自昇腾 MindIE 2.2.RC1 起）](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0538.html)。

!!! note

    1. 量化需要特定权重，并且必须调整模型的 `config.json`。请按[参考指南](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0279.html)准备正确权重。

    2. 某些功能互斥，使用时请注意。例如，启用前缀缓存后，无法使用扩展上下文长度功能。

### 参数参考 { #parameters-reference_2 }

MindIE 提供可配置的[参数](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_service0285.html)和[环境变量](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0416.html)。

为避免直接配置 JSON，GPUStack 提供以下命令行参数。

| 参数 | 默认值 | 范围 | 作用域 | 描述 |
| --- | --- | --- | --- | --- |
| `--log-level` | Info |  | 日志配置 | MindIE 日志级别。可选值：`Verbose`、`Info`、`Warning`、`Warn`、`Error`、`Debug`。 |
| `--max-link-num` | 1000 | [1, 1000] | 服务器配置 | 最大并行请求数。 |
| `--token-timeout` | 60 | [1, 3600] | 服务器配置 | 单个 token 生成的超时时间，单位为秒。 |
| `--e2e-timeout` | 60 | [1, 3600] | 服务器配置 | 端到端（从接受请求到推理停止）的超时时间，单位为秒。 |
| `--openai-support` | vllm |  | 服务器配置 | OpenAI API 兼容模式。 |
| `--kv-pool-config` |  |  | 后端配置 | JSON 格式的 KV pool 配置。例如：`{"backend":"<KV pool backend name>", "configPath":"/path/to/your/config/file"}`。 |
| `--max-seq-len` | 8192 | (0, \*) | 模型部署配置 | 模型上下文长度。未指定时，从模型配置推导。 |
| `--max-input-token-len` |  | (0, `--max-seq-len`] | 模型部署配置 | 最大输入 token 长度。未指定时，从 `min(--max-seq-len, --max-prefill-tokens)` 推导。 |
| `--truncation` |  |  | 模型部署配置 | 输入 token 长度超过 `--max-input-token-len` 与 `--max-seq-len` - 1 中的较小值时，截断输入。 |
| `--cpu-mem-size` | 0 | [0, \*) | 模型配置 | CPU 交换空间大小，单位为 GiB。指定 `--max-preempt-count` 时生效。 |
| `--npu-memory-fraction` (env: `NPU_MEMORY_FRACTION`) | 0.9 | (0, 1] | 模型配置 | 供模型执行器使用的 NPU 内存比例（0 到 1）。例如：`0.5` 表示 50% 内存利用率。 |
| `--trust-remote-code` |  |  | 模型配置 | 信任远程代码（用于加载模型）。 |
| `--models` |  |  | 模型配置 | JSON 格式的模型配置，用于某些特定配置，例如专家并行实现方式、张量并行 LM Header/Output Attention 拆分。 |
| `--async-scheduler-wait-time` | 120 | [1, 3600] | 模型配置 | 等待异步调度器启动的时间，单位为秒。 |
| `--cache-block-size` | 128 |  | 调度配置 | KV 缓存块大小，必须是 2 的幂。 |
| `--max-prefill-batch-size` | 50 | [1, `--max-batch-size`] | 调度配置 | 预填充阶段批处理的最大请求数，必须小于 `--max-batch-size`。 |
| `--max-prefill-tokens  ` |  | [1, `--max-seq-len`] | 调度配置 | 每次预填充时，当前批次的全部输入 token 总数不能超过 `--max-prefill-tokens`。默认与 `--max-seq-len` 相同。 |
| `--prefill-time-ms-per-req` | 150 | [0, 1000] | 调度配置 | 预估每个请求的预填充时间（ms），用于在预填充阶段和解码阶段之间决策。 |
| `--prefill-policy-type` | 0 |  | 调度配置 | 预填充阶段策略：<br> `0`：FCFS（先到先服务）。<br> `1`：STATE（与 FCFS 相同）。<br> `2`：PRIORITY（优先级队列）。<br> `3`：MLFQ（多级反馈队列）。 |
| `--max-batch-size` | 200 | [1, 5000] | 调度配置 | 解码阶段批处理的最大请求数。 |
| `--max-iter-times` |  | [1, `--max-seq-len`] | 调度配置 | 解码阶段的最大迭代次数，默认与 `--max-seq-len` 相同。 |
| `--decode-time-ms-per-req` | 50 | [0, 1000] | 调度配置 | 预估每个请求的解码时间（ms），与 `--prefill-time-ms-per-req` 一起用于批次选择。 |
| `--decode-policy-type` | 0 |  | 调度配置 | 解码阶段策略：<br> `0`：FCFS <br> `1`：STATE（优先处理被抢占或换出的请求）<br> `2`：PRIORITY <br> `3`：MLFQ |
| `--max-preempt-count` | 0 | [0, `--max-batch-size`] | 调度配置 | 解码期间允许被抢占的最大请求数，必须小于 `--max-batch-size`。 |
| `--support-select-batch` |  |  | 调度配置 | 启用批次选择，根据 `--prefill-time-ms-per-req` 和 `--decode-time-ms-per-req` 确定执行优先级。使用 `--no-support-select-batch` 显式禁用。 |
| `--max-queue-delay-microseconds` | 5000 | [500, 1000000] | 调度配置 | 队列最大等待时间，单位为微秒。 |
| `--max-first-token-wait-time` | 2500 | [0, 3600000] | 调度配置 | 等待首个 token 生成的最大毫秒数。 |
| `--override-generation-config` |  |  |  | 以 JSON 格式覆盖或设置生成配置，例如：`{"temperature": 0.5}`。它会合并到模型结构的 `generation_config.json` 中。 |
| `--enforce-eager` |  |  |  | 以 eager 模式执行算子。 |
| `--no-metrics` |  |  |  | 禁止通过 `/metrics` 端点暴露指标。 |
| `--dtype` | auto |  |  | 模型权重和激活值的数据类型。<br> `auto`：使用模型配置的默认数据类型。<br> `half`/`float16`：用于 FP16。<br> `bfloat16`：用于 BF16。<br> `float`/`float32`：用于 FP32。 |
| `--rope-scaling` |  |  | 扩展上下文长度 | JSON 格式的 RoPE 缩放配置，例如：`{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}`。它会合并到模型结构的 `config.json` 中。 |
| `--rope-theta` |  |  | 扩展上下文长度 | RoPE theta 配置，会合并到模型结构的 `config.json` 中。 |
| `--enable-split` |  |  | Split Fuse | 启用 Split Fuse，类似分块预填充。使用 `--no-enable-split` 显式禁用。 |
| `--policy-type` | 0 |  | Split Fuse | Split Fuse 策略。<br> `0`：FCFS，先到先服务。<br> `4`：SJF，最短任务优先。<br> `5`：LJF，最长任务优先。<br> `6`：Skip-Join MLFQ，跳跃加入多级反馈队列。<br> `7`：SJF-MLFQ，最短任务优先与多级反馈队列。 |
| `--split-chunk-tokens` | 512 | [1, 8192] | Split Fuse | Split Fuse 批处理的 token 大小，必须为 16 的倍数。 |
| `--split-start-batch-size` | 16 | [0, `--max-batch-size`] | Split Fuse | Split Fuse 开始拆分的批次大小。 |
| `--enable-memory-decoding` |  |  | 推测解码 / Memory Decoding | 启用 memory decoding 推测。使用 `--no-enable-memory-decoding` 显式禁用。 |
| `--memory-decoding-length` | 16 | [1, 16] | 推测解码 / Memory Decoding | memory decoding 推测长度。 |
| `--memory-decoding-dynamic-algo` |  |  | 推测解码 / Memory Decoding | 为 memory decoding 推测启用动态算法。 |
| `--enable-lookahead` |  |  | 推测解码 / Lookahead | 启用 lookahead 推测。使用 `--no-enable-lookahead` 显式禁用。 |
| `--lookahead-level` | 4 | [3, 16] | 推测解码 / Lookahead | lookahead 推测层级。 |
| `--lookahead-window` | 5 | [1, 16] | 推测解码 / Lookahead | lookahead 推测窗口大小。 |
| `--lookahead-guess-set-size` | 5 | [1, 16] | 推测解码 / Lookahead | lookahead 推测猜测集合大小。 |
| `--enable-multi-token-prediction` |  |  | 多 token 预测 | 启用多 token 预测。使用 `--no-enable-multi-token-prediction` 显式禁用。 |
| `--multi-token-prediction-tokens` | 1 | (0, \*) | 多 token 预测 | 多 token 预测的 token 数量。仅在启用 `--enable-multi-token-prediction` 时生效。 |
| `--enable-prefix-caching` |  |  | 前缀缓存 | 启用前缀缓存。使用 `--no-enable-prefix-caching` 显式禁用。 |
| `--pipeline-parallel-size`, `-pp` | 1 | (0, \*) | 并行 | 流水线并行组数量。 |
| `--data-parallel-size`, `-dp` | -1 |  | 并行 | Attention 层的数据并行组数量。`-1` 表示禁用数据并行，否则必须是 2 的幂。 |
| `--context-parallel-size`, `-cp` | -1 |  | 并行 | Attention 层的上下文并行组数量。`-1` 表示禁用上下文并行，否则必须是 2 的幂。 |
| `--tensor-parallel-size`, `-tp` | -1 |  | 并行 | Attention 层的张量并行组数量。`-1` 表示使用 world size 作为张量并行大小，否则必须是 2 的幂。 |
| `--sequence-parallel-size`, `-sp` | -1 | `--tensor-parallel-size` | 并行 | MLP 层的序列并行组数量。`-1` 表示禁用序列并行，否则必须是 2 的幂。 |
| `--moe-expert-parallel-size`, `-moe-ep` | -1 |  | 并行 | 专家并行组数量。`-1` 表示禁用 MoE 专家并行，否则必须是 2 的幂。 |
| `--moe-tensor-parallel-size`, `-moe-tp` | -1 |  | 并行 | MoE MLP 层的张量并行组数量。`-1` 表示使用 world size 作为 MoE 张量并行大小，否则必须是 2 的幂。 |
| `--enable-buffer-response` |  |  | 缓冲响应 | 启用缓冲响应。使用 `--no-enable-buffer-response` 显式禁用。 |
| `--prefill-expected-time-ms` |  |  | 缓冲响应 | 首 token 时间（TTFT）的预期延迟（SLO），单位为毫秒。 |
| `--decode-expected-time-ms` |  |  | 缓冲响应 | 每输出 token 时间（TPOT）的预期延迟（SLO），单位为毫秒。 |

!!! note

    GPUStack 允许用户在模型部署时注入自定义环境变量，但某些变量可能与 GPUStack 的管理逻辑冲突。

    因此，GPUStack 会覆盖或阻止这些变量。请对照模型实例日志输出，确认其符合预期。

## VoxBox { #voxbox }

[VoxBox](https://github.com/gpustack/vox-box) 是为部署文本转语音和语音转文本模型设计的推理引擎，还提供与 OpenAI 音频 API 完全兼容的 API。

### 支持的模型 { #supported-models_3 }

| 模型                           | 类型           | 链接                                                                                                                                                                | 支持的平台 |
| ------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| Faster-whisper-large-v3         | 语音转文本 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v3)                 | AMD64,ARM64         |
| Faster-whisper-large-v2         | 语音转文本 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v2), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v2)                 | AMD64,ARM64         |
| Faster-whisper-large-v1         | 语音转文本 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v1), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v1)                 | AMD64,ARM64         |
| Faster-whisper-medium           | 语音转文本 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-medium)                     | AMD64,ARM64         |
| Faster-whisper-medium.en        | 语音转文本 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-medium.en)               | AMD64,ARM64         |
| Faster-whisper-small            | 语音转文本 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-small)                       | AMD64,ARM64         |
| Faster-whisper-small.en         | 语音转文本 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-small.en)                 | AMD64,ARM64         |
| Faster-distil-whisper-large-v3  | 语音转文本 | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v3), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-large-v3)   | AMD64,ARM64         |
| Faster-distil-whisper-large-v2  | 语音转文本 | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v2), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-large-v2)   | AMD64,ARM64         |
| Faster-distil-whisper-medium.en | 语音转文本 | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-medium.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-medium.en) | AMD64,ARM64         |
| Faster-whisper-tiny             | 语音转文本 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-tiny)                         | AMD64,ARM64         |
| Faster-whisper-tiny.en          | 语音转文本 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-tiny.en)                   | AMD64,ARM64         |
| CosyVoice-300M-Instruct         | 文本转语音 | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M-Instruct), [ModelScope](https://modelscope.cn/models/gpustack/CosyVoice-300M-Instruct)                | AMD64               |
| CosyVoice-300M-SFT              | 文本转语音 | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M-SFT), [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-SFT)                               | AMD64               |
| CosyVoice-300M                  | 文本转语音 | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M), [ModelScope](https://modelscope.cn/models/gpustack/CosyVoice-300M)                                  | AMD64               |
| CosyVoice-300M-25Hz             | 文本转语音 | [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-25Hz)                                                                                                  | AMD64               |
| CosyVoice2-0.5B                 | 文本转语音 | [Hugging Face](https://huggingface.co/gpustack/CosyVoice2-0.5B), [ModelScope](https://modelscope.cn/models/iic/CosyVoice2-0.5B)                                     | AMD64               |
| Dia-1.6B                        | 文本转语音 | [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B), [ModelScope](https://modelscope.cn/models/nari-labs/Dia-1.6B)                                            | AMD64               |

### 支持的功能 { #supported-features_3 }

#### 允许 GPU/CPU 卸载 { #allow-gpucpu-offloading }

VoxBox 支持将模型部署到 NVIDIA GPU。如果 GPU 资源不足，会自动将模型部署到 CPU。

!!! note "不支持流式输出"

    VoxBox 不支持文本转语音（`/v1/audio/speech`）或语音转文本（`/v1/audio/transcriptions`）的流式输出。要使用流式音频 API，请改为在 `vLLM` 后端上部署兼容 vLLM 的音频模型。
