---
translation_status: translated
---

# 兼容性检查 { #compatibility-check }

GPUStack 会在模型部署前执行兼容性检查，提供模型与当前 GPUStack 环境兼容情况的详细信息。检查包括以下内容：

## 推理后端兼容性 { #inference-backend-compatibility }

检查所选推理后端是否与当前环境兼容，包括操作系统、GPU 和架构。

## 模型兼容性 { #model-compatibility }

判断所选推理后端是否支持该模型，包括检查模型格式和架构（例如 `LlamaForCausalLM`、`Qwen3ForCausalLM` 等）。此检查基于内置推理后端及其支持的模型。如果使用自定义后端或后端版本，则跳过此检查。

## 可调度性检查 { #schedulability-check }

评估模型能否在当前环境中调度，包括验证 RAM 和 VRAM 等可用资源，以及已配置的调度规则。

### 调度规则 { #scheduling-rules }

调度规则（包括 Worker 选择器、GPU 选择器和调度策略）用于判断模型能否在当前环境中调度。

### 资源检查 { #resource-check }

资源检查确保系统有足够的资源来部署模型。GPUStack 会估算所需资源，并与环境中的可用资源进行比较。估算采用以下方法：

1. **对于 GGUF 模型**：GPUStack 使用 [GGUF parser](https://github.com/gpustack/gguf-parser-go) 估算模型的资源需求。
2. **对于其他模型**：GPUStack 使用以下公式估算 VRAM 用量：

$$
\text{VRAM} = \text{WEIGHT\_SIZE} \times 1.2 + \text{FRAMEWORK\_FOOTPRINT}
$$

- `WEIGHT_SIZE` 指模型权重的大小，单位为字节。
- `FRAMEWORK_FOOTPRINT` 是表示框架内存开销的常量。例如，vLLM 可能会为 CUDA graphs 占用数 GB 的 VRAM。
- 1.2 倍系数是经验估算值。更多详情参见[此说明](https://blog.eleuther.ai/transformer-math/#total-inference-memory)。

此公式仅提供粗略估算，不一定适用于所有模型。通常，它反映的是所需 VRAM 的下限估算。如果估算不足，用户可以手动选择 Worker 和 GPU，或调整高级后端参数，以进行细粒度调度。例如，使用 vLLM 时，可指定 `--tensor-parallel-size` 和 `--pipeline-parallel-size` 来控制模型的 GPU 分配。

#### 卸载查找表 { #offloaded-lookup-tables }

!!! note

    查找表自动估算目前仅覆盖已适配的 vLLM 和 SGLang 架构、格式和卸载配置。原生 MindIE、custom 后端以及尚未适配的 vLLM Ascend 路径不在自动估算范围内。如果自动估算不符合部署预期，可以通过[自定义资源需求](#custom-resource-requirements)进行调整。

对于受支持且将查找表卸载到 CPU 内存的模型（如 Engram 和 PLE），GPUStack 根据模型配置和精度元数据估算查找表大小。应用上述 VRAM 公式前，会从 `WEIGHT_SIZE` 中减去卸载的字节数，并在每个参与部署的 Worker 上预留查找表所需的 RAM，再叠加本地扩展 KV 缓存的 RAM 需求。

RAM 估算假设同一 Worker 上的数据并行副本共享查找表存储，并计入并行布局所需的独立副本，不包含额外余量，实际内存用量可能更高。如果无法确定查找表大小或卸载行为，GPUStack 会保留按完整权重计算的 VRAM 估算。

#### 自定义资源需求 { #custom-resource-requirements }

如果自动估算与部署的实际需求不符，可以通过模型的[资源需求环境变量](../environment-variables.md#model-deployment-configuration)，以字节为单位自定义 RAM 和 VRAM 需求。RAM 声明值会替代每个 Worker 的总量估算，包括查找表和本地缓存内存。这些声明用于调度，不会配置后端的实际内存用量。RAM 和 VRAM 可以分别声明。未声明的资源继续使用对应后端的估算值。

例如，声明每个 Worker 需要 64 GiB RAM、模型共需 32 GiB VRAM，可添加以下模型环境变量（`1 GiB = 1024^3 bytes`）。VRAM 值表示模型在所用 GPU 上的总估算需求，而非每张 GPU 的预留量。

```text
GPUSTACK_MODEL_RAM_CLAIM=68719476736
GPUSTACK_MODEL_VRAM_CLAIM=34359738368
```
