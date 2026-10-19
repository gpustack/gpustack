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
