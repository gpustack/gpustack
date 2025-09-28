# 兼容性检查

在部署模型之前，GPUStack 会执行一次兼容性检查。此检查将提供模型与当前 GPUStack 环境的兼容性详细信息。将执行以下兼容性检查：

## 推理后端兼容性

检查所选推理后端是否与当前环境兼容，包括操作系统、GPU 和架构。

## 模型兼容性

判断所选推理后端是否支持该模型。这包括检查支持的模型格式与架构（例如，`LlamaForCausalLM`、`Qwen3ForCausalLM` 等）。此检查基于内置推理后端及其支持的模型。如果使用自定义后端版本，则会跳过该检查。

## 可调度性检查

评估该模型是否能在当前环境中被调度。这包括验证可用资源（如内存和显存）以及已配置的调度规则。

### 调度规则

调度规则（包括工作节点选择器、GPU 选择器和调度策略）用于确定模型是否能在当前环境中被调度。

### 资源检查

资源检查确保部署该模型所需的系统资源充足。GPUStack 会估算所需资源，并与环境中的可用资源进行比较。估算通过以下方法进行：

1. 对于 GGUF 模型：GPUStack 使用 [GGUF 解析器](https://github.com/gpustack/gguf-parser-go) 来估算模型的资源需求。
2. 对于其他模型：GPUStack 使用以下公式估算显存占用：

$$
\text{VRAM} = \text{WEIGHT\_SIZE} \times 1.2 + \text{FRAMEWORK\_FOOTPRINT}
$$

- `WEIGHT_SIZE` 指模型权重的字节大小。
- `FRAMEWORK_FOOTPRINT` 是表示框架内存开销的常量。例如，vLLM 可能会为了 CUDA graphs 占用数 GB 的显存。
- 1.2 倍系数是基于经验的估算。更多细节请参阅[此说明](https://blog.eleuther.ai/transformer-math/#total-inference-memory)。

该公式仅提供粗略估算，可能并不适用于所有模型。通常它反映的是所需显存的下限估计。如果估算不足，用户可以通过手动选择工作节点和 GPU，或调整后端的高级参数来进行更精细的调度。例如，在 vLLM 中，用户可以通过指定 `--tensor-parallel-size` 和 `--pipeline-parallel-size` 来控制该模型的 GPU 分配。