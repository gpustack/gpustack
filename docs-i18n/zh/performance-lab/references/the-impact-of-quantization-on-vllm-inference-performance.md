---
translation_status: translated
---

# 量化对 vLLM 推理性能的影响 { #the-impact-of-quantization-on-vllm-inference-performance }

量化使用 8-bit 整数（int8）等低精度数据类型表示权重和激活值，替代通常使用的 32-bit 浮点数（float32），以降低推理的计算和内存成本。

本文旨在评估 vLLM 中常用的[量化技术](https://docs.vllm.ai/en/latest/features/quantization/index.html)及其对推理性能的影响，重点关注高并发场景下的吞吐量。


## 结论 { #conclusions }

1. **权重与激活值量化**（例如 FP8）能够在质量损失很小的情况下显著提升性能，推荐用于大多数场景。静态量化的吞吐量高于动态量化，准确率取决于校准数据：

    - **动态 FP8**：+14.8% TPS，-14.1% TTFT，-21.8% TPOT
    - **静态 FP8**：+26.7% TPS，-20.4% TTFT，-32.4% TPOT

2. 在显存受限的场景中，**大多数量化方法**都能显著提高吞吐量，实验中观察到的提升最高为 +46%。

3. 在显存不受限时，**仅权重量化**会因反量化开销而使 TPS、TTFT 和 TPOT 表现下降。

4. **在仅权重量化方法中**，AWQ 和 GPTQ 的推理性能最佳；bitsandbytes 和 GGUF 存在性能较差或兼容性问题，因此不推荐使用。

5. 实验中，AWQ 和 GPTQ 的 **vLLM 默认内核**性能优于自定义内核实现。

6. 与其他优化技术相比，**KV-Cache 量化**带来的吞吐量提升相对有限。

## 技术背景 { #technical-background }

### 量化类型 { #quantization-type }

| 量化类型 | 说明 |
|-------------------|-------------|
| 仅权重量化 | 仅在训练后量化权重；激活值保持全精度 |
| 动态 | 预先量化权重；在推理过程中实时量化激活值 |
| 静态 | 使用有代表性的数据集校准后，提前量化权重和激活值 |
| 量化感知训练 | 在训练期间模拟量化，使模型适应较低精度 |

### 校准 { #calibration }

校准是量化过程中计算 float32 数值范围的步骤。对于权重，由于量化时已知实际范围，这一步比较简单。对于激活值，范围并不明确，因此有不同的处理方法：

1. **训练后动态量化**：在运行时实时计算每个激活值的范围。这种方法无需太多准备就能取得良好效果，但每次计算范围都会引入额外开销，因此可能比静态量化更慢。此外，部分硬件不支持这种方式。

2. **训练后静态量化**：在量化阶段提前计算每个激活值的范围，通常会将有代表性的数据输入模型并记录激活值。具体步骤如下：
   - 在激活值处设置观测器，记录其数值
   - 使用校准数据集执行一定次数的前向传播（约 200 个样例即可）
   - 根据选定的校准方法计算各项运算的数值范围

3. **量化感知训练**：在训练阶段计算每个激活值的范围，思路与训练后静态量化相同，但使用“伪量化”算子替代观测器。这些算子既会像观测器一样记录数值，也会模拟量化引入的误差，使模型适应该误差。

### KV Cache 量化 { #kv-cache-quantization }

KV cache 量化以更低精度存储键值缓存，降低推理时的内存占用，从而缓存更多 tokens 并提高吞吐量。vLLM 支持 FP8 数据类型（E4M3 和 E5M2），但不支持 INT8 KV cache。虽然已有研究探索 4-bit 或 2-bit KV cache 量化，但这些方法通常会造成明显的准确率下降，例如 MMLU 得分降低。

### 量化内核 { #quantization-kernels }

vLLM 为 AWQ、GPTQ 等量化方法提供多种量化内核实现。AWQ 量化默认使用官方 AWQ 内核，GPTQ 模型默认使用 ExLlamaV2 内核。此外，还可使用 Marlin、Machete 等优化内核，尤其能改善较大批大小下的性能。

### 推理质量下降 { #inference-quality-degradation }

量化可能导致推理质量下降。本文主要关注性能指标，同时提供以下参考数据，帮助评估对质量的影响：

Qwen3-8B 模型使用不同量化方法的基准测试结果：

| 量化 | CEVAL | MMLU | GSM8K | HUMANEVAL |
|-------------|-------|------|-------|-----------|
| BF16 | 79.27 | 74.78 | 87.79 | 63.41 |
| FP8-Static | 78.23 | 74.79 | 86.96 | 62.20 |
| FP8-Dynamic | 78.45 | 74.75 | 87.64 | 62.80 |
| INT8-Dynamic | 78.01 | 74.84 | 86.96 | 67.07 |
| INT4-GPTQ | 77.19 | 73.26 | 86.43 | 62.20 |
| INT4-AWQ | 76.15 | 73.59 | 86.96 | 63.41 |

更多详情请参阅 [AngelSlim 基准测试](https://github.com/Tencent/AngelSlim/tree/main?tab=readme-ov-file#-benchmark)。

## 实验环境 { #experimental-setup }

- **模型**: Qwen3-8B
- **硬件**: NVIDIA RTX 4090 24GB / A100 80GB / H100 80GB
- **vLLM 版本**: v0.9.2
- **数据集**: ShareGPT
- **基准测试脚本**:
```bash
# Prepare the ShareGPT dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Benchmark on ShareGPT dataset
vllm bench serve --model Qwen/Qwen3-8B --endpoint-type openai-chat --endpoint /v1/chat/completions --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000
```

## 实验结果 { #experimental-results }

### RTX 4090 { #rtx-4090 }

| 量化方法 | 吞吐量 (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |
|---------------------|-------------------|----------------|----------------|
| BF16                | 3869.3            | 15385.7        | 63.45          |
| AWQ                 | 5653.4 (+46.1%)   | 7913.4         | 87.7           |
| AWQ Marlin          | 5536.7 (+43.1%)   | 8133.8         | 90.42          |
| GPTQ (Int8)         | 4918.98 (+27.1%)  | 8415.70        | 96.07          |
| GPTQ Marlin         | 5025.82 (+29.9%)  | 8143.93        | 93.05          |

### A100 80GB { #a100-80gb }


| 量化方法 | 吞吐量 (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |
|---------------------|-------------------|----------------|----------------|
| BF16                | 10338.25          | 3412.85        | 200.02         |
| GPTQ-Marlin (Int4)  | 8146.73           | 10336.24       | 261.81         |
| GPTQ (Int4)         | 8129.27           | 10414.74       | 261.64         |
| AWQ                 | 9611.61           | 3950.64        | 249.06         |
| AWQ Marlin          | 8066.03           | 10506.70       | 264.33         |
| GPTQ Marlin (Int8)  | 7119.60           | 12359.22       | 309.37         |
| GPTQ (Int8)         | 7100.46           | 12380.82       | 309.34         |
| bitsandbytes        | 5916.34           | 9115.43        | 252.91         |
| GGUF (Q4_K_M)       | N/A               | N/A            | N/A            |

!!! note
    不支持 qwen3 架构的 GGUF 模型。

### H100 80GB { #h100-80gb }

| 量化方法 | 吞吐量 (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |
|---------------------|-------------------|----------------|----------------|
| FP8 静态          | 16452.52 (+26.7%) | 4116.87 (-20.4%) | 85.57 (-32.4%) |
| FP8 动态         | 15275.64 (+14.8%) | 4445.10 (-14.1%) | 98.94 (-21.8%) |
| Int4-W4A16          | 13605.46          | 5302.14         | 130.54         |
| BF16                | 13305.38          | 5172.78         | 126.55         |
| AWQ                 | 9756.06           | 8794.29         | 209.13         |
| GPTQ                | N/A               | N/A             | N/A            |

!!! note
    GPTQ 实现存在故障，参见 [vLLM issue](https://github.com/vllm-project/vllm/issues/20986)。

### KV Cache 量化 { #kv-cache-quantization_1 }

| 配置 | 吞吐量 (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |
|---------------|-------------------|----------------|----------------|
| BF16 (基准配置) | 13305.38 | 5172.78 | 126.55 |
| BF16 + KV Cache FP8 (E4M3) | 13513.19 (+1.6%) | 5688.39 | 103.21 |
