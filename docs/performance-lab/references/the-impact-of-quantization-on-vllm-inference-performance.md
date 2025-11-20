# The Impact of Quantization on vLLM Inference Performance

Quantization is a technique to reduce the computational and memory costs of running inference by representing the weights and activations with low-precision data types like 8-bit integer (int8) instead of the usual 32-bit floating point (float32).

The purpose of this documentation is to evaluate commonly used [quantization techniques](https://docs.vllm.ai/en/latest/features/quantization/index.html) in vLLM and how they affect inference performance, especially throughput in high concurrency scenarios.


## Conclusions

1. **Weight-activation quantization** (e.g., FP8) provides significant performance improvements with minimal quality loss and is recommended for most scenarios. Static quantization delivers higher throughput than dynamic, with accuracy dependent on calibration data:

    - **Dynamic FP8**: +14.8% TPS, -14.1% TTFT, -21.8% TPOT
    - **Static FP8**: +26.7% TPS, -20.4% TTFT, -32.4% TPOT

2. **Most quantization approaches** can substantially improve throughput in VRAM-constrained scenarios, with up to +46% improvement observed in experiments.

3. **Weight-only quantization** causes performance degradation in TPS, TTFT, and TPOT due to dequantization overhead when VRAM is not constrained.

4. **Among weight-only methods**, AWQ and GPTQ deliver the best inference performance, while bitsandbytes and GGUF show poor performance or compatibility issues and are not recommended.

5. **Default vLLM kernels** for AWQ and GPTQ outperform custom kernel implementations in experimental testing.

6. **KV-Cache quantization** provides relatively modest throughput improvements compared to other optimization techniques.

## Technical Background

### Quantization Type

| Quantization Type | Description |
|-------------------|-------------|
| Weight-only | Only the weights are quantized after training; activations remain full-precision |
| Dynamic | Weights are pre-quantized; activations are quantized on-the-fly during inference |
| Static | Weights and activations are quantized ahead of time after calibration with a representative dataset |
| Quantization-aware Training | Simulates quantization during training so the model adapts to reduced precision |

### Calibration

Calibration is the step during quantization where the float32 ranges are computed. For weights it is quite easy since the actual range is known at quantization-time. But it is less clear for activations, and different approaches exist:

1. **Post-training dynamic quantization**: The range for each activation is computed on the fly at runtime. While this gives great results without too much work, it can be slower than static quantization because of the overhead introduced by computing the range each time. It is also not an option on certain hardware.

2. **Post-training static quantization**: The range for each activation is computed in advance at quantization-time, typically by passing representative data through the model and recording the activation values. In practice, the steps are:
   - Observers are put on activations to record their values
   - A certain number of forward passes on a calibration dataset is done (around 200 examples is enough)
   - The ranges for each computation are computed according to some calibration technique

3. **Quantization-aware training**: The range for each activation is computed at training-time, following the same idea as post-training static quantization. But "fake quantize" operators are used instead of observers: they record values just as observers do, but they also simulate the error induced by quantization to let the model adapt to it.

### KV Cache Quantization

KV cache quantization reduces memory footprint during inference by storing key-value cache in lower precision, allowing more tokens to be cached and improving throughput. vLLM supports FP8 datatypes (E4M3 and E5M2) but not INT8 KV cache. While research has explored 4-bit or 2-bit KV cache quantization, these approaches typically cause noticeable accuracy degradation such as reduced MMLU scores.

### Quantization Kernels

vLLM offers multiple quantization kernel implementations for quantization methods like AWQ and GPTQ. For AWQ quantization, the official AWQ kernel serves as the default, while GPTQ models utilize the ExLlamaV2 kernel by default. Additional optimized kernels such as Marlin and Machete are also available, offering enhanced performance particularly for larger batch sizes.

### Inference Quality Degradation

Quantization can lead to degradation in inference quality. While this documentation primarily focuses on performance metrics, we provide the following references for assessing quality impact:

Qwen3-8B model benchmarks on various quantization methods:

| Quantization | CEVAL | MMLU | GSM8K | HUMANEVAL |
|-------------|-------|------|-------|-----------|
| BF16 | 79.27 | 74.78 | 87.79 | 63.41 |
| FP8-Static | 78.23 | 74.79 | 86.96 | 62.20 |
| FP8-Dynamic | 78.45 | 74.75 | 87.64 | 62.80 |
| INT8-Dynamic | 78.01 | 74.84 | 86.96 | 67.07 |
| INT4-GPTQ | 77.19 | 73.26 | 86.43 | 62.20 |
| INT4-AWQ | 76.15 | 73.59 | 86.96 | 63.41 |

For more details, please refer to the [AngelSlim benchmarks](https://github.com/Tencent/AngelSlim/tree/main?tab=readme-ov-file#-benchmark).

## Experimental Setup

- **Model**: Qwen3-8B
- **Hardware**: NVIDIA RTX 4090 24GB / A100 80GB / H100 80GB
- **vLLM Version**: v0.9.2
- **Dataset**: ShareGPT
- **Benchmark Script**:
```bash
# Prepare the ShareGPT dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Benchmark on ShareGPT dataset
vllm bench serve --model Qwen/Qwen3-8B --endpoint-type openai-chat --endpoint /v1/chat/completions --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000
```

## Experimental Results

### RTX 4090

| Quantization Method | Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |
|---------------------|-------------------|----------------|----------------|
| BF16                | 3869.3            | 15385.7        | 63.45          |
| AWQ                 | 5653.4 (+46.1%)   | 7913.4         | 87.7           |
| AWQ Marlin          | 5536.7 (+43.1%)   | 8133.8         | 90.42          |
| GPTQ (Int8)         | 4918.98 (+27.1%)  | 8415.70        | 96.07          |
| GPTQ Marlin         | 5025.82 (+29.9%)  | 8143.93        | 93.05          |

### A100 80GB


| Quantization Method | Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |
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
    GGUF model with architecture qwen3 is not supported.

### H100 80GB

| Quantization Method | Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |
|---------------------|-------------------|----------------|----------------|
| FP8 Static          | 16452.52 (+26.7%) | 4116.87 (-20.4%) | 85.57 (-32.4%) |
| FP8 Dynamic         | 15275.64 (+14.8%) | 4445.10 (-14.1%) | 98.94 (-21.8%) |
| Int4-W4A16          | 13605.46          | 5302.14         | 130.54         |
| BF16                | 13305.38          | 5172.78         | 126.55         |
| AWQ                 | 9756.06           | 8794.29         | 209.13         |
| GPTQ                | N/A               | N/A             | N/A            |

!!! note
    GPTQ implementation is broken, see [vLLM issue](https://github.com/vllm-project/vllm/issues/20986).

### KV Cache Quantization

| Configuration | Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |
|---------------|-------------------|----------------|----------------|
| BF16 (Baseline) | 13305.38 | 5172.78 | 126.55 |
| BF16 + KV Cache FP8 (E4M3) | 13513.19 (+1.6%) | 5688.39 | 103.21 |
