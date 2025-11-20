# Inference Performance Tuning Overview


## Overview

Open-source inference engines like vLLM and SGLang deliver excellent inference performance, but the performance gap between a tuned deployment and an untuned one might be larger than you think. The most effective validation method is to run benchmarks with your actual traffic on your target devices. Nonetheless, we have conducted numerous experiments across different inference engines, GPU devices, models, and configuration parameter combinations. Some general observations from these experiments can offer initial guidance before you dive into deep optimization.

## Observations

The following observations are based on the scope of our current experiments and may be updated or supplemented as more testing is done or as the community makes progress. For optimization methods and conclusions related to specific models on specific GPUs, please refer to the corresponding experimental documentation.

### Inference Engines

The choice of inference engine is crucial. Inference optimization often involves meticulous engine-specific tuning for particular scenarios, such as specific models, specific quantization schemes, specific GPUs, etc. Consequently, whether an engine is optimized for a given scenario makes a significant difference. For instance, vLLM runs gpt-oss-20b more than ten times faster than SGLang/TensorRT-LLM on an A100 GPU([see details](gpt-oss-20b/a100.md#1-choosing-the-inference-engine)). However, we cannot simply state that Engine A is universally better than Engine B. In our experimental results, vLLM, SGLang, and TensorRT-LLM each achieved the best performance in specific scenarios.

1. **vLLM** stands out for its excellent user experience, strong compatibility, and comprehensive, timely model support. It consistently delivers high-quality inference performance, making it a well-rounded and reliable choice across a wide range of scenarios.
2. **SGLang** also provides an excellent user experience and competitive inference performance. While its compatibility and model support breadth trail slightly behind vLLM, its model support timeliness remains among the best. It particularly excels in Speculative Decoding and in running large FP8 MoE models on Hopper GPUs.
3. **TensorRT-LLM** offers highly optimized inference performance and demonstrates exceptional results in certain experimental settings. However, its applicability is more specialized, with tighter vendor coupling and relatively slower support for newly released models.

### Quantization

1.  Quantization is a crucial technique for **maximizing throughput**.
2.  **Weight-Activation quantization** also helps reduce latency.

### Speculative Decoding

Speculative decoding is an effective method for optimizing latency. However, its effectiveness degrades significantly as the batch size increases. Therefore, **it is not suitable for improving throughput.**

### Parallelism Strategies

Parallelism strategies are essential for multi-GPU distributed inference.

### Kernels

vLLM/SGLang typically provide reasonable default selections based on the hardware environment. In most scenarios, the default attention backend is the most appropriate. Kernel optimizations like **DeepGEMM** are applicable to specific precisions/GPUs. While often enabled by default, there might be cases where disabling them is more suitable.

### Generalizing

A deployment configuration that performs well generally shows positive performance improvements across different Input Sequence Lengths (ISL) and Output Sequence Lengths (OSL), **though the ratios can differ significantly.**

### Deeper Tuning

Some parameters require tuning based on the actual inference request patterns, such as ISL/OSL, prefix repetition in the data, concurrency, etc. These include:

  1.  **Max batch size** (Related to concurrency)
  2.  **Scheduling config** (e.g., API server scale-out [vLLM], async scheduling [vLLM], schedule conservativeness [SGLang], schedule policy [SGLang]) (Related to various factors)
  3.  **Extended KV Cache** (Related to sequence length and prefix repetition)
  4.  **CUDA graph** (Related to concurrency)
  5.  **Torch compile** (Related to model architecture and concurrency)
