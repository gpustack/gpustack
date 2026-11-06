---
translation_status: translated
---

# Qwen3.5-9B 延迟优化 { #optimizing-qwen35-9b-latency }

## 结论 { #conclusion }

![延迟优化结果](../../assets/performance-lab/qwen3.5-9b-h100-latency.png)

Qwen/Qwen3.5-9B 在 NVIDIA H100 80GB HBM3 上的延迟优化推荐配置：

???+ tip "推理服务命令"
    ```bash
    vllm serve Qwen/Qwen3.5-9B \
        --reasoning-parser=qwen3 \
        --max-model-len=32768 \
        --speculative-config={"method":"mtp","num_speculative_tokens":1} \
        --language-model-only \
        --performance-mode=interactivity
    ```

优化前后的基准测试结果对比：

| 基准测试用例 | 基准配置（未做任何优化的 vLLM） | 优化后 |
|----------|-------------------------------------------|-----------|
| **请求速率 1** | Mean latency: 2.09s/req | Mean latency: 1.65s/req <span style="background-color:lightgreen;">(1.26x，更快)</span> |
| **请求速率 4** | Mean latency: 2.33s/req | Mean latency: 1.90s/req <span style="background-color:lightgreen;">(1.23x，更快)</span> |
| **请求速率 8** | Mean latency: 2.62s/req | Mean latency: 2.32s/req <span style="background-color:lightgreen;">(1.13x，更快)</span> |
| **请求速率 16** | Mean latency: 4.37s/req | Mean latency: 3.72s/req <span style="background-color:lightgreen;">(1.17x，更快)</span> |

!!! note
    1. 基准测试未覆盖所有可能的优化组合。例如，我们会选择默认配置下表现最好的推理引擎作为后续调优的起点。这种剪枝方式得到的是局部最优解，未必是全局最优解。
    2. 其他优化方法取决于具体使用场景，包括最大批大小、调度配置、扩展 KV cache、CUDA graph 等。本文结论可作为进一步针对性优化的起点。
    3. 测试基于特定的硬件和软件环境。推理引擎的改进可能带来新的结论。
    4. 量化可能影响准确率，但对于大多数模型，FP8 量化的准确率下降可控制在 1% 以内。详情参见[评估结果](https://github.com/Tencent/AngelSlim/blob/main/README_en.md#-benchmark)。因此，强烈建议在低延迟推理服务场景中使用 FP8 量化。
    5. 推测解码能够显著降低低并发请求的延迟，但加速效果取决于基准测试数据集的数据分布和草稿模型的选择。例如，本测试所选的草稿模型使用英文数据训练，在其他语言上的表现可能不够理想。

如有遗漏或需要根据最新变化更新的内容，请[告诉我们](https://github.com/gpustack/gpustack/issues/new/choose)。

## 实验环境 { #experimental-setup }

### 模型 { #model }

Qwen/Qwen3.5-9B

### 硬件 { #hardware }

NVIDIA H100 80GB HBM3

### 引擎版本 { #engine-version }

- vLLM v0.17.1
- SGLang v0.5.9

### 基准测试方法 { #benchmark-method }

本项目使用 GPUStack 的一键基准测试功能评估推理服务工作负载。本文中的基准测试均通过该流程执行。

GPUStack 的基准测试通过封装项目 [benchmark-runner](https://github.com/gpustack/benchmark-runner)，基于 [guidellm](https://github.com/vllm-project/guidellm) 实现。

对于以下基准测试配置，GPUStack 负责模型部署、基准测试任务提交和结果收集。

#### 基准测试配置 { #benchmark-profiles }

##### ShareGPT 8RPS { #sharegpt-8rps }

```yaml
dataset_name: ShareGPT
request_rate: 8
total_requests: 1000
```

##### ShareGPT 1RPS { #sharegpt-1rps }

```yaml
dataset_name: ShareGPT
request_rate: 1
total_requests: 1000
```

##### ShareGPT 4RPS { #sharegpt-4rps }

```yaml
dataset_name: ShareGPT
request_rate: 4
total_requests: 1000
```

##### ShareGPT 16RPS { #sharegpt-16rps }

```yaml
dataset_name: ShareGPT
request_rate: 16
total_requests: 1000
```

### 开源工具替代方案 { #open-source-replacement }

如果不使用 GPUStack，可以直接执行 `guidellm benchmark` 命令替代 GPUStack 基准测试流程。

对于包含 `dataset_name: ShareGPT` 的测试配置：

```bash
guidellm benchmark \
  --target ${target} \
  --profile constant \
  --rate ${request_rate} \
  --max-requests ${total_requests} \
  --processor ${model_path} \
  --data ./ShareGPT_V3_unfiltered_cleaned_split.json
```

## 实验结果 { #experiment-results }

### 选择推理引擎 { #choosing-the-inference-engine }

#### vLLM { #vllm }

- 测试配置：`ShareGPT 8RPS`
- 后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  ```

??? info "基准测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             31
    Benchmark duration (s):                  130.18
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.68
    Output token throughput (tok/s):         2197.17
    Peak output token throughput (tok/s):    2885041.51
    Peak concurrent requests:                31.00
    Total Token throughput (tok/s):          4867.84
    ----------------------Latency---------------------
    Mean Latency(s):                          2.62
    Median Latency(s):                        2.29
    P95 Latency(s):                           6.81
    P99 Latency(s):                           7.75
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          41.12
    Median TTFT (ms):                        33.92
    P95 TTFT (ms):                           72.24
    P99 TTFT (ms):                           141.44
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.34
    Median TPOT (ms):                        0.11
    P95 TPOT (ms):                           9.18
    P99 TPOT (ms):                           10.07
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.19
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            9.08
    P99 ITL (ms):                            10.04
    ==================================================
    ```

#### SGLang { #sglang }

- 测试配置：`ShareGPT 8RPS`
- 后端参数：
  ```bash
  --reasoning-parser=qwen3
  --context-length=32768
  ```

??? info "基准测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             96
    Benchmark duration (s):                  131.60
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.60
    Output token throughput (tok/s):         2145.85
    Peak output token throughput (tok/s):    33380.30
    Peak concurrent requests:                96.00
    Total Token throughput (tok/s):          4754.13
    ----------------------Latency---------------------
    Mean Latency(s):                          8.13
    Median Latency(s):                        6.94
    P95 Latency(s):                           20.20
    P99 Latency(s):                           26.21
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          98.72
    Median TTFT (ms):                        51.48
    P95 TTFT (ms):                           212.72
    P99 TTFT (ms):                           460.47
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          28.88
    Median TPOT (ms):                        28.26
    P95 TPOT (ms):                           43.34
    P99 TPOT (ms):                           51.52
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           28.63
    Median ITL (ms):                         27.88
    P95 ITL (ms):                            43.23
    P99 ITL (ms):                            50.55
    ==================================================
    ```

- 汇总：`vLLM` Mean Latency = 2.62s, `SGLang` Mean Latency = 8.13s. `vLLM` 性能更优，差值为 5.51s (速度提升至 3.10x). TTFT = 41.12 ms 与 98.72 ms, 减少 57.60 ms (速度提升至 2.40x); TPOT = 1.34 ms 与 28.88 ms, 减少 27.54 ms (速度提升至 21.62x).

### 前缀缓存 { #prefix-cache }

- 测试配置：`ShareGPT 8RPS`
- 后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  --enable-prefix-caching
  ```

??? info "基准测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             32
    Benchmark duration (s):                  130.23
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.68
    Output token throughput (tok/s):         2212.47
    Peak output token throughput (tok/s):    1443445.16
    Peak concurrent requests:                32.00
    Total Token throughput (tok/s):          4901.75
    ----------------------Latency---------------------
    Mean Latency(s):                          2.69
    Median Latency(s):                        2.36
    P95 Latency(s):                           6.95
    P99 Latency(s):                           7.94
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          46.19
    Median TTFT (ms):                        34.75
    P95 TTFT (ms):                           90.30
    P99 TTFT (ms):                           233.30
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.38
    Median TPOT (ms):                        0.12
    P95 TPOT (ms):                           9.49
    P99 TPOT (ms):                           10.50
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.22
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            9.35
    P99 ITL (ms):                            10.40
    ==================================================
    ```

### 最大批处理 Token 数 { #max-batch-token }

#### 16k { #16k }

- 测试配置：`ShareGPT 8RPS`
- 后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  --max-num-batched-tokens=16384
  ```

??? info "基准测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             31
    Benchmark duration (s):                  130.20
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.68
    Output token throughput (tok/s):         2220.17
    Peak output token throughput (tok/s):    2763536.75
    Peak concurrent requests:                31.00
    Total Token throughput (tok/s):          4918.81
    ----------------------Latency---------------------
    Mean Latency(s):                          2.64
    Median Latency(s):                        2.30
    P95 Latency(s):                           6.87
    P99 Latency(s):                           7.79
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          42.05
    Median TTFT (ms):                        34.13
    P95 TTFT (ms):                           76.02
    P99 TTFT (ms):                           190.03
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.26
    Median TPOT (ms):                        0.11
    P95 TPOT (ms):                           9.19
    P99 TPOT (ms):                           10.11
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.11
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            9.13
    P99 ITL (ms):                            10.04
    ==================================================
    ```

#### 32k { #32k }

- 测试配置：`ShareGPT 8RPS`
- 后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  --max-num-batched-tokens=32768
  ```

??? info "基准测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             31
    Benchmark duration (s):                  130.20
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.68
    Output token throughput (tok/s):         2196.67
    Peak output token throughput (tok/s):    1167858.46
    Peak concurrent requests:                31.00
    Total Token throughput (tok/s):          4866.73
    ----------------------Latency---------------------
    Mean Latency(s):                          2.64
    Median Latency(s):                        2.31
    P95 Latency(s):                           6.88
    P99 Latency(s):                           7.81
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          41.88
    Median TTFT (ms):                        34.03
    P95 TTFT (ms):                           76.91
    P99 TTFT (ms):                           164.85
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.38
    Median TPOT (ms):                        0.11
    P95 TPOT (ms):                           9.34
    P99 TPOT (ms):                           10.14
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.23
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            9.20
    P99 ITL (ms):                            10.09
    ==================================================
    ```

- 汇总：`16k` Mean Latency = 2.64s, `32k` Mean Latency = 2.64s. `16k` 性能更优，差值为 0.00s (速度提升至 1.00x). TTFT = 42.05 ms 与 41.88 ms, 增加 0.17 ms (耗时增加至 1.00x); TPOT = 1.26 ms 与 1.38 ms, 减少 0.12 ms (速度提升至 1.10x).

### 最大序列数 { #max-num-seqs }

#### 序列数 128 { #seqs-128 }

- 测试配置：`ShareGPT 8RPS`
- 后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  --max-num-seqs=128
  ```

??? info "基准测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             31
    Benchmark duration (s):                  130.18
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.68
    Output token throughput (tok/s):         2213.28
    Peak output token throughput (tok/s):    3033763.06
    Peak concurrent requests:                31.00
    Total Token throughput (tok/s):          4903.53
    ----------------------Latency---------------------
    Mean Latency(s):                          2.66
    Median Latency(s):                        2.32
    P95 Latency(s):                           6.91
    P99 Latency(s):                           7.82
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          44.11
    Median TTFT (ms):                        35.02
    P95 TTFT (ms):                           75.65
    P99 TTFT (ms):                           197.93
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.33
    Median TPOT (ms):                        0.11
    P95 TPOT (ms):                           9.53
    P99 TPOT (ms):                           10.28
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.17
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            9.48
    P99 ITL (ms):                            10.23
    ==================================================
    ```

#### 序列数 256 { #seqs-256 }

- 测试配置：`ShareGPT 8RPS`
- 后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  --max-num-seqs=256
  ```

??? info "基准测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             31
    Benchmark duration (s):                  130.18
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.68
    Output token throughput (tok/s):         2196.96
    Peak output token throughput (tok/s):    1994998.46
    Peak concurrent requests:                31.00
    Total Token throughput (tok/s):          4867.37
    ----------------------Latency---------------------
    Mean Latency(s):                          2.64
    Median Latency(s):                        2.30
    P95 Latency(s):                           6.82
    P99 Latency(s):                           7.79
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          42.97
    Median TTFT (ms):                        34.30
    P95 TTFT (ms):                           77.05
    P99 TTFT (ms):                           187.56
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.32
    Median TPOT (ms):                        0.11
    P95 TPOT (ms):                           9.23
    P99 TPOT (ms):                           10.28
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.17
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            9.16
    P99 ITL (ms):                            10.16
    ==================================================
    ```

- 汇总：`Seqs 256` Mean Latency = 2.64s, `Seqs 128` Mean Latency = 2.66s. `Seqs 256` 性能更优，差值为 0.02s (速度提升至 1.01x). TTFT = 42.97 ms 与 44.11 ms, 减少 1.13 ms (速度提升至 1.03x); TPOT = 1.32 ms 与 1.33 ms, 减少 0.01 ms (速度提升至 1.01x).

### 推测解码 { #speculative-decoding }

- 测试配置：`ShareGPT 8RPS`
- 后端参数：
  ```bash
  --reasoning-parser=qwen3
  --speculative-config={"method":"mtp","num_speculative_tokens":1}
  --max-model-len=32768
  ```

??? info "基准测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             28
    Benchmark duration (s):                  129.03
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.75
    Output token throughput (tok/s):         2224.24
    Peak output token throughput (tok/s):    9620127.05
    Peak concurrent requests:                28.00
    Total Token throughput (tok/s):          4927.83
    ----------------------Latency---------------------
    Mean Latency(s):                          2.39
    Median Latency(s):                        2.08
    P95 Latency(s):                           6.12
    P99 Latency(s):                           7.17
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          81.50
    Median TTFT (ms):                        55.93
    P95 TTFT (ms):                           96.41
    P99 TTFT (ms):                           788.19
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.29
    Median TPOT (ms):                        0.18
    P95 TPOT (ms):                           8.45
    P99 TPOT (ms):                           9.48
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.01
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            8.31
    P99 ITL (ms):                            9.42
    ==================================================
    ```

### 推测解码 && 性能模式 { #speculative-decoding-performance-mode }

- 测试配置：`ShareGPT 8RPS`
- 后端参数：
  ```bash
  --reasoning-parser=qwen3
  --performance-mode=interactivity
  --max-model-len=32768
  --speculative-config={"method":"mtp","num_speculative_tokens":1}
  ```

??? info "基准测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             28
    Benchmark duration (s):                  129.07
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.75
    Output token throughput (tok/s):         2214.80
    Peak output token throughput (tok/s):    1904162.55
    Peak concurrent requests:                28.00
    Total Token throughput (tok/s):          4906.90
    ----------------------Latency---------------------
    Mean Latency(s):                          2.38
    Median Latency(s):                        2.09
    P95 Latency(s):                           6.09
    P99 Latency(s):                           7.20
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          82.37
    Median TTFT (ms):                        55.71
    P95 TTFT (ms):                           96.16
    P99 TTFT (ms):                           960.28
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.42
    Median TPOT (ms):                        0.18
    P95 TPOT (ms):                           8.54
    P99 TPOT (ms):                           9.65
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.13
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            8.39
    P99 ITL (ms):                            9.08
    ==================================================
    ```

### 推测解码 && 性能模式 && 语言模型 { #speculative-decoding-performance-mode-language-model }

- 测试配置：`ShareGPT 8RPS`
- 后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  --speculative-config={"method":"mtp","num_speculative_tokens":1}
  --language-model-only
  --performance-mode=interactivity
  ```

??? info "基准测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             27
    Benchmark duration (s):                  128.98
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.75
    Output token throughput (tok/s):         2214.46
    Peak output token throughput (tok/s):    4861500.66
    Peak concurrent requests:                27.00
    Total Token throughput (tok/s):          4906.15
    ----------------------Latency---------------------
    Mean Latency(s):                          2.32
    Median Latency(s):                        2.02
    P95 Latency(s):                           5.99
    P99 Latency(s):                           6.89
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          80.63
    Median TTFT (ms):                        54.04
    P95 TTFT (ms):                           96.82
    P99 TTFT (ms):                           933.41
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.31
    Median TPOT (ms):                        0.17
    P95 TPOT (ms):                           8.19
    P99 TPOT (ms):                           9.47
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.02
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            8.03
    P99 ITL (ms):                            8.91
    ==================================================
    ```

### 优化选项汇总 { #summary-of-optimization-options }

| 基准测试用例 | 组别 | 优化后 | 基准配置 | 对比 |
|---|---|---:|---:|---|
| ShareGPT 8RPS (r=8) | 选择推理引擎 | 2.62s | 2.62s | <span style="background-color:lightgreen;">(1.00x，更快)</span> |
| ShareGPT 8RPS (r=8) | 前缀缓存 | 2.69s | 2.62s | <span style="background-color:#ffd6d6;">(1.03x，更慢)</span> |
| ShareGPT 8RPS (r=8) | 最大批处理 Token 数 | 2.64s | 2.62s | <span style="background-color:#ffd6d6;">(1.01x，更慢)</span> |
| ShareGPT 8RPS (r=8) | 最大序列数 | 2.64s | 2.62s | <span style="background-color:#ffd6d6;">(1.01x，更慢)</span> |
| ShareGPT 8RPS (r=8) | 推测解码 | 2.39s | 2.62s | <span style="background-color:lightgreen;">(1.09x，更快)</span> |
| ShareGPT 8RPS (r=8) | 推测解码 && 性能模式 | 2.38s | 2.62s | <span style="background-color:lightgreen;">(1.10x，更快)</span> |
| ShareGPT 8RPS (r=8) | 推测解码 && 性能模式 && 语言模型 | 2.32s | 2.62s | <span style="background-color:lightgreen;">(1.13x，更快)</span> |

### 其他基准测试用例 { #other-benchmark-cases }

#### 请求速率 1 { #rate-1 }

- 基准配置的后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  ```

??? info "基准配置的测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             6
    Benchmark duration (s):                  1001.23
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              1.00
    Output token throughput (tok/s):         281.69
    Peak output token throughput (tok/s):    211793.17
    Peak concurrent requests:                6.00
    Total Token throughput (tok/s):          624.09
    ----------------------Latency---------------------
    Mean Latency(s):                          2.09
    Median Latency(s):                        1.85
    P95 Latency(s):                           5.42
    P99 Latency(s):                           6.27
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          37.03
    Median TTFT (ms):                        31.51
    P95 TTFT (ms):                           63.17
    P99 TTFT (ms):                           95.35
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          0.94
    Median TPOT (ms):                        0.10
    P95 TPOT (ms):                           7.38
    P99 TPOT (ms):                           7.51
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           0.81
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            7.34
    P99 ITL (ms):                            7.47
    ==================================================
    ```

- 优化后的后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  --speculative-config={"method":"mtp","num_speculative_tokens":1}
  --language-model-only
  --performance-mode=interactivity
  ```

??? info "优化后的测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             5
    Benchmark duration (s):                  1000.35
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              1.00
    Output token throughput (tok/s):         281.69
    Peak output token throughput (tok/s):    271386.26
    Peak concurrent requests:                5.00
    Total Token throughput (tok/s):          624.08
    ----------------------Latency---------------------
    Mean Latency(s):                          1.65
    Median Latency(s):                        1.45
    P95 Latency(s):                           4.28
    P99 Latency(s):                           4.98
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          54.58
    Median TTFT (ms):                        50.45
    P95 TTFT (ms):                           75.37
    P99 TTFT (ms):                           123.14
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          0.91
    Median TPOT (ms):                        0.15
    P95 TPOT (ms):                           5.90
    P99 TPOT (ms):                           6.22
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           0.72
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            5.75
    P99 ITL (ms):                            6.14
    ==================================================
    ```

#### 请求速率 4 { #rate-4 }

- 基准配置的后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  ```

??? info "基准配置的测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             15
    Benchmark duration (s):                  254.36
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              3.93
    Output token throughput (tok/s):         1126.78
    Peak output token throughput (tok/s):    800262.91
    Peak concurrent requests:                15.00
    Total Token throughput (tok/s):          2496.39
    ----------------------Latency---------------------
    Mean Latency(s):                          2.33
    Median Latency(s):                        2.06
    P95 Latency(s):                           5.96
    P99 Latency(s):                           6.94
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          38.09
    Median TTFT (ms):                        31.67
    P95 TTFT (ms):                           67.05
    P99 TTFT (ms):                           109.44
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.15
    Median TPOT (ms):                        0.10
    P95 TPOT (ms):                           8.25
    P99 TPOT (ms):                           8.73
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.02
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            8.19
    P99 ITL (ms):                            8.67
    ==================================================
    ```

- 优化后的后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  --speculative-config={"method":"mtp","num_speculative_tokens":1}
  --language-model-only
  --performance-mode=interactivity
  ```

??? info "优化后的测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             13
    Benchmark duration (s):                  253.27
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              3.95
    Output token throughput (tok/s):         1126.67
    Peak output token throughput (tok/s):    996666.30
    Peak concurrent requests:                13.00
    Total Token throughput (tok/s):          2496.15
    ----------------------Latency---------------------
    Mean Latency(s):                          1.90
    Median Latency(s):                        1.66
    P95 Latency(s):                           4.91
    P99 Latency(s):                           5.82
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          57.94
    Median TTFT (ms):                        52.57
    P95 TTFT (ms):                           77.15
    P99 TTFT (ms):                           112.48
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          0.97
    Median TPOT (ms):                        0.16
    P95 TPOT (ms):                           6.79
    P99 TPOT (ms):                           7.25
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           0.76
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            6.64
    P99 ITL (ms):                            7.11
    ==================================================
    ```

#### 请求速率 8 { #rate-8 }

- 基准配置的后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  ```

??? info "基准配置的测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             31
    Benchmark duration (s):                  130.18
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.68
    Output token throughput (tok/s):         2197.17
    Peak output token throughput (tok/s):    2885041.51
    Peak concurrent requests:                31.00
    Total Token throughput (tok/s):          4867.84
    ----------------------Latency---------------------
    Mean Latency(s):                          2.62
    Median Latency(s):                        2.29
    P95 Latency(s):                           6.81
    P99 Latency(s):                           7.75
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          41.12
    Median TTFT (ms):                        33.92
    P95 TTFT (ms):                           72.24
    P99 TTFT (ms):                           141.44
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.34
    Median TPOT (ms):                        0.11
    P95 TPOT (ms):                           9.18
    P99 TPOT (ms):                           10.07
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.19
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            9.08
    P99 ITL (ms):                            10.04
    ==================================================
    ```

- 优化后的后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  --speculative-config={"method":"mtp","num_speculative_tokens":1}
  --language-model-only
  --performance-mode=interactivity
  ```

??? info "优化后的测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             27
    Benchmark duration (s):                  128.98
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.75
    Output token throughput (tok/s):         2214.46
    Peak output token throughput (tok/s):    4861500.66
    Peak concurrent requests:                27.00
    Total Token throughput (tok/s):          4906.15
    ----------------------Latency---------------------
    Mean Latency(s):                          2.32
    Median Latency(s):                        2.02
    P95 Latency(s):                           5.99
    P99 Latency(s):                           6.89
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          80.63
    Median TTFT (ms):                        54.04
    P95 TTFT (ms):                           96.82
    P99 TTFT (ms):                           933.41
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.31
    Median TPOT (ms):                        0.17
    P95 TPOT (ms):                           8.19
    P99 TPOT (ms):                           9.47
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.02
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            8.03
    P99 ITL (ms):                            8.91
    ==================================================
    ```

#### 请求速率 16 { #rate-16 }

- 基准配置的后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  ```

??? info "基准配置的测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             97
    Benchmark duration (s):                  68.59
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              14.58
    Output token throughput (tok/s):         4119.85
    Peak output token throughput (tok/s):    4297231.71
    Peak concurrent requests:                97.00
    Total Token throughput (tok/s):          9127.56
    ----------------------Latency---------------------
    Mean Latency(s):                          4.37
    Median Latency(s):                        3.74
    P95 Latency(s):                           10.93
    P99 Latency(s):                           14.05
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          63.36
    Median TTFT (ms):                        52.12
    P95 TTFT (ms):                           111.22
    P99 TTFT (ms):                           283.00
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          2.07
    Median TPOT (ms):                        0.17
    P95 TPOT (ms):                           16.46
    P99 TPOT (ms):                           18.09
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.85
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            16.34
    P99 ITL (ms):                            17.94
    ==================================================
    ```

- 优化后的后端参数：
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  --speculative-config={"method":"mtp","num_speculative_tokens":1}
  --language-model-only
  --performance-mode=interactivity
  ```

??? info "优化后的测试结果"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             77
    Benchmark duration (s):                  67.27
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              14.86
    Output token throughput (tok/s):         4191.07
    Peak output token throughput (tok/s):    3207595.57
    Peak concurrent requests:                77.00
    Total Token throughput (tok/s):          9285.35
    ----------------------Latency---------------------
    Mean Latency(s):                          3.72
    Median Latency(s):                        3.31
    P95 Latency(s):                           9.44
    P99 Latency(s):                           11.45
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          81.88
    Median TTFT (ms):                        66.01
    P95 TTFT (ms):                           149.68
    P99 TTFT (ms):                           375.11
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.92
    Median TPOT (ms):                        0.22
    P95 TPOT (ms):                           13.69
    P99 TPOT (ms):                           15.01
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.63
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            13.50
    P99 ITL (ms):                            14.78
    ==================================================
    ```
