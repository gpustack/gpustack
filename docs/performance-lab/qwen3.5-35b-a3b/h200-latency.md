# Optimizing Qwen3.5-35B-A3B Latency

## Conclusion

![Latency Optimization Result](../../assets/performance-lab/qwen3.5-35b-a3b-h200-latency.png)

Recommended configuration for optimizing latency of Qwen/Qwen3.5-35B-A3B-FP8 on NVIDIA H200:

???+ tip "Serving Command"
    ```bash
    vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
        --reasoning-parser=qwen3 \
        --speculative-config={"method":"mtp","num_speculative_tokens":1} \
        --max-model-len=32768
    ```

Comparison of benchmark results before and after optimization:

| Benchmark Case | Baseline (vLLM without any optimizations) | Optimized |
|----------|-------------------------------------------|-----------|
| **Rate 1** | Mean latency: 1.85s/req | Mean latency: 1.50s/req <span style="background-color:lightgreen;">(1.23x faster)</span> |
| **Rate 4** | Mean latency: 3.30s/req | Mean latency: 2.52s/req <span style="background-color:lightgreen;">(1.31x faster)</span> |
| **Rate 8** | Mean latency: 5.32s/req | Mean latency: 3.28s/req <span style="background-color:lightgreen;">(1.63x faster)</span> |
| **Rate 16** | Mean latency: 18.59s/req | Mean latency: 6.79s/req <span style="background-color:lightgreen;">(2.74x faster)</span> |

!!! note
    1. Our benchmark tests do not cover all possible optimization combinations. For example, we select the inference engine that performs best under its default configuration as the starting point for further tuning. This pruning approach yields a local optimum, which may not be the global optimum.
    2. There are other optimization methods that depend on specific user scenarios, including max batch size, schedule configuration, extended KV cache, CUDA graph, etc. The conclusions in this document can serve as a starting point for more targeted optimizations.
    3. The tests are conducted on specific hardware and software setups. Advances in the inference engine may lead to new conclusions.
    4. Although using quantization may impact accuracy. FP8 quantization can achieve less than 1% accuracy drop for most models. See the [evaluation results](https://github.com/Tencent/AngelSlim/blob/main/README_en.md#-benchmark) for more details. Therefore, it is highly recommended to use FP8 quantization for low-latency serving scenarios.
    5. Speculative decoding can significantly reduce latency for low-concurrency requests. However, the acceleration effect may vary depending on the data distribution of different benchmark datasets and the choice of draft models. For example, the chosen draft model here is trained on English data, which may lead to suboptimal performance on other languages.

If there are any missing points or updates reflecting new changes, please [let us know](https://github.com/gpustack/gpustack/issues/new/choose).

## Experimental Setup

### Model

Qwen/Qwen3.5-35B-A3B

### Hardware

NVIDIA H200

### Engine Version

- vLLM v0.17.1
- SGLang v0.5.9

### Benchmark Method

This project uses GPUStack's one-click benchmark capability for serving workloads. The benchmark tests in this document were executed with that workflow.

GPUStack's benchmark implementation is built on top of [guidellm](https://github.com/vllm-project/guidellm) via the wrapper project [benchmark-runner](https://github.com/gpustack/benchmark-runner).

GPUStack handles model deployment, benchmark job submission, and result collection for the benchmark configurations listed below.

#### Benchmark Profiles

##### ShareGPT 8RPS

```yaml
dataset_name: ShareGPT
request_rate: 8
total_requests: 1000
```

##### ShareGPT 1RPS

```yaml
dataset_name: ShareGPT
request_rate: 1
total_requests: 1000
```

##### ShareGPT 4RPS

```yaml
dataset_name: ShareGPT
request_rate: 4
total_requests: 1000
```

##### ShareGPT 16RPS

```yaml
dataset_name: ShareGPT
request_rate: 16
total_requests: 1000
```

### Open-Source Replacement

If you do not use GPUStack, you can replace the GPUStack benchmark workflow with direct `guidellm benchmark` commands.

For profiles with `dataset_name: ShareGPT`:

```bash
guidellm benchmark \
  --target ${target} \
  --profile constant \
  --rate ${request_rate} \
  --max-requests ${total_requests} \
  --processor ${model_path} \
  --data ./ShareGPT_V3_unfiltered_cleaned_split.json
```

## Experiment Results

### Choosing the Inference Engine

#### vLLM

- Profile: `ShareGPT 8RPS`
- Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  ```

??? info "Benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             59
    Benchmark duration (s):                  132.25
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.56
    Output token throughput (tok/s):         2137.15
    Peak output token throughput (tok/s):    1410355.61
    Peak concurrent requests:                59.00
    Total Token throughput (tok/s):          4734.87
    ----------------------Latency---------------------
    Mean Latency(s):                          5.32
    Median Latency(s):                        4.72
    P95 Latency(s):                           13.39
    P99 Latency(s):                           16.27
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          68.48
    Median TTFT (ms):                        62.59
    P95 TTFT (ms):                           102.52
    P99 TTFT (ms):                           131.38
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          2.35
    Median TPOT (ms):                        0.20
    P95 TPOT (ms):                           18.88
    P99 TPOT (ms):                           20.04
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           2.11
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            18.78
    P99 ITL (ms):                            19.94
    ==================================================
    ```

#### SGLang

- Profile: `ShareGPT 8RPS`
- Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --context-length=32768
  ```

??? info "Benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             246
    Benchmark duration (s):                  148.88
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              6.72
    Output token throughput (tok/s):         1896.19
    Peak output token throughput (tok/s):    36196.65
    Peak concurrent requests:                246.00
    Total Token throughput (tok/s):          4201.01
    ----------------------Latency---------------------
    Mean Latency(s):                          25.15
    Median Latency(s):                        23.44
    P95 Latency(s):                           50.27
    P99 Latency(s):                           61.67
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          6456.42
    Median TTFT (ms):                        6537.59
    P95 TTFT (ms):                           12387.08
    P99 TTFT (ms):                           12670.79
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          89.37
    Median TPOT (ms):                        82.58
    P95 TPOT (ms):                           134.49
    P99 TPOT (ms):                           222.06
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           66.66
    Median ITL (ms):                         68.70
    P95 ITL (ms):                            93.84
    P99 ITL (ms):                            125.45
    ==================================================
    ```

- Summary: `vLLM` Mean Latency = 5.32s, `SGLang` Mean Latency = 25.15s. `vLLM` is faster by 19.83s (4.73x faster). TTFT = 68.48 ms vs 6456.42 ms, reduced by 6387.94 ms (94.29x faster); TPOT = 2.35 ms vs 89.37 ms, reduced by 87.02 ms (38.10x faster).

### Quantization

- Profile: `ShareGPT 8RPS`
- Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  ```

??? info "Benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             42
    Benchmark duration (s):                  130.13
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.68
    Output token throughput (tok/s):         2194.80
    Peak output token throughput (tok/s):    716367.44
    Peak concurrent requests:                42.00
    Total Token throughput (tok/s):          4862.60
    ----------------------Latency---------------------
    Mean Latency(s):                          3.53
    Median Latency(s):                        3.09
    P95 Latency(s):                           8.98
    P99 Latency(s):                           10.58
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          51.56
    Median TTFT (ms):                        43.82
    P95 TTFT (ms):                           91.52
    P99 TTFT (ms):                           107.55
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.61
    Median TPOT (ms):                        0.14
    P95 TPOT (ms):                           12.48
    P99 TPOT (ms):                           13.57
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.43
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            12.41
    P99 ITL (ms):                            13.52
    ==================================================
    ```

### Cuda Graph && Quantization

#### Size 256

- Profile: `ShareGPT 8RPS`
- Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --max-cudagraph-capture-size=256
  --max-model-len=32768
  ```

??? info "Benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             43
    Benchmark duration (s):                  130.15
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.68
    Output token throughput (tok/s):         2179.71
    Peak output token throughput (tok/s):    634473.24
    Peak concurrent requests:                43.00
    Total Token throughput (tok/s):          4829.15
    ----------------------Latency---------------------
    Mean Latency(s):                          3.66
    Median Latency(s):                        3.21
    P95 Latency(s):                           9.22
    P99 Latency(s):                           11.00
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          55.07
    Median TTFT (ms):                        44.14
    P95 TTFT (ms):                           94.53
    P99 TTFT (ms):                           109.17
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.54
    Median TPOT (ms):                        0.15
    P95 TPOT (ms):                           12.82
    P99 TPOT (ms):                           13.89
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.35
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            12.71
    P99 ITL (ms):                            13.81
    ==================================================
    ```

#### Size 512

- Profile: `ShareGPT 8RPS`
- Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --max-cudagraph-capture-size=512
  --max-model-len=32768
  ```

??? info "Benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             42
    Benchmark duration (s):                  130.15
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.68
    Output token throughput (tok/s):         2180.27
    Peak output token throughput (tok/s):    2532773.65
    Peak concurrent requests:                42.00
    Total Token throughput (tok/s):          4830.39
    ----------------------Latency---------------------
    Mean Latency(s):                          3.54
    Median Latency(s):                        3.11
    P95 Latency(s):                           8.97
    P99 Latency(s):                           10.69
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          51.99
    Median TTFT (ms):                        44.02
    P95 TTFT (ms):                           92.66
    P99 TTFT (ms):                           110.20
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.77
    Median TPOT (ms):                        0.15
    P95 TPOT (ms):                           12.69
    P99 TPOT (ms):                           13.91
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.59
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            12.59
    P99 ITL (ms):                            13.85
    ==================================================
    ```

- Summary: `Size 512` Mean Latency = 3.54s, `Size 256` Mean Latency = 3.66s. `Size 512` is faster by 0.12s (1.03x faster). TTFT = 51.99 ms vs 55.07 ms, reduced by 3.07 ms (1.06x faster); TPOT = 1.77 ms vs 1.54 ms, increased by 0.23 ms (1.15x slower).

### Max Batch Token && Quantization

#### 16k

- Profile: `ShareGPT 8RPS`
- Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  --max-num-batched-tokens=16384
  ```

??? info "Benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             59
    Benchmark duration (s):                  130.18
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.68
    Output token throughput (tok/s):         2179.90
    Peak output token throughput (tok/s):    13119091.07
    Peak concurrent requests:                59.00
    Total Token throughput (tok/s):          4829.57
    ----------------------Latency---------------------
    Mean Latency(s):                          3.82
    Median Latency(s):                        3.34
    P95 Latency(s):                           9.63
    P99 Latency(s):                           11.31
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          179.88
    Median TTFT (ms):                        44.93
    P95 TTFT (ms):                           262.90
    P99 TTFT (ms):                           4297.33
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          2.28
    Median TPOT (ms):                        0.15
    P95 TPOT (ms):                           13.17
    P99 TPOT (ms):                           19.61
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.65
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            12.74
    P99 ITL (ms):                            14.42
    ==================================================
    ```

#### 32k

- Profile: `ShareGPT 8RPS`
- Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  --max-num-batched-tokens=32768
  ```

??? info "Benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             57
    Benchmark duration (s):                  130.14
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.68
    Output token throughput (tok/s):         2216.28
    Peak output token throughput (tok/s):    17112760.32
    Peak concurrent requests:                57.00
    Total Token throughput (tok/s):          4910.18
    ----------------------Latency---------------------
    Mean Latency(s):                          3.82
    Median Latency(s):                        3.36
    P95 Latency(s):                           9.55
    P99 Latency(s):                           11.33
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          198.01
    Median TTFT (ms):                        44.82
    P95 TTFT (ms):                           239.63
    P99 TTFT (ms):                           4772.71
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          2.26
    Median TPOT (ms):                        0.15
    P95 TPOT (ms):                           13.35
    P99 TPOT (ms):                           21.22
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.56
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            13.02
    P99 ITL (ms):                            14.55
    ==================================================
    ```

- Summary: `32k` Mean Latency = 3.82s, `16k` Mean Latency = 3.82s. `32k` is faster by 0.00s (1.00x faster). TTFT = 198.01 ms vs 179.88 ms, increased by 18.12 ms (1.10x slower); TPOT = 2.26 ms vs 2.28 ms, reduced by 0.02 ms (1.01x faster).

### Performance Mode && Quantization

- Profile: `ShareGPT 8RPS`
- Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --performance-mode=interactivity
  --max-model-len=32768
  ```

??? info "Benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             55
    Benchmark duration (s):                  129.88
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.70
    Output token throughput (tok/s):         2187.29
    Peak output token throughput (tok/s):    10792660.29
    Peak concurrent requests:                55.00
    Total Token throughput (tok/s):          4845.96
    ----------------------Latency---------------------
    Mean Latency(s):                          3.67
    Median Latency(s):                        3.25
    P95 Latency(s):                           9.10
    P99 Latency(s):                           11.00
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          184.63
    Median TTFT (ms):                        43.80
    P95 TTFT (ms):                           311.58
    P99 TTFT (ms):                           4340.84
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          2.16
    Median TPOT (ms):                        0.15
    P95 TPOT (ms):                           13.05
    P99 TPOT (ms):                           20.02
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.51
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            12.37
    P99 ITL (ms):                            15.66
    ==================================================
    ```

### Prefix Cache && Quantization

- Profile: `ShareGPT 8RPS`
- Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  --enable-prefix-caching
  ```

??? info "Benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             44
    Benchmark duration (s):                  130.20
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.68
    Output token throughput (tok/s):         2179.34
    Peak output token throughput (tok/s):    1595392.00
    Peak concurrent requests:                44.00
    Total Token throughput (tok/s):          4828.33
    ----------------------Latency---------------------
    Mean Latency(s):                          3.70
    Median Latency(s):                        3.24
    P95 Latency(s):                           9.43
    P99 Latency(s):                           11.34
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          58.68
    Median TTFT (ms):                        44.61
    P95 TTFT (ms):                           125.39
    P99 TTFT (ms):                           249.12
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.81
    Median TPOT (ms):                        0.15
    P95 TPOT (ms):                           12.83
    P99 TPOT (ms):                           14.84
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.60
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            12.75
    P99 ITL (ms):                            14.80
    ==================================================
    ```

### Speculative Decoding && Quantization

- Profile: `ShareGPT 8RPS`
- Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --speculative-config={"method":"mtp","num_speculative_tokens":1}
  --max-model-len=32768
  ```

??? info "Benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             57
    Benchmark duration (s):                  129.19
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.74
    Output token throughput (tok/s):         2203.31
    Peak output token throughput (tok/s):    8794775.78
    Peak concurrent requests:                57.00
    Total Token throughput (tok/s):          4881.45
    ----------------------Latency---------------------
    Mean Latency(s):                          3.28
    Median Latency(s):                        2.85
    P95 Latency(s):                           8.22
    P99 Latency(s):                           10.49
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          263.11
    Median TTFT (ms):                        60.45
    P95 TTFT (ms):                           891.65
    P99 TTFT (ms):                           5402.99
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          2.34
    Median TPOT (ms):                        0.21
    P95 TPOT (ms):                           11.39
    P99 TPOT (ms):                           22.27
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.41
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            10.58
    P99 ITL (ms):                            14.34
    ==================================================
    ```

### Summary of Optimization Options

| Benchmark Case | Group | Optimized | Baseline | Comparison |
|---|---|---:|---:|---|
| ShareGPT 8RPS (r=8) | Choosing the Inference Engine | 5.32s | 5.32s | <span style="background-color:lightgreen;">(1.00x faster)</span> |
| ShareGPT 8RPS (r=8) | Quantization | 3.53s | 5.32s | <span style="background-color:lightgreen;">(1.51x faster)</span> |
| ShareGPT 8RPS (r=8) | Cuda Graph && Quantization | 3.54s | 5.32s | <span style="background-color:lightgreen;">(1.50x faster)</span> |
| ShareGPT 8RPS (r=8) | Max Batch Token && Quantization | 3.82s | 5.32s | <span style="background-color:lightgreen;">(1.39x faster)</span> |
| ShareGPT 8RPS (r=8) | Performance Mode && Quantization | 3.67s | 5.32s | <span style="background-color:lightgreen;">(1.45x faster)</span> |
| ShareGPT 8RPS (r=8) | Prefix Cache && Quantization | 3.70s | 5.32s | <span style="background-color:lightgreen;">(1.44x faster)</span> |
| ShareGPT 8RPS (r=8) | Speculative Decoding && Quantization | 3.28s | 5.32s | <span style="background-color:lightgreen;">(1.63x faster)</span> |

### Other Benchmark Cases

#### Rate 1

- Baseline Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  ```

??? info "Baseline benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             6
    Benchmark duration (s):                  1000.76
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              1.00
    Output token throughput (tok/s):         281.70
    Peak output token throughput (tok/s):    205535.27
    Peak concurrent requests:                6.00
    Total Token throughput (tok/s):          624.11
    ----------------------Latency---------------------
    Mean Latency(s):                          1.85
    Median Latency(s):                        1.57
    P95 Latency(s):                           4.80
    P99 Latency(s):                           5.90
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          46.01
    Median TTFT (ms):                        35.63
    P95 TTFT (ms):                           90.77
    P99 TTFT (ms):                           98.52
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.06
    Median TPOT (ms):                        0.12
    P95 TPOT (ms):                           6.90
    P99 TPOT (ms):                           7.56
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           0.90
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            6.81
    P99 ITL (ms):                            7.50
    ==================================================
    ```

- Optimized Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --speculative-config={"method":"mtp","num_speculative_tokens":1}
  --max-model-len=32768
  ```

??? info "Optimized benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             5
    Benchmark duration (s):                  1000.01
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              1.00
    Output token throughput (tok/s):         281.71
    Peak output token throughput (tok/s):    160685.52
    Peak concurrent requests:                5.00
    Total Token throughput (tok/s):          624.13
    ----------------------Latency---------------------
    Mean Latency(s):                          1.50
    Median Latency(s):                        1.31
    P95 Latency(s):                           3.88
    P99 Latency(s):                           4.65
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          66.56
    Median TTFT (ms):                        55.47
    P95 TTFT (ms):                           128.15
    P99 TTFT (ms):                           145.32
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          0.89
    Median TPOT (ms):                        0.17
    P95 TPOT (ms):                           5.38
    P99 TPOT (ms):                           5.89
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           0.66
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            5.25
    P99 ITL (ms):                            5.75
    ==================================================
    ```

#### Rate 4

- Baseline Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  ```

??? info "Baseline benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             34
    Benchmark duration (s):                  254.48
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              3.93
    Output token throughput (tok/s):         1119.46
    Peak output token throughput (tok/s):    12576036.09
    Peak concurrent requests:                34.00
    Total Token throughput (tok/s):          2480.18
    ----------------------Latency---------------------
    Mean Latency(s):                          3.30
    Median Latency(s):                        2.84
    P95 Latency(s):                           8.64
    P99 Latency(s):                           10.14
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          126.16
    Median TTFT (ms):                        42.08
    P95 TTFT (ms):                           96.81
    P99 TTFT (ms):                           3590.00
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          2.02
    Median TPOT (ms):                        0.14
    P95 TPOT (ms):                           11.71
    P99 TPOT (ms):                           15.87
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.57
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            11.49
    P99 ITL (ms):                            13.24
    ==================================================
    ```

- Optimized Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --speculative-config={"method":"mtp","num_speculative_tokens":1}
  --max-model-len=32768
  ```

??? info "Optimized benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             35
    Benchmark duration (s):                  253.07
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              3.95
    Output token throughput (tok/s):         1126.78
    Peak output token throughput (tok/s):    14184016.34
    Peak concurrent requests:                35.00
    Total Token throughput (tok/s):          2496.38
    ----------------------Latency---------------------
    Mean Latency(s):                          2.52
    Median Latency(s):                        2.13
    P95 Latency(s):                           6.48
    P99 Latency(s):                           10.07
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          221.71
    Median TTFT (ms):                        62.58
    P95 TTFT (ms):                           143.76
    P99 TTFT (ms):                           5977.29
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          1.78
    Median TPOT (ms):                        0.21
    P95 TPOT (ms):                           8.39
    P99 TPOT (ms):                           20.07
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           0.99
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            7.98
    P99 ITL (ms):                            9.26
    ==================================================
    ```

#### Rate 8

- Baseline Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  ```

??? info "Baseline benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             59
    Benchmark duration (s):                  132.25
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.56
    Output token throughput (tok/s):         2137.15
    Peak output token throughput (tok/s):    1410355.61
    Peak concurrent requests:                59.00
    Total Token throughput (tok/s):          4734.87
    ----------------------Latency---------------------
    Mean Latency(s):                          5.32
    Median Latency(s):                        4.72
    P95 Latency(s):                           13.39
    P99 Latency(s):                           16.27
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          68.48
    Median TTFT (ms):                        62.59
    P95 TTFT (ms):                           102.52
    P99 TTFT (ms):                           131.38
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          2.35
    Median TPOT (ms):                        0.20
    P95 TPOT (ms):                           18.88
    P99 TPOT (ms):                           20.04
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           2.11
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            18.78
    P99 ITL (ms):                            19.94
    ==================================================
    ```

- Optimized Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --speculative-config={"method":"mtp","num_speculative_tokens":1}
  --max-model-len=32768
  ```

??? info "Optimized benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             57
    Benchmark duration (s):                  129.19
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              7.74
    Output token throughput (tok/s):         2203.31
    Peak output token throughput (tok/s):    8794775.78
    Peak concurrent requests:                57.00
    Total Token throughput (tok/s):          4881.45
    ----------------------Latency---------------------
    Mean Latency(s):                          3.28
    Median Latency(s):                        2.85
    P95 Latency(s):                           8.22
    P99 Latency(s):                           10.49
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          263.11
    Median TTFT (ms):                        60.45
    P95 TTFT (ms):                           891.65
    P99 TTFT (ms):                           5402.99
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          2.34
    Median TPOT (ms):                        0.21
    P95 TPOT (ms):                           11.39
    P99 TPOT (ms):                           22.27
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           1.41
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            10.58
    P99 ITL (ms):                            14.34
    ==================================================
    ```

#### Rate 16

- Baseline Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --max-model-len=32768
  ```

??? info "Baseline benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             447
    Benchmark duration (s):                  80.77
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              12.38
    Output token throughput (tok/s):         3499.91
    Peak output token throughput (tok/s):    4256990.61
    Peak concurrent requests:                447.00
    Total Token throughput (tok/s):          7754.06
    ----------------------Latency---------------------
    Mean Latency(s):                          18.59
    Median Latency(s):                        16.18
    P95 Latency(s):                           43.61
    P99 Latency(s):                           53.53
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          232.24
    Median TTFT (ms):                        229.86
    P95 TTFT (ms):                           402.69
    P99 TTFT (ms):                           465.50
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          8.90
    Median TPOT (ms):                        0.67
    P95 TPOT (ms):                           68.77
    P99 TPOT (ms):                           94.53
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           8.11
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            68.39
    P99 ITL (ms):                            93.98
    ==================================================
    ```

- Optimized Backend Parameters:
  ```bash
  --reasoning-parser=qwen3
  --speculative-config={"method":"mtp","num_speculative_tokens":1}
  --max-model-len=32768
  ```

??? info "Optimized benchmark result"
    ```
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Maximum request concurrency:             145
    Benchmark duration (s):                  69.63
    Total input tokens:                      342058
    Total generated tokens:                  281412
    Request throughput (req/s):              14.36
    Output token throughput (tok/s):         4070.93
    Peak output token throughput (tok/s):    3898930.48
    Peak concurrent requests:                145.00
    Total Token throughput (tok/s):          9019.16
    ----------------------Latency---------------------
    Mean Latency(s):                          6.79
    Median Latency(s):                        5.91
    P95 Latency(s):                           16.32
    P99 Latency(s):                           21.54
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          150.30
    Median TTFT (ms):                        133.22
    P95 TTFT (ms):                           293.44
    P99 TTFT (ms):                           345.17
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          3.23
    Median TPOT (ms):                        0.42
    P95 TPOT (ms):                           25.40
    P99 TPOT (ms):                           28.30
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           2.70
    Median ITL (ms):                         0.00
    P95 ITL (ms):                            25.13
    P99 ITL (ms):                            28.02
    ==================================================
    ```
