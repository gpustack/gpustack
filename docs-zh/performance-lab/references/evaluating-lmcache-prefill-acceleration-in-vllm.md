---
translation_status: translated
---

# 评估 LMCache 在 vLLM 中的预填充加速效果 { #evaluating-lmcache-prefill-acceleration-in-vllm }

LMCache 是面向 LLM 推理的可扩展 KV Cache 层，旨在解决大规模部署场景中的关键问题。本文评估 LMCache 对 vLLM 推理性能的影响，重点关注预填充阶段加速及其在不同工作负载模式下的表现。

## 结论 { #conclusions }

1. 在缓存命中率高的场景中，**LMCache 能够显著加速预填充**。实验中的长上下文（20K tokens）多轮对话场景，input TPS 最高提升 +355.3%，TTFT 变化幅度达到 -58.8%。

2. **性能收益高度依赖工作负载**：
   - **理想场景**：具有共享前缀和重复模式的多轮对话
   - **非理想场景**：没有缓存复用模式的随机输入

3. **分块大小优化**：在测试配置中，默认分块大小 256 的结果最佳。

4. **缓存未命中会引入额外开销**。在无法复用缓存时，性能变化为 -3% 至 -15%。因此，LMCache 最适合具有可预测前缀模式的工作负载。

## 技术背景 { #technical-background }

### LMCache 概述 { #lmcache-overview }

LMCache 通过以下方式扩展 vLLM 的 KV cache 能力：

| 组件 | 说明 |
|-----------|-------------|
| CPU 卸载 | 扩展缓存容量，突破 GPU 显存容量限制 |
| 分块管理 | 通过可配置的分块大小高效存储和检索缓存 |
| 多种后端 | 支持本地存储、Redis，以及 Mooncake 等自定义后端 |
| 分布式 KV Cache | 多个 vLLM 实例之间共享缓存 |

### 主要使用场景 { #key-use-cases }

1. **前缀缓存命中率低**：缓解 GPU 显存容量限制和缓存驱逐问题
2. **分布式缓存共享**：支持在多个 vLLM 实例之间共享缓存
3. **PD 分离**：支持预填充与解码分离的部署架构

## 实验环境 { #experimental-setup }

- **模型**: Qwen3-8B
- **硬件**: NVIDIA RTX 4090 24GB
- **vLLM 版本**: v0.10.1.1
- **基准测试方法**：多轮对话基准测试

??? info "推理服务命令"

    ```bash
    # Standard vLLM serving
    vllm serve Qwen/Qwen3-8B

    # LMCache-enabled serving
    ##### lmcache_config.yaml
    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 50
    #####
    LMCACHE_CONFIG_FILE=/root/lmcache_config.yaml vllm serve /root/Qwen3-8B \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
    ```

??? info "基准测试脚本"

    ```bash
    # Multi-turn bench scripts
    # Ref: https://github.com/vllm-project/vllm/tree/main/benchmarks/multi_turn

    ##### generate_multi_turn.json
    {
        "filetype": "generate_conversations",
        "num_conversations": 24,
        "text_files": ["pg1184.txt"],
        "print_stats": false,
        "prompt_input": {
            "num_turns": {
                "distribution": "uniform",
                "min": 12,
                "max": 18
            },
            "common_prefix_num_tokens": {
                "distribution": "constant",
                "value": 500
            },
            "prefix_num_tokens": {
                "distribution": "lognormal",
                "average": 4000,
                "max": 20000
            },
            "num_tokens": {
                "distribution": "uniform",
                "min": 120,
                "max": 160
            }
        },
        "prompt_output": {
            "num_tokens": {
                "distribution": "uniform",
                "min": 80,
                "max": 120
            }
        }
    }
    #####

    python benchmark_serving_multi_turn.py --model $MODEL_PATH --input-file generate_multi_turn.json --num-clients 10 --max-active-conversations 10
    ```

## 实验结果 { #experimental-results }

### 多轮对话性能 { #multi-turn-conversation-performance }

#### 5K 输入 Tokens { #5k-input-tokens }
| 配置 | Input TPS | Total TPS | Mean TTFT (ms) | Mean TPOT (ms) |
|---------------|-----------|-----------|----------------|----------------|
| 不使用 LMCache | 5849 | 5957 | 4350.48 | 48.47 |
| 使用 LMCache | 9426 (+61.2%) | 9592 | 2646.09 (-39.2%) | 30.60 (-36.9%) |

#### 20K 输入 Tokens { #20k-input-tokens }
| 配置 | Input TPS | Total TPS | Mean TTFT (ms) | Mean TPOT (ms) |
|---------------|-----------|-----------|----------------|----------------|
| 不使用 LMCache | 4312.17 | 4335.71 | 5070.52 | 33.91 |
| 使用 LMCache | 7750.60 (+79.7%) | 7792.92 | 2091.00 (-58.8%) | 25.83 (-23.8%) |

#### 20K 输入 Tokens + 1 输出 Token { #20k-input-tokens-1-output-token }
| 配置 | Input TPS | Total TPS | Mean TTFT (ms) |
|---------------|-----------|-----------|----------------|
| 不使用 LMCache | 7443.2 | 7443.6 | 4658.66 |
| 使用 LMCache | 33887.9 (+355.3%) | 33889.8 | 980.87 |

### 分块大小调优 { #tuning-chunk-size }

| 分块大小 | Input TPS | 性能收益 | Mean TTFT (ms) |
|------------|-----------|------------------|----------------|
| 64 | 33820.3 | +354.4% | 985.28 |
| 256 | 33887.9 | +355.3% | 980.87 |
| 1024 | 31634.0 | +325.0% | 1055.69 |

### 缓存未命中场景（随机数据集） { #cache-miss-scenarios-random-dataset }

??? info "基准测试脚本"

    ```
    vllm bench serve --model Qwen/Qwen3-8B --endpoint-type openai-chat --endpoint /v1/chat/completions --dataset-name random --random-input-len 1024 --random-output-len 128 --num-prompts 100 --seed 40
    ```

#### 1K 输入 Tokens { #1k-input-tokens }
| 指标 | 不使用 LMCache | 使用 LMCache | 变化 |
|--------|-----------------|---------------|---------|
| Output TPS | 579.86 | 561.44 | -3.2% |
| Total TPS | 5212.32 | 5046.72 | -3.2% |
| Mean TTFT (ms) | 8886.36 | 9242.72 | +4.0% |
| Mean TPOT (ms) | 42.08 | 43.47 | +3.3% |

#### 8K 输入 Tokens { #8k-input-tokens }
| 指标 | 不使用 LMCache | 使用 LMCache | 变化 |
|--------|-----------------|---------------|---------|
| Output TPS | 77.87 | 66.77 | -14.3% |
| Total TPS | 5060.79 | 4338.96 | -14.3% |
| Mean TTFT (ms) | 80610.70 | 92682.22 | +15.0% |
| Mean TPOT (ms) | 43.33 | 42.27 | -2.4% |

#### 20K 输入 Tokens { #20k-input-tokens_1 }
| 指标 | 不使用 LMCache | 使用 LMCache | 变化 |
|--------|-----------------|---------------|---------|
| Output TPS | 22.97 | 21.77 | -5.2% |
| Total TPS | 3698.09 | 3504.41 | -5.2% |
| Mean TTFT (ms) | 277456.13 | 292811.62 | +5.5% |
| Mean TPOT (ms) | 31.68 | 32.80 | +3.5% |



### KV cache 全部在显存中命中的场景 { #all-vram-kv-cache-hit-scenarios }

#### 1K 输入 Tokens { #1k-input-tokens_1 }
| 指标 | 不使用 LMCache | 使用 LMCache | 变化 |
|--------|-----------------|---------------|---------|
| Output TPS | 5954.33 | 5752.71 | -3.3% |
| Total TPS | 53589.01 | 51802.45 | -3.3% |
| Mean TTFT (ms) | 3052.08 | 3247.10 | +6.4% |
| Mean TPOT (ms) | 38.40 | 39.04 | +1.7% |

#### 8K 输入 Tokens { #8k-input-tokens_1 }
| 指标 | 不使用 LMCache | 使用 LMCache | 变化 |
|--------|-----------------|---------------|---------|
| Output TPS | 3676.71 | 3656.30 | -0.6% |
| Total TPS | 238986.41 | 237659.44 | -0.6% |
| Mean TTFT (ms) | 5060.41 | 5326.37 | +5.3% |
| Mean TPOT (ms) | 54.37 | 53.86 | -1.0% |

#### 20K 输入 Tokens { #20k-input-tokens_2 }
| 指标 | 不使用 LMCache | 使用 LMCache | 变化 |
|--------|-----------------|---------------|---------|
| Output TPS | 2213.12 | 1972.32 | -10.9% |
| Total TPS | 356312.70 | 317543.74 | -10.9% |
| Mean TTFT (ms) | 9649.76 | 10109.51 | +4.8% |
| Mean TPOT (ms) | 87.10 | 94.26 | +8.2% |

### 远程存储后端性能（20K Tokens TTFT） { #remote-storage-backend-performance-20k-tokens-ttft }

| 后端 | 缓存未命中 (s) | 缓存命中 (s) | 性能提升 |
|---------|----------------|---------------|-------------------|
| lmcache_server | 0.739 | 0.324 | 2.28x |
| Redis | 0.746 | 0.388 | 1.92x |
| Mooncake (TCP) | 0.759 | 0.362 | 2.10x |
