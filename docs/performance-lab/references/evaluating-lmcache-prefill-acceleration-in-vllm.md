# Evaluating LMCache Prefill Acceleration in vLLM

LMCache is an extensible KV Cache Layer for LLM inference designed to address key challenges in large-scale deployment scenarios. This documentation evaluates the performance impact of LMCache on vLLM inference, particularly focusing on prefill stage acceleration and its implications for various workload patterns.

## Conclusions

1. **LMCache provides significant prefill acceleration** in scenarios with high cache hit rates, achieving up to +355.3% input TPS improvement and -58.8% reduction in TTFT for long-context (20K tokens) multi-turn conversations in the experiments.

2. **Performance benefits are highly workload-dependent**:
   - **Optimal scenarios**: Multi-turn conversations with shared prefixes and repeated patterns
   - **Suboptimal scenarios**: Random inputs with no cache reuse patterns

3. **Chunk size optimization** The default 256 chunk size shows the optimal results in tested configurations.

4. **Cache miss scenarios incur overhead**, showing -3% to -15% performance degradation when no cache reuse occurs, making LMCache most suitable for workloads with predictable prefix patterns.

## Technical Background

### LMCache Overview

LMCache extends vLLM's KV cache capabilities through:

| Component | Description |
|-----------|-------------|
| CPU Offloading | Extends cache capacity beyond GPU VRAM limits |
| Chunk-based Management | Efficient cache storage and retrieval with configurable chunk sizes |
| Multiple Backends | Support for local storage, Redis, and custom backends like Mooncake |
| Distributed KV Cache | Shared cache across multiple vLLM instances |

### Key Use Cases

1. **Low Prefix Cache Hit Rates**: Mitigates GPU VRAM limitations and cache eviction issues
2. **Distributed Cache Sharing**: Enables cache sharing across multiple vLLM instances
3. **PD Disaggregation**: Supports disaggregated deployment architectures

## Experimental Setup

- **Model**: Qwen3-8B
- **Hardware**: NVIDIA RTX 4090 24GB
- **vLLM Version**: v0.10.1.1
- **Benchmark Method**: Multi-turn conversation benchmark

??? info "Serving Commands"

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

??? info "Benchmark Scripts"

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

## Experimental Results

### Multi-turn Conversation Performance

#### 5K Input Tokens
| Configuration | Input TPS | Total TPS | Mean TTFT (ms) | Mean TPOT (ms) |
|---------------|-----------|-----------|----------------|----------------|
| Without LMCache | 5849 | 5957 | 4350.48 | 48.47 |
| With LMCache | 9426 (+61.2%) | 9592 | 2646.09 (-39.2%) | 30.60 (-36.9%) |

#### 20K Input Tokens
| Configuration | Input TPS | Total TPS | Mean TTFT (ms) | Mean TPOT (ms) |
|---------------|-----------|-----------|----------------|----------------|
| Without LMCache | 4312.17 | 4335.71 | 5070.52 | 33.91 |
| With LMCache | 7750.60 (+79.7%) | 7792.92 | 2091.00 (-58.8%) | 25.83 (-23.8%) |

#### 20K Input Tokens + 1 Output Token
| Configuration | Input TPS | Total TPS | Mean TTFT (ms) |
|---------------|-----------|-----------|----------------|
| Without LMCache | 7443.2 | 7443.6 | 4658.66 |
| With LMCache | 33887.9 (+355.3%) | 33889.8 | 980.87 |

### Tuning Chunk Size

| Chunk Size | Input TPS | Performance Gain | Mean TTFT (ms) |
|------------|-----------|------------------|----------------|
| 64 | 33820.3 | +354.4% | 985.28 |
| 256 | 33887.9 | +355.3% | 980.87 |
| 1024 | 31634.0 | +325.0% | 1055.69 |

### Cache Miss Scenarios (Random Dataset)

??? info "Benchmark Scripts"

    ```
    vllm bench serve --model Qwen/Qwen3-8B --endpoint-type openai-chat --endpoint /v1/chat/completions --dataset-name random --random-input-len 1024 --random-output-len 128 --num-prompts 100 --seed 40
    ```

#### 1K Input Tokens
| Metric | Without LMCache | With LMCache | Change |
|--------|-----------------|---------------|---------|
| Output TPS | 579.86 | 561.44 | -3.2% |
| Total TPS | 5212.32 | 5046.72 | -3.2% |
| Mean TTFT (ms) | 8886.36 | 9242.72 | +4.0% |
| Mean TPOT (ms) | 42.08 | 43.47 | +3.3% |

#### 8K Input Tokens
| Metric | Without LMCache | With LMCache | Change |
|--------|-----------------|---------------|---------|
| Output TPS | 77.87 | 66.77 | -14.3% |
| Total TPS | 5060.79 | 4338.96 | -14.3% |
| Mean TTFT (ms) | 80610.70 | 92682.22 | +15.0% |
| Mean TPOT (ms) | 43.33 | 42.27 | -2.4% |

#### 20K Input Tokens
| Metric | Without LMCache | With LMCache | Change |
|--------|-----------------|---------------|---------|
| Output TPS | 22.97 | 21.77 | -5.2% |
| Total TPS | 3698.09 | 3504.41 | -5.2% |
| Mean TTFT (ms) | 277456.13 | 292811.62 | +5.5% |
| Mean TPOT (ms) | 31.68 | 32.80 | +3.5% |



### All VRAM KV Cache Hit Scenarios

#### 1K Input Tokens
| Metric | Without LMCache | With LMCache | Change |
|--------|-----------------|---------------|---------|
| Output TPS | 5954.33 | 5752.71 | -3.3% |
| Total TPS | 53589.01 | 51802.45 | -3.3% |
| Mean TTFT (ms) | 3052.08 | 3247.10 | +6.4% |
| Mean TPOT (ms) | 38.40 | 39.04 | +1.7% |

#### 8K Input Tokens
| Metric | Without LMCache | With LMCache | Change |
|--------|-----------------|---------------|---------|
| Output TPS | 3676.71 | 3656.30 | -0.6% |
| Total TPS | 238986.41 | 237659.44 | -0.6% |
| Mean TTFT (ms) | 5060.41 | 5326.37 | +5.3% |
| Mean TPOT (ms) | 54.37 | 53.86 | -1.0% |

#### 20K Input Tokens
| Metric | Without LMCache | With LMCache | Change |
|--------|-----------------|---------------|---------|
| Output TPS | 2213.12 | 1972.32 | -10.9% |
| Total TPS | 356312.70 | 317543.74 | -10.9% |
| Mean TTFT (ms) | 9649.76 | 10109.51 | +4.8% |
| Mean TPOT (ms) | 87.10 | 94.26 | +8.2% |

### Remote Storage Backend Performance (20K Tokens TTFT)

| Backend | Cache Miss (s) | Cache Hit (s) | Performance Boost |
|---------|----------------|---------------|-------------------|
| lmcache_server | 0.739 | 0.324 | 2.28x |
| Redis | 0.746 | 0.388 | 1.92x |
| Mooncake (TCP) | 0.759 | 0.362 | 2.10x |
