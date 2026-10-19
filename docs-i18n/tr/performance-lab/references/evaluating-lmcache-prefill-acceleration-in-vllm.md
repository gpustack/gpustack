---
translation_status: translated
---

# vLLM'de LMCache Prefill Hızlandırmasının Değerlendirilmesi { #evaluating-lmcache-prefill-acceleration-in-vllm }

LMCache, büyük ölçekli dağıtımların temel sorunlarını çözmek için tasarlanmış, LLM çıkarımına yönelik genişletilebilir bir KV önbellek katmanıdır. Bu belge LMCache'in vLLM çıkarım performansına etkisini, özellikle prefill hızlanmasını ve farklı iş yüklerine yansımalarını değerlendirir.

## Sonuçlar { #conclusions }

1. **LMCache, yüksek önbellek isabet oranlarında prefill aşamasını önemli ölçüde hızlandırır**. Deneylerde uzun bağlamlı (20K token) çok turlu konuşmalarda giriş TPS artışı +355.3%'e, TTFT azalması -58.8%'e ulaşmıştır.

2. **Performans kazançları iş yüküne büyük ölçüde bağlıdır**:
   - **En uygun senaryolar**: ortak önekler ve yinelenen kalıplar içeren çok turlu konuşmalar
   - **Daha az uygun senaryolar**: önbellek yeniden kullanım kalıbı olmayan rastgele girdiler

3. **Blok boyutu optimizasyonu**: test edilen yapılandırmalarda varsayılan 256 blok boyutu en iyi sonucu vermiştir.

4. **Önbellek isabetsizlikleri ek yük doğurur**. Önbellek yeniden kullanılmadığında -3% ile -15% arasında performans kaybı görülmüştür. Bu nedenle LMCache öngörülebilir önek kalıpları olan iş yüklerine en uygundur.

## Teknik arka plan { #technical-background }

### LMCache'e genel bakış { #lmcache-overview }

LMCache, vLLM'in KV önbelleği yeteneklerini şu yollarla genişletir:

| Bileşen | Açıklama |
|-----------|-------------|
| CPU'ya aktarma | Önbellek kapasitesini GPU VRAM sınırlarının ötesine genişletir |
| Blok tabanlı yönetim | Yapılandırılabilir blok boyutlarıyla verimli önbellek saklama ve alma |
| Birden çok arka uç | Yerel depolama, Redis ve Mooncake gibi özel arka uçları destekler |
| Dağıtık KV önbelleği | Birden çok vLLM örneği arasında paylaşılan önbellek |

### Başlıca kullanım senaryoları { #key-use-cases }

1. **Düşük önek önbelleği isabet oranları**: GPU VRAM sınırlarını ve önbellek tahliye sorunlarını hafifletir
2. **Dağıtık önbellek paylaşımı**: birden çok vLLM örneği arasında önbellek paylaşımını sağlar
3. **PD ayrıştırması**: ayrıştırılmış dağıtım mimarilerini destekler

## Deney kurulumu { #experimental-setup }

- **Model**: Qwen3-8B
- **Donanım**: NVIDIA RTX 4090 24GB
- **vLLM sürümü**: v0.10.1.1
- **Benchmark yöntemi**: Çok turlu konuşma benchmark testi

??? info "Model sunum komutları"

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

??? info "Benchmark betikleri"

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

## Deney sonuçları { #experimental-results }

### Çok turlu konuşma performansı { #multi-turn-conversation-performance }

#### 5K giriş token'ı { #5k-input-tokens }
| Yapılandırma | Giriş TPS | Toplam TPS | Ortalama TTFT (ms) | Ortalama TPOT (ms) |
|---------------|-----------|-----------|----------------|----------------|
| LMCache olmadan | 5849 | 5957 | 4350.48 | 48.47 |
| LMCache ile | 9426 (+61.2%) | 9592 | 2646.09 (-39.2%) | 30.60 (-36.9%) |

#### 20K giriş token'ı { #20k-input-tokens }
| Yapılandırma | Giriş TPS | Toplam TPS | Ortalama TTFT (ms) | Ortalama TPOT (ms) |
|---------------|-----------|-----------|----------------|----------------|
| LMCache olmadan | 4312.17 | 4335.71 | 5070.52 | 33.91 |
| LMCache ile | 7750.60 (+79.7%) | 7792.92 | 2091.00 (-58.8%) | 25.83 (-23.8%) |

#### 20K giriş token'ı + 1 çıkış token'ı { #20k-input-tokens-1-output-token }
| Yapılandırma | Giriş TPS | Toplam TPS | Ortalama TTFT (ms) |
|---------------|-----------|-----------|----------------|
| LMCache olmadan | 7443.2 | 7443.6 | 4658.66 |
| LMCache ile | 33887.9 (+355.3%) | 33889.8 | 980.87 |

### Blok boyutunu ayarlama { #tuning-chunk-size }

| Blok boyutu | Giriş TPS | Performans kazancı | Ortalama TTFT (ms) |
|------------|-----------|------------------|----------------|
| 64 | 33820.3 | +354.4% | 985.28 |
| 256 | 33887.9 | +355.3% | 980.87 |
| 1024 | 31634.0 | +325.0% | 1055.69 |

### Önbellek isabetsizliği senaryoları (rastgele veri kümesi) { #cache-miss-scenarios-random-dataset }

??? info "Benchmark betikleri"

    ```
    vllm bench serve --model Qwen/Qwen3-8B --endpoint-type openai-chat --endpoint /v1/chat/completions --dataset-name random --random-input-len 1024 --random-output-len 128 --num-prompts 100 --seed 40
    ```

#### 1K giriş token'ı { #1k-input-tokens }
| Metrik | LMCache olmadan | LMCache ile | Değişim |
|--------|-----------------|---------------|---------|
| Çıkış TPS | 579.86 | 561.44 | -3.2% |
| Toplam TPS | 5212.32 | 5046.72 | -3.2% |
| Ortalama TTFT (ms) | 8886.36 | 9242.72 | +4.0% |
| Ortalama TPOT (ms) | 42.08 | 43.47 | +3.3% |

#### 8K giriş token'ı { #8k-input-tokens }
| Metrik | LMCache olmadan | LMCache ile | Değişim |
|--------|-----------------|---------------|---------|
| Çıkış TPS | 77.87 | 66.77 | -14.3% |
| Toplam TPS | 5060.79 | 4338.96 | -14.3% |
| Ortalama TTFT (ms) | 80610.70 | 92682.22 | +15.0% |
| Ortalama TPOT (ms) | 43.33 | 42.27 | -2.4% |

#### 20K giriş token'ı { #20k-input-tokens_1 }
| Metrik | LMCache olmadan | LMCache ile | Değişim |
|--------|-----------------|---------------|---------|
| Çıkış TPS | 22.97 | 21.77 | -5.2% |
| Toplam TPS | 3698.09 | 3504.41 | -5.2% |
| Ortalama TTFT (ms) | 277456.13 | 292811.62 | +5.5% |
| Ortalama TPOT (ms) | 31.68 | 32.80 | +3.5% |



### Tüm VRAM KV önbelleği isabet senaryoları { #all-vram-kv-cache-hit-scenarios }

#### 1K giriş token'ı { #1k-input-tokens_1 }
| Metrik | LMCache olmadan | LMCache ile | Değişim |
|--------|-----------------|---------------|---------|
| Çıkış TPS | 5954.33 | 5752.71 | -3.3% |
| Toplam TPS | 53589.01 | 51802.45 | -3.3% |
| Ortalama TTFT (ms) | 3052.08 | 3247.10 | +6.4% |
| Ortalama TPOT (ms) | 38.40 | 39.04 | +1.7% |

#### 8K giriş token'ı { #8k-input-tokens_1 }
| Metrik | LMCache olmadan | LMCache ile | Değişim |
|--------|-----------------|---------------|---------|
| Çıkış TPS | 3676.71 | 3656.30 | -0.6% |
| Toplam TPS | 238986.41 | 237659.44 | -0.6% |
| Ortalama TTFT (ms) | 5060.41 | 5326.37 | +5.3% |
| Ortalama TPOT (ms) | 54.37 | 53.86 | -1.0% |

#### 20K giriş token'ı { #20k-input-tokens_2 }
| Metrik | LMCache olmadan | LMCache ile | Değişim |
|--------|-----------------|---------------|---------|
| Çıkış TPS | 2213.12 | 1972.32 | -10.9% |
| Toplam TPS | 356312.70 | 317543.74 | -10.9% |
| Ortalama TTFT (ms) | 9649.76 | 10109.51 | +4.8% |
| Ortalama TPOT (ms) | 87.10 | 94.26 | +8.2% |

### Uzak depolama arka ucu performansı (20K token TTFT) { #remote-storage-backend-performance-20k-tokens-ttft }

| Arka uç | Önbellek isabetsizliği (s) | Önbellek isabeti (s) | Hızlanma |
|---------|----------------|---------------|-------------------|
| lmcache_server | 0.739 | 0.324 | 2.28x |
| Redis | 0.746 | 0.388 | 1.92x |
| Mooncake (TCP) | 0.759 | 0.362 | 2.10x |
