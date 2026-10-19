---
translation_status: translated
---

# vLLM における LMCache のプリフィル高速化の評価 { #evaluating-lmcache-prefill-acceleration-in-vllm }

LMCache は、LLM 推論向けの拡張可能な KV キャッシュ層であり、大規模デプロイにおける主要な課題への対応を目的としています。本書では、vLLM 推論に対する LMCache の性能上の効果、特にプリフィルの高速化とさまざまなワークロードへの影響を評価します。

## 結論 { #conclusions }

1. キャッシュヒット率が高い場合、**LMCache はプリフィルを大幅に高速化します**。実験では、長コンテキスト（20K トークン）の複数ターン対話で、入力 TPS が最大 +355.3% 改善し、TTFT が -58.8% 短縮しました。

2. **性能上の利点はワークロードに大きく依存します**：
   - **適したシナリオ**：共通プレフィックスや繰り返しパターンを含む複数ターンの対話
   - **適さないシナリオ**：キャッシュを再利用できるパターンのないランダムな入力

3. **チャンクサイズの最適化**：テストした構成では、デフォルトのチャンクサイズ 256 が最も良い結果を示しました。

4. **キャッシュミスはオーバーヘッドを生みます**。キャッシュを再利用できない場合は -3% ～ -15% の性能低下が見られました。このため、LMCache はプレフィックスのパターンを予測できるワークロードに最も適しています。

## 技術的な背景 { #technical-background }

### LMCache の概要 { #lmcache-overview }

LMCache は、次の機能で vLLM の KV キャッシュを拡張します：

| 要素 | 説明 |
|-----------|-------------|
| CPU へのオフロード | GPU VRAM の制約を超えてキャッシュ容量を拡張 |
| チャンク単位の管理 | 設定可能なチャンクサイズでキャッシュを効率的に保存・取得 |
| 複数バックエンド | ローカルストレージ、Redis、Mooncake などのカスタムバックエンドに対応 |
| 分散 KV キャッシュ | 複数の vLLM インスタンス間でキャッシュを共有 |

### 主なユースケース { #key-use-cases }

1. **プレフィックスキャッシュのヒット率が低い場合**：GPU VRAM の制約やキャッシュ追い出しの問題を緩和します
2. **分散キャッシュ共有**：複数の vLLM インスタンス間でキャッシュを共有できます
3. **PD 分離**：プリフィルとデコードを分離したデプロイ構成をサポートします

## 実験環境 { #experimental-setup }

- **モデル**: Qwen3-8B
- **ハードウェア**: NVIDIA RTX 4090 24GB
- **vLLM バージョン**: v0.10.1.1
- **ベンチマーク手法**: 複数ターン対話のベンチマーク

??? info "推論サービスのコマンド"

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

??? info "ベンチマークスクリプト"

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

## 実験結果 { #experimental-results }

### 複数ターン対話の性能 { #multi-turn-conversation-performance }

#### 5K 入力トークン { #5k-input-tokens }
| 構成 | 入力 TPS | 合計 TPS | 平均 TTFT (ms) | 平均 TPOT (ms) |
|---------------|-----------|-----------|----------------|----------------|
| LMCache なし | 5849 | 5957 | 4350.48 | 48.47 |
| LMCache あり | 9426 (+61.2%) | 9592 | 2646.09 (-39.2%) | 30.60 (-36.9%) |

#### 20K 入力トークン { #20k-input-tokens }
| 構成 | 入力 TPS | 合計 TPS | 平均 TTFT (ms) | 平均 TPOT (ms) |
|---------------|-----------|-----------|----------------|----------------|
| LMCache なし | 4312.17 | 4335.71 | 5070.52 | 33.91 |
| LMCache あり | 7750.60 (+79.7%) | 7792.92 | 2091.00 (-58.8%) | 25.83 (-23.8%) |

#### 20K 入力トークン + 1 出力トークン { #20k-input-tokens-1-output-token }
| 構成 | 入力 TPS | 合計 TPS | 平均 TTFT (ms) |
|---------------|-----------|-----------|----------------|
| LMCache なし | 7443.2 | 7443.6 | 4658.66 |
| LMCache あり | 33887.9 (+355.3%) | 33889.8 | 980.87 |

### チャンクサイズの調整 { #tuning-chunk-size }

| チャンクサイズ | 入力 TPS | 性能改善率 | 平均 TTFT (ms) |
|------------|-----------|------------------|----------------|
| 64 | 33820.3 | +354.4% | 985.28 |
| 256 | 33887.9 | +355.3% | 980.87 |
| 1024 | 31634.0 | +325.0% | 1055.69 |

### キャッシュミスのシナリオ（ランダムデータセット） { #cache-miss-scenarios-random-dataset }

??? info "ベンチマークスクリプト"

    ```
    vllm bench serve --model Qwen/Qwen3-8B --endpoint-type openai-chat --endpoint /v1/chat/completions --dataset-name random --random-input-len 1024 --random-output-len 128 --num-prompts 100 --seed 40
    ```

#### 1K 入力トークン { #1k-input-tokens }
| 指標 | LMCache なし | LMCache あり | 変化 |
|--------|-----------------|---------------|---------|
| 出力 TPS | 579.86 | 561.44 | -3.2% |
| 合計 TPS | 5212.32 | 5046.72 | -3.2% |
| 平均 TTFT (ms) | 8886.36 | 9242.72 | +4.0% |
| 平均 TPOT (ms) | 42.08 | 43.47 | +3.3% |

#### 8K 入力トークン { #8k-input-tokens }
| 指標 | LMCache なし | LMCache あり | 変化 |
|--------|-----------------|---------------|---------|
| 出力 TPS | 77.87 | 66.77 | -14.3% |
| 合計 TPS | 5060.79 | 4338.96 | -14.3% |
| 平均 TTFT (ms) | 80610.70 | 92682.22 | +15.0% |
| 平均 TPOT (ms) | 43.33 | 42.27 | -2.4% |

#### 20K 入力トークン { #20k-input-tokens_1 }
| 指標 | LMCache なし | LMCache あり | 変化 |
|--------|-----------------|---------------|---------|
| 出力 TPS | 22.97 | 21.77 | -5.2% |
| 合計 TPS | 3698.09 | 3504.41 | -5.2% |
| 平均 TTFT (ms) | 277456.13 | 292811.62 | +5.5% |
| 平均 TPOT (ms) | 31.68 | 32.80 | +3.5% |



### すべての VRAM KV キャッシュがヒットするシナリオ { #all-vram-kv-cache-hit-scenarios }

#### 1K 入力トークン { #1k-input-tokens_1 }
| 指標 | LMCache なし | LMCache あり | 変化 |
|--------|-----------------|---------------|---------|
| 出力 TPS | 5954.33 | 5752.71 | -3.3% |
| 合計 TPS | 53589.01 | 51802.45 | -3.3% |
| 平均 TTFT (ms) | 3052.08 | 3247.10 | +6.4% |
| 平均 TPOT (ms) | 38.40 | 39.04 | +1.7% |

#### 8K 入力トークン { #8k-input-tokens_1 }
| 指標 | LMCache なし | LMCache あり | 変化 |
|--------|-----------------|---------------|---------|
| 出力 TPS | 3676.71 | 3656.30 | -0.6% |
| 合計 TPS | 238986.41 | 237659.44 | -0.6% |
| 平均 TTFT (ms) | 5060.41 | 5326.37 | +5.3% |
| 平均 TPOT (ms) | 54.37 | 53.86 | -1.0% |

#### 20K 入力トークン { #20k-input-tokens_2 }
| 指標 | LMCache なし | LMCache あり | 変化 |
|--------|-----------------|---------------|---------|
| 出力 TPS | 2213.12 | 1972.32 | -10.9% |
| 合計 TPS | 356312.70 | 317543.74 | -10.9% |
| 平均 TTFT (ms) | 9649.76 | 10109.51 | +4.8% |
| 平均 TPOT (ms) | 87.10 | 94.26 | +8.2% |

### リモートストレージバックエンドの性能（20K トークンの TTFT） { #remote-storage-backend-performance-20k-tokens-ttft }

| バックエンド | キャッシュミス (s) | キャッシュヒット (s) | 高速化率 |
|---------|----------------|---------------|-------------------|
| lmcache_server | 0.739 | 0.324 | 2.28x |
| Redis | 0.746 | 0.388 | 1.92x |
| Mooncake (TCP) | 0.759 | 0.362 | 2.10x |
