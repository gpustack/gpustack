# 可観測性 { #observability }

本ドキュメントでは、Prometheus と Grafana を使って、GPUStack Server、Worker、LLM 提供ランタイムのメトリクスを監視する方法を説明します。

## 概要 { #overview }

GPUStack は、モデル提供と GPU リソース管理に関する幅広いメトリクスを提供します。Prometheus と Grafana を連携させることで、メトリクスをリアルタイムに収集、保存、可視化し、効率的な監視とトラブルシューティングを行えます。

## 組み込みの可観測性（既定） { #built-in-observability-default }

GPUStack は、既定で組み込みの Prometheus と Grafana を起動します。次のアドレスからアクセスできます。

- **Prometheus**：`http://your_gpustack_server_host_ip/prometheus`
- **Grafana**：`http://your_gpustack_server_host_ip/grafana`

組み込み Grafana は匿名の Viewer アクセスを許可し、ログインフォームは無効です。管理者の認証情報の既定値は、引き続き `admin` / `grafana` です。

## 外部の可観測性基盤（任意） { #external-observability-optional }

外部の Prometheus/Grafana スタックを使用する場合は、付属の Docker Compose ファイルを推奨します。

次のコマンドで、最新の安定版リリースをクローンします。

```bash
LATEST_TAG=$(
    curl -s "https://api.github.com/repos/gpustack/gpustack/releases" \
    | grep '"tag_name"' \
    | sed -E 's/.*"tag_name": "([^"]+)".*/\1/' \
    | grep -Ev 'rc|beta|alpha|preview' \
    | head -1
)
echo "Latest stable release: $LATEST_TAG"
git clone -b "$LATEST_TAG" https://github.com/gpustack/gpustack.git
cd gpustack/docker-compose
```

起動前に、`GPUSTACK_GRAFANA_URL` をブラウザーからアクセスできる Grafana URL に設定します。`grafana` のようなコンテナ内だけで使えるホスト名ではありません。

外部 Prometheus/Grafana を起動します（組み込みスタックは無効になります）。

```bash
sudo docker compose -f docker-compose.external-observability.yaml up -d
```

既存の外部 Prometheus/Grafana スタックがある場合は、代わりに手動で構成できます。

1. **GPUStack メトリクスを収集するように Prometheus を構成する**  
   GPUStack のメトリクスエンドポイント（既定 `:10161`）と Worker 検出エンドポイントをターゲットに追加します。例：

   ```yaml
   scrape_configs:
     - job_name: gpustack-worker-discovery
       scrape_interval: 5s
       http_sd_configs:
         - url: "http://<gpustack_server_host>:10161/metrics/targets"
           refresh_interval: 1m
     - job_name: gpustack-server
       scrape_interval: 10s
       static_configs:
         - targets: ["<gpustack_server_host>:10161"]
   ```
2. **GPUStack ダッシュボードを Grafana にインポートする**  
   `docker-compose/grafana/grafana_dashboards/` ディレクトリのダッシュボードを出発点として使用します。
3. **GPUStack の参照先を自分の Grafana に設定する**  
   ダッシュボードへのリダイレクトが機能するように、`GPUSTACK_GRAFANA_URL` を外部からアクセスできる Grafana URL に設定します。ブラウザーからアクセスできる URL が必要です。

## メトリクスへのアクセス { #accessing-metrics }

- **GPUStack メトリクスエンドポイント**：  
  `http://<gpustack_server_host>:10161/metrics` でアクセスします。
- **GPUStack Worker メトリクスのターゲット**：  
  `http://<gpustack_server_host>:10161/metrics/targets` でアクセスします。
- **Prometheus UI**：  
  既定では `http://<host>:19090`、または `--builtin-prometheus-port` / `GPUSTACK_BUILTIN_PROMETHEUS_PORT` で設定したポートにアクセスします。
- **Grafana UI**：  
  既定では `http://<host>:13000`、または `--builtin-grafana-port` / `GPUSTACK_BUILTIN_GRAFANA_PORT` で設定したポートにアクセスします。組み込み Grafana は匿名の Viewer アクセスを許可し、ログインフォームは無効です。管理者の認証情報の既定値は、引き続き `admin` / `grafana` です。

## 以前の Compose 構成からの移行 { #migration-from-older-compose-setups }

以前、Docker Compose で GPUStack とともに Prometheus/Grafana を実行していた場合：

- **外部の可観測性基盤を維持する（継続性を重視する場合に推奨）**：  
  既存の Prometheus/Grafana コンテナを実行したままにします。Prometheus の収集ターゲットを新しい GPUStack メトリクスエンドポイントに更新し、`GPUSTACK_GRAFANA_URL` を既存の Grafana に設定します。

- **組み込みの可観測性基盤に切り替える**：  
  古い Prometheus/Grafana コンテナを停止し、最新の `docker-compose.server.yaml`（GPUStack のみ）を使用します。組み込みの Grafana/Prometheus が引き継ぎます。古いスタックを読み取り専用で保持しない限り、古い Prometheus の過去のメトリクスは移行されません。

## メトリクスマッピングのカスタマイズ { #customizing-metrics-mapping }

GPUStack は、構成 API によるメトリクスマッピングの動的なカスタマイズに対応しています。ランタイムエンジンのメトリクスを GPUStack のメトリクスへマッピングする方法を、サービスを再起動せずに更新できます。構成は Server で一元管理され、HTTP API から取得または変更できます。

### API エンドポイント { #api-endpoints }

- **現在のメトリクス構成の取得**

  - GET `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config`
  - 現在のメトリクスマッピング構成を JSON 形式で返します。

- **メトリクス構成の更新**

  - POST `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config`
  - メトリクスマッピング構成を更新する JSON ペイロードを受け入れます。変更はすべての Worker ですぐに有効になります。

- **既定のメトリクス構成の取得**
  - GET `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/default-config`
  - 参照やリセットに使える、既定のメトリクスマッピング構成を JSON 形式で返します。

### 使用例 { #example-usage }

**現在の構成の取得：**

```bash
curl http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config
```

**構成の更新：**

```bash
curl -X POST http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config \
     -H "Content-Type: application/json" \
     -d @custom_metrics_config.json
```

*（`custom_metrics_config.json` は、新しい構成ファイルです）*

**既定の構成の取得：**

```bash
curl http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/default-config
```

> **注意**：構成は有効な JSON 形式で指定してください。

## GPUStack が公開するメトリクス { #metrics-exposed-by-gpustack }

次のメトリクスを GPUStack が公開し、Prometheus で収集できます。各メトリクスには、クラスター、Worker、モデル、インスタンスを識別する階層的なラベルが含まれます。

### LLM 提供ランタイムのメトリクス { #llm-serving-runtime-metrics }

| メトリクス名 | 型 | 説明 |
| --- | --- | --- |
| gpustack:num_requests_running | Gauge | 現在処理中のリクエスト数。 |
| gpustack:num_requests_waiting | Gauge | キューで待機中のリクエスト数。 |
| gpustack:num_requests_swapped | Gauge | CPU にスワップアウトされたリクエスト数。 |
| gpustack:prefix_cache_hit_rate | Gauge | プレフィックスキャッシュのヒット率。 |
| gpustack:kv_cache_usage_ratio | Gauge | KV キャッシュ使用率。1.0 は全使用を意味します。 |
| gpustack:prefix_cache_queries | Counter | プレフィックスキャッシュの問い合わせ数（合計トークン数）。 |
| gpustack:prefix_cache_hits | Counter | プレフィックスキャッシュのヒット数（合計トークン数）。 |
| gpustack:prompt_tokens | Counter | 処理したプリフィルトークンの合計数。 |
| gpustack:generation_tokens | Counter | 生成したトークンの合計数。 |
| gpustack:request_prompt_tokens | Histogram | リクエストごとに処理したプリフィルトークン数。 |
| gpustack:request_generation_tokens | Histogram | リクエストごとに処理した生成トークン数。 |
| gpustack:time_to_first_token_seconds | Histogram | 最初のトークンを生成するまでの時間。 |
| gpustack:inter_token_latency_seconds | Histogram | 前のトークンを生成してから、次のトークンを生成するまでの時間。 |
| gpustack:time_per_output_token_seconds | Histogram | 生成トークンあたりの時間。 |
| gpustack:e2e_request_latency_seconds | Histogram | エンドツーエンドのリクエストレイテンシー。 |
| gpustack:request_success | Counter | 成功したリクエストの合計数。 |
| gpustack:request_preemptions | Counter | エンジンによるプリエンプションの合計数。 |

これらのメトリクスは、metrics_config.yaml の定義に従って、vLLM、SGLang、MindIE などの各ランタイムエンジンからマッピングされます。

### Worker のメトリクス { #worker-metrics }

| メトリクス名 | 型 | 説明 |
| --- | --- | --- |
| gpustack:worker_status | Gauge | Worker の状態（state ラベル付き）。 |
| gpustack:worker_node_os | Info | Worker ノードの OS 情報。 |
| gpustack:worker_node_kernel | Info | Worker ノードのカーネル情報。 |
| gpustack:worker_node_uptime_seconds | Gauge | Worker ノードの稼働時間（秒）。 |
| gpustack:worker_node_cpu_cores | Gauge | Worker ノードの CPU コア総数。 |
| gpustack:worker_node_cpu_utilization_rate | Gauge | Worker ノードの CPU 利用率。 |
| gpustack:worker_node_memory_total_bytes | Gauge | Worker ノードの総メモリ（バイト）。 |
| gpustack:worker_node_memory_used_bytes | Gauge | Worker ノードの使用中メモリ（バイト）。 |
| gpustack:worker_node_memory_utilization_rate | Gauge | Worker ノードのメモリ使用率。 |
| gpustack:worker_node_gpu | Info | Worker ノードの GPU 情報。 |
| gpustack:worker_node_gpu_cores | Gauge | Worker ノードの GPU コア総数。 |
| gpustack:worker_node_gpu_utilization_rate | Gauge | Worker ノードの GPU 利用率。 |
| gpustack:worker_node_gpu_temperature_celsius | Gauge | GPU 温度（摂氏）。 |
| gpustack:worker_node_gpu_power_limit_watts | Gauge | GPU の電力管理上限（ワット）。 |
| gpustack:worker_node_gpu_power_used_watts | Gauge | GPU の消費電力（ワット）。 |
| gpustack:worker_node_gram_total_bytes | Gauge | GPU RAM の総量（バイト）。 |
| gpustack:worker_node_gram_allocated_bytes | Gauge | 割り当て済み GPU RAM（バイト）。 |
| gpustack:worker_node_gram_used_bytes | Gauge | 使用中の GPU RAM（バイト）。 |
| gpustack:worker_node_gram_utilization_rate | Gauge | GPU RAM の使用率。 |
| gpustack:worker_node_filesystem_total_bytes | Gauge | ファイルシステムの総容量（バイト）。 |
| gpustack:worker_node_filesystem_used_bytes | Gauge | ファイルシステムの使用量（バイト）。 |
| gpustack:worker_node_filesystem_utilization_rate | Gauge | ファイルシステムの使用率。 |

### Server のメトリクス { #server-metrics }

| メトリクス名 | 型 | 説明 |
| --- | --- | --- |
| gpustack:cluster | Info | クラスター情報（ID、名前、プロバイダー）。 |
| gpustack:cluster_status | Gauge | クラスターの状態（state ラベル付き）。 |
| gpustack:model | Info | モデル情報（ID、名前、ランタイム、ソース、カテゴリー）。 |
| gpustack:model_desired_instances | Gauge | モデルインスタンスの希望数。 |
| gpustack:model_running_instances | Gauge | 実行中のモデルインスタンス数。 |
| gpustack:model_instance_status | Gauge | 各モデルインスタンスの状態（state ラベル付き）。 |
| gpustack:model_instance_restart_count | Gauge | モデルインスタンスの再起動回数。 |
| gpustack:model_instance_latest_restart_time | Gauge | モデルインスタンスの最終再起動時刻（Unix タイムスタンプ、秒）。 |

### イベントバスのメトリクス { #event-bus-metrics }

| メトリクス名 | 型 | 説明 |
| --- | --- | --- |
| gpustack:bus_subscribers | Gauge | `topic` + `source` ごとのアクティブなバス購読者数（`source=streaming` を共有する場合があり、開いている watch ストリームごとに 1 つ）。 |
| gpustack:bus_queue_depth | Gauge | `topic`+`source` を共有する購読者間の最大キュー深度。 |
| gpustack:bus_queue_capacity | Gauge | 購読者ごとのキューの最大サイズ（後述の環境設定を参照）。 |
| gpustack:bus_queue_full | Gauge | `topic`+`source` を共有する購読者に満杯のキューがあれば 1。 |
| gpustack:bus_queue_saturation_ratio | Gauge | `[0, 1]` 内の `qsize / maxsize` の最大値。0.8 を継続して超える場合は、消費が遅いことを示します。 |
| gpustack:bus_subscriber_latest_keys | Gauge | 集約された UPDATED 配信を待つ ID の最大数（`latest_by_key` のサイズ）。 |
| gpustack:bus_events_total | Counter | `topic`+`source` を共有する購読者全体の累積イベント数の合計。追加ラベルは `kind`、`event_type`。 |

> **注意**：すべてのメトリクスには、詳細な監視と絞り込みのために、クラスター、Worker、モデル、インスタンス、ユーザーなどの関連識別子のラベルが付いています。
