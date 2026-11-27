# 可観測性 { #observability }

Prometheus と Grafana で GPUStack のメトリクスを監視し、推論ログとトレースで個々のリクエストを調べる方法を説明します。

## 概要 { #overview }

GPUStack は、モデル提供と GPU リソース管理に関する幅広いメトリクスを提供します。Prometheus と Grafana を連携させることで、メトリクスをリアルタイムに収集、保存、可視化し、効率的な監視とトラブルシューティングを行えます。

## モデルのネイティブ監視 { #native-model-monitoring }

**Model Service → Deployments** でデプロイ名を開くと、Overview、Monitoring、Instances、GPU Resources、Configuration を表示できます。監視グループは個別に展開でき、時間範囲、インスタンス、PD ロールのフィルターを共有します。

- ランタイムのグラフは推論エンジンのメトリクスを使用します。欠けたエンジンサンプルをゲートウェイの値で置き換えません。PD のレイテンシーには観測元のロールを表示し、ローカルのサブリクエストとして扱います。
- 対応するレイテンシー統計は Avg、P50、P95、P99 です。統計行は選択期間全体、曲線の各点はページに表示される短いレート期間を対象とします。欠測をゼロにせず、未対応、対象外、取得失敗、古いデータを区別します。
- GPU とホストの履歴には所有者付きの安定した識別子を使用します。新しい履歴系列を収集するには Worker を更新してください。古い Worker の最新報告値は表示できますが、GPU インデックスから未検証の過去データを推測しません。物理値には共有ワークロードの使用量も含まれます。
- ランタイム履歴は現在のインスタンスと観測済みの実行世代、ハードウェア履歴は現在関連付けられたデバイスと記録された所有者が対象です。完全な過去の配置監査ではありません。

モデルの監視とダッシュボードへのリダイレクト、ベンチマークの監視には、組織の Owner ロールまたはプラットフォーム管理者権限が必要です。GPUStack は、要求されたデプロイまたはベンチマークと、関連リソースの閲覧権限も確認します。

Worker と GPU の監視およびダッシュボードへのリダイレクトには、対象リソースの閲覧権限を使用します。組織のメンバーも、閲覧できるリソースについてはこれらの API を利用できます。Grafana と Prometheus のプロキシ API はそれぞれ独自の権限を確認します。

意味付け済みの API は `GET /v2/models/{id}/metrics` と `GET /v2/models/{id}/resource-metrics` です。制御されたフィルターのみ受け付け、クライアントの PromQL は受け付けません。グラフ、表、CSV は同じ認可済み結果を使用します。

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

   `tunnel` プロキシモードの Worker は、サーバーのフォワードプロキシ（既定ポート `30079`）を経由してのみアクセスできます。このプロキシは中継するすべてのリクエストを認証します。これらの Worker のメトリクスをスクレイプするには、プロキシホップの資格情報として GPUStack API キー（UI で作成し、そのままコピーします。生成されたキーは `gpustack_<access_key>_<secret_key>` の形式です）を提示するジョブを追加します：

   ```yaml
     - job_name: gpustack-proxy-worker-discovery
       scrape_interval: 5s
       proxy_url: "http://<gpustack_server_host>:30079"
       authorization:
         credentials: "<gpustack_api_key>"
       http_sd_configs:
         - url: "http://<gpustack_server_host>:10161/metrics/proxy-targets"
           refresh_interval: 1m
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

## 推論ログとトレース { #inference-logs-and-traces }

レスポンスヘッダーの `X-GPUStack-Request-Id` を使って、推論ログとトレースを検索できます。ログまたはトレースを構成すると、組み込みゲートウェイがこのヘッダーを追加します。Docker のサーバーは、デフォルトで組み込み Tempo にトレースをエクスポートします。Helm のサーバーリリースでは、デフォルトで専用の Tempo ワークロードがデプロイされます。ログは別途設定し、ログ用エンドポイントを指定します。

| トレースの送信先・設定 | Docker | Helm |
| --- | --- | --- |
| 組み込み Tempo | サーバーでデフォルト | サーバーリリースでデフォルト |
| 外部の OTLP/gRPC レシーバー | `GPUSTACK_TRACE_OTLP_ENDPOINT` を設定 | `tracing.endpoint` を設定 |
| 組み込み Tempo のブロック保持期間（デフォルトは `168h`） | `GPUSTACK_BUILTIN_TEMPO_RETENTION` を設定 | `tracing.tempo.retention` を設定 |
| 無効 | `GPUSTACK_DISABLE_TRACING=true` を設定 | `tracing.enabled=false` を設定 |

Tempo は定期クリーンアップで保持期間を過ぎたブロックを削除します。保持期間はブロックの経過時間を基準にします。ディスクの使用量は、取り込まれるトレースの量や、書き込み先行ログ（WAL）、コンパクション、削除待ちのブロックに必要な容量によって変わります。

保持ブロックの必要容量は、1 秒あたりに取り込むトレース件数に、トレース 1 件あたりのディスク上の平均ブロックサイズ（バイト単位）と保持秒数を掛けて見積もります。計算例として、毎秒 10 リクエスト、サンプリング率 100%、サンプリングしたリクエストあたりトレース 1 件、トレース 1 件あたりのブロックサイズを 1,000 バイトと仮定します。ブロックの容量は 1 日あたり約 864 MB、7 日間で約 6.05 GB です。仮に 50% の余裕を加えると約 9.1 GB です。実際の利用を代表するワークロードで、トレース 1 件あたりのブロックサイズと WAL、コンパクション、削除待ちのブロックのピーク使用量を計測し、必要容量と余裕を決めてください。

デプロイの詳細画面にある `Request Tracing` タブでは、最初に直近 24 時間を検索します。トレースの表示には、組織のオーナーロールまたはプラットフォーム管理者権限が必要です。GPUStack はデプロイのメトリクスと同じモデルの参照範囲のルールを適用します。結果をページ単位で確認するか、GPUStack や Tempo に表示される Request ID または Trace ID をそのまま貼り付けて検索できます。ログのエクスポートを設定している場合は、リクエスト詳細から Request ID をコピーし、ログ基盤で検索してください。外部 Tempo を使う場合は、`GPUSTACK_TRACE_QUERY_URL` または `tracing.queryURL` でクエリーエンドポイントを設定します。

外部 Tempo を使用する場合、各 GPUStack 環境には専用の Tempo テナントまたは専用の Tempo インスタンスが必要です。その環境のトレースの書き込みとクエリはすべて同じテナントに転送してください。モデル ID は各 GPUStack 環境内で採番されるため、テナントを共有すると、別の環境で同じモデル ID を持つリクエストが表示されるおそれがあります。

ログのエクスポートを有効にすると、サーバーとワーカーは JSON 形式のログをコンテナーのコンソールに出力します。Alloy はこのログと、対応する形式のエンジンログを読み取ります。破棄されたレコードを確認するには、Alloy の `otelcol_processor_filter_logs_filtered_total` メトリクスを `engine_format` または `higress_format` コンポーネントで確認します。エンジンのフィルターは、認識できるログプレフィックスを持たない単独のプレーンテキスト行を除外します。

標準の vLLM および SGLang の起動では、GPUStack が有効なログのエクスポート（`vLLM >= 0.10.1`、`SGLang >= 0.5.9`）および管理対象ランナーイメージでのトレースのエクスポート（`vLLM >= 0.17.0`、`SGLang >= 0.5.12`）を自動設定します。管理対象の PD Router レシピは、それぞれがサポートする可観測性のデフォルト設定を適用します。旧バージョンやカスタムイメージは既存の起動動作を維持し、明示的なテレメトリー設定が常に優先されます。

特定のデプロイのエンジンとマネージド PD Router の自動トレースを無効にするには、そのデプロイの詳細設定の環境変数に `GPUSTACK_DISABLE_ENGINE_TRACING=1` を設定し、インスタンスを再起動します。ゲートウェイと Worker のトレースは引き続きグローバル設定に従います。変数を削除すると次回の起動時に自動トレースが復元されます。

各トレースコンポーネントは、選択したレシーバーにスパンを直接送信します。ログエンドポイントには、ログをエクスポートするコンポーネントから到達できる必要があります。トレースエンドポイントには、トレースをエクスポートするサーバー、ワーカー、ゲートウェイ、推論コンテナーから到達できる必要があります。取り込みには平文の gRPC を使うため、信頼できるネットワークのアドレスを指定してください。外部レシーバーに切り替えたりトレースを無効にしたりすると、組み込み Tempo は停止します。

### Docker の設定 { #configure-docker }

1. ログのエクスポートを有効にするには、サーバーとログをエクスポートする各ワーカーに `GPUSTACK_LOG_OTLP_ENDPOINT=http://<collector-host>:4318/v1/logs` を設定します。`/var/run/docker.sock:/var/run/docker.sock` をマウントし、`docker logs` に対応するロギングドライバーを使用します。
2. 外部レシーバーにトレースを送る場合は、サーバーと個別に起動する各ワーカーに `GPUSTACK_TRACE_OTLP_ENDPOINT=<collector-host>:4317` を設定します。リモートワーカーで組み込み Tempo を使う場合は、[既存の Docker ワーカーでトレースを有効にする](../upgrade/cluster.md#enable-tracing-on-existing-docker-workers)を参照してください。
3. `<collector-host>` をコンテナーから到達できる名前またはアドレスに置き換えます。`docker run -e` または Compose ファイルの `environment` で変数を渡し、コンテナーを再作成します。

Docker は GPUStack コンテナー内で Tempo と Alloy を実行します。Tempo は WAL とブロックを `<data_dir>/tempo` 配下に保存し、クエリーと OTLP 取り込みの両方をデフォルトでループバック（`127.0.0.1`）のみで待ち受けます。個別に実行する Docker ワーカーは、ポート `4317` を公開せずに、認証済みのサーバーのトレース取り込み API を経由してワーカーとエンジンのスパンを自動転送します。トレースファイルは、サーバーのデータボリュームのライフサイクルに従います。ブロックの保持期間を変更するには、サーバーに `GPUSTACK_BUILTIN_TEMPO_RETENTION` を設定し、コンテナーを再作成します。代わりに信頼できるネットワーク上のリモートワーカーから直接 OTLP スパンを受け付けるには、サーバーに `GPUSTACK_BUILTIN_TEMPO_OTLP_LISTEN_ADDRESS` を設定します。

### Helm の設定 { #configure-helm }

1. 各 GPUStack リリースの Helm values にエンドポイントを設定します。`logging.endpoint` はログのエクスポートを有効にし、`tracing.endpoint` は外部のトレースレシーバーを指定します。ワーカー専用リリースでは、`tracing.endpoint` にサーバーのレシーバーを指定できます。

   ```yaml
   logging:
     endpoint: "http://<collector-host>:4318/v1/logs"
   tracing:
     endpoint: "<collector-host>:4317"
   ```

2. `<collector-host>` をレシーバーのアドレスに置き換え、Helm リリースをアップグレードします。別の名前空間にある Collector には、名前空間を含む Service の DNS 名を使います。

組み込み Higress ゲートウェイは、デフォルトで JSON 形式のアクセスログを書き込みます。形式を変えるには、Helm values の `higress-core.meshConfig.accessLogEncoding` と `higress-core.meshConfig.accessLogFormat` を設定します。

Tempo と Alloy のイメージは `global.hub` を使用します。デフォルトは `docker.io` です。プライベートレジストリーへミラーリングする場合は、`grafana/tempo` と `grafana/alloy` のリポジトリーパスを維持してください。

チャートはサーバー Pod から独立して Tempo を実行し、デフォルトで `/var/tempo` に `emptyDir` をマウントします。Tempo Pod を置き換えると、このボリュームに保存したトレースは失われます。Pod の置き換え後もトレースを保持するには、リリースと同じ名前空間にある既存の PVC を `tracing.tempo.existingClaim` に指定します。ボリュームへの書き込みを UID またはグループ `10001` に許可してください。ブロックの保持期間を変更するには、`tracing.tempo.retention` を設定してリリースをアップグレードします。 保持期間が空または無効な場合は `168h` が使われます。

すべてのサーバー Pod は、サーバーが 1 つのリリースでも、相互 TLS を使って `https://<release>-tempo.<namespace>.svc.<cluster-domain>:3200` の組み込み Tempo に問い合わせます。チャートはクエリー用 TLS Secret をデフォルトで作成します。ワーカーと推論エンジンは Service の OTLP/gRPC ポート `4317` にトレースを送信します。

GitOps デプロイでは、サーバーリリースの名前空間に Secret を作成し、その名前を `tracing.tempo.queryTLSSecretName` に設定します。Secret には `ca.crt`、`server.crt`、`server.key`、`client.crt`、`client.key` を含めます。サーバー証明書は `<release>-tempo.<namespace>.svc.<cluster-domain>` をカバーする必要があります。サーバー証明書とクライアント証明書の更新には同じ CA を使います。新しい Secret を作成して Helm values の名前を更新すると、Tempo とサーバーの Pod が新しい証明書で更新されます。

トレースのエクスポート先を変更した後は、稼働中のモデルインスタンスを再起動するか、再デプロイします。

### エクスポートするログのマスキング { #redact-exported-logs }

Alloy は、エクスポート前にログ本文、例外のスタックトレース、`path` の一致箇所を `***` に置き換えられます。空でない RE2 正規表現のリストを指定します。空のリストではマスキングを行いません。リクエスト、トレース、スパンの ID は相互参照に使えます。

Docker では、ログをエクスポートする各コンテナーに `GPUSTACK_LOG_REDACT_PATTERNS` を JSON の文字列配列として渡します。次の例は、メールアドレスと中国本土の携帯電話番号をマスクします。

```bash
GPUSTACK_LOG_REDACT_PATTERNS='["(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+[.][A-Z]{2,}","\\b1[3-9][0-9]{9}\\b"]'
```

Helm では、ログをエクスポートする各リリースに同等の YAML リストを設定します。

```yaml
logging:
  redactPatterns:
    - '(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+[.][A-Z]{2,}'
    - '\b1[3-9][0-9]{9}\b'
```

ルールを適用するには、Docker コンテナーを再作成するか Helm リリースをアップグレードします。ログ基盤で `request_id` を使い、エクスポートされたレコードを確認してください。ルールは、GPUStack が管理する Alloy が新たにエクスポートするログに適用されます。元のレコードは `docker logs`、`kubectl logs`、またはランタイムのログファイルで参照できます。すでにエクスポートされたレコードやトレースはルールの対象外です。無効なルールによって Alloy のログエクスポートが止まる場合も、GPUStack はリクエストの処理を続けます。

### リクエストの検索 { #look-up-a-request }

1. 稼働中のチャットモデルにリクエストを送り、レスポンスヘッダーを保存します。`<gpustack-host>`、`<api-key>`、`<model-name>` を実際のデプロイの値に置き換えてください。推論の権限については、[API キー管理](api-key-management.md)を参照してください。

   ```bash
   curl --silent --show-error --no-buffer \
     --dump-header response-headers.txt \
     'http://<gpustack-host>/v1/chat/completions' \
     --header 'Authorization: Bearer <api-key>' \
     --header 'Content-Type: application/json' \
     --data '{"model":"<model-name>","messages":[{"role":"user","content":"Reply with one word."}],"max_tokens":8,"stream":true}'

   grep -i '^X-GPUStack-Request-Id:' response-headers.txt
   ```

2. ログ基盤でリクエスト前後の時間範囲を指定し、ログ属性 `request_id` で絞り込みます。OTLP リソース属性 `service.name` では、`higress`、`gpustack.server`、`gpustack.engine` などの送信元を選べます。レコードに `worker_id`、`model_instance_id`、`role` がある場合は、これらでも絞り込めます。
3. ログレコードに OTLP の Trace ID がある場合は、その ID をトレース基盤で検索します。トレースが有効なら、ログのエクスポートを無効にしていても、スパン属性 `gpustack.request_id` でプラットフォームの Request ID を検索できます。

### Loki と Grafana の例 { #example-with-loki-and-grafana }

1. [OTLP の取り込みと構造化メタデータを有効にした](https://grafana.com/docs/loki/latest/send-data/otel/) Loki では、`GPUSTACK_LOG_OTLP_ENDPOINT` または `logging.endpoint` に、`http://<loki-host>:3100/otlp/v1/logs` のような取り込み先の完全な URL を指定します。トレース用の `GPUSTACK_TRACE_OTLP_ENDPOINT` または `tracing.endpoint` には、スパンを送信するコンポーネントが到達できる Tempo の OTLP/gRPC アドレスを指定します。Grafana のデータソースには、Loki と Tempo の HTTP クエリー URL を設定します。
2. Grafana の `Explore` で Loki を選択し、リクエストの時間範囲を指定します。`<request-id>` をレスポンスヘッダー `X-GPUStack-Request-Id` の値に置き換えてください。Loki のデフォルトの OTLP マッピングでは、`service.name` はストリームラベル `service_name` に変換され、`request_id` は構造化メタデータになります。

   ```logql
   {service_name=~"higress|gpustack[.](server|worker|engine)"} | request_id="<request-id>"
   ```

   レコードに該当するフィールドがある場合は、`| worker_id="<worker-id>"` または `| model_instance_id="<instance-id>"` を追加して絞り込めます。
3. 結果に `trace_id` がある場合は、その値を Tempo データソースの `TraceQL` クエリー欄に貼り付けます。ログからトレースを直接開くには、Loki の [派生フィールド](https://grafana.com/docs/grafana/latest/datasources/loki/configure/#derived-fields)をタイプ `Label`、マッチ条件 `^trace_id$` で設定します。Tempo データソースへの内部リンクを選び、クエリーに `${__value.raw}` を指定します。

保持期間とアクセス権は外部基盤で管理します。テナントユーザーに検索を許可する場合は、その基盤でテナント間を分離してください。

!!! tip "PD のエンジンログを探す"

    SGLang Mooncake では、`request_id` で両方のエンジンログを検索し、`pd_attempt_id` でディスパッチ試行を区別します。vLLM NIXL では、`request_id` で入口のログを見つけ、その `pd_coordinator_id` で両方のエンジンログを検索します。入口のログにこのフィールドがない場合は、リクエスト時刻とインスタンスで Prefill のログを絞り込み、該当するレコードの `pd_coordinator_id` をコピーします。

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
