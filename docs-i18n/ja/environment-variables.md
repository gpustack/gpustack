---
translation_status: translated
---

# 環境変数 { #environment-variables }

GPUStack は、設定用のさまざまな環境変数をサポートします。

ほとんどのコマンドラインパラメーターは、`GPUSTACK_` 接頭辞を付けた大文字の環境変数でも設定できます（例：`--data-dir` は `GPUSTACK_DATA_DIR` で設定可能）。

環境変数で設定できるコマンドラインパラメーターの全一覧は、[CLI リファレンス](cli-reference/start.md#available-environment-variables)をご覧ください。ここでは扱いません。

### 優先順位 { #priority-order }

設定値は、次の優先順位で適用されます（高い順）。

1. コマンドライン引数
2. 環境変数
3. 設定ファイル
4. デフォルト値

コマンドライン引数は常に環境変数を上書きし、環境変数は設定ファイルの値を上書きします。

## GPUStack コアの環境変数 { #gpustack-core-environment-variables }

これらの環境変数は、主に外部サービスとの連携に使用します。

**適用先**列は、環境変数を設定する場所を示します。

- **サーバー**：GPUStack サーバーに適用。
- **ワーカー**：GPUStack ワーカーに適用。
- **モデル**：モデルのデプロイ設定に適用。

### プロキシ設定 { #proxy-configuration }

| <div style="width:100px">変数</div> | 説明 | デフォルト | 適用先 |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------- | --------------- |
| `HTTP_PROXY`                            | HTTP プロキシの URL。例：`http://proxy-server:port` | （空） | サーバーとワーカー |
| `HTTPS_PROXY`                           | HTTPS プロキシの URL。例：`https://proxy-server:port` | （空） | サーバーとワーカー |
| `NO_PROXY`                              | 対象外とするホストのコンマ区切り一覧。例：`127.0.0.1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/16,localhost,cluster.local` | （空） | サーバーとワーカー |

### Hugging Face Hub { #hugging-face-hub }

| <div style="width:100px">変数</div> | 説明 | デフォルト | 適用先 |
| --------------------------------------- | -------------------------------------------------------- | ------- | --------------- |
| `HF_ENDPOINT`                           | Hugging Face Hub のエンドポイント。例：`https://hf-mirror.com` | （空） | サーバーとワーカー |
| `HF_TOKEN`                              | Hugging Face Hub のアクセストークン。 | （空） | サーバーとワーカー |

### デバッグ { #debugging }

| 変数 | 説明 | 既定値 | 適用対象 |
| --- | --- | --- | --- |
| `GPUSTACK_DEBUG_ENABLE_TRACEMALLOC` | 起動時に Python のメモリ割り当て追跡を有効にします。`--debug` とは独立しています。追跡には CPU とメモリの追加負荷がかかります。変更後は GPUStack の再起動が必要です。 | `false` | Server と Worker |

### データベース設定 { #database-configuration }

| 変数 | 説明 | デフォルト | 適用先 |
| -------------------------- | -------------------------------------------- | ------- | ---------- |
| `GPUSTACK_DB_ECHO`         | データベースのクエリーログを有効化。 | `false` | サーバー |
| `GPUSTACK_DB_POOL_SIZE`    | データベース接続プールのサイズ。 | `30`    | サーバー |
| `GPUSTACK_DB_MAX_OVERFLOW` | データベース接続プールの最大オーバーフロー数。 | `20`    | サーバー |
| `GPUSTACK_DB_POOL_TIMEOUT` | データベース接続プールのタイムアウト（秒）。 | `30`    | サーバー |
| `GPUSTACK_DB_POOL_RECYCLE` | データベース接続の再利用時間（秒）。 | `1800`  | サーバー |

### ネットワーク設定 { #network-configuration }

| 変数 | 説明 | デフォルト | 適用先 |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- | --------------- |
| `GPUSTACK_PROXY_TIMEOUT_SECONDS`               | プロキシする推論リクエストの絶対的な時間上限（秒）。 | `1800`  | サーバーとワーカー |
| `GPUSTACK_PROXY_TTFT_TIMEOUT_SECONDS`          | ストリーミング応答の最初のチャンクを待つ時間の上限（秒）。キュー待ちとプリフィルを含むため、余裕のある値にします。 | `300`   | サーバーとワーカー |
| `GPUSTACK_PROXY_STREAM_IDLE_TIMEOUT_SECONDS`   | ストリーミング応答開始後のチャンク間の最大間隔（秒）。本文が最後にまとめて届く非ストリーミング応答には適用しません。 | `60`    | サーバーとワーカー |
| `GPUSTACK_PROXY_UPSTREAM_IDLE_TIMEOUT_SECONDS` | Higress のアップストリーム接続プールのアイドルタイムアウト（秒）。アイドル接続を回収する設定で、リクエストの継続時間を制限するものではありません。 | `3`     | サーバー |
| `GPUSTACK_TCP_CONNECTOR_LIMIT`                 | HTTP クライアントの TCP コネクター上限。 | `1000`  | サーバーとワーカー |

### サーバーキャッシュ設定 { #server-cache-configuration }

| 変数 | 説明 | デフォルト | 適用先 |
| ----------------------------------- | ---------------------------- | ------- | ---------- |
| `GPUSTACK_SERVER_CACHE_TTL_SECONDS` | サーバーキャッシュの TTL（秒）。 | `600`   | サーバー |

### 認証とセキュリティ { #authentication-security }

| 変数 | 説明 | デフォルト | 適用先 |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------------- |
| `GPUSTACK_JWT_TOKEN_EXPIRE_MINUTES` | JWT トークンの有効期限（分）。 | `120`   | サーバー |
| `GPUSTACK_PROVIDER_TEST_EGRESS_ALLOWLIST` | 呼び出し元が指定するベース URL に接続するプロバイダーエンドポイント（`/get-models`、`/test-model`、`/test-decision-model`）の接続先を許可する CIDR、IP アドレス、DNS 名の一覧。ホスト名が指定名と一致またはドット境界で終わる場合、あるいは名前解決された全アドレスが指定 CIDR 内にある場合にのみ許可します。それ以外（http(s) 以外のベース URL を含む）は拒否します。空の場合は制限なし。例：`api.openai.com,10.0.1.0/24,vllm.internal.corp` | （空） | サーバー |

### ゲートウェイ設定 { #gateway-configuration }

| 変数 | 説明 | デフォルト | 適用先 |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------- |
| `GPUSTACK_GATEWAY_PORT_CHECK_INTERVAL`             | GPUStack サーバーが組み込みゲートウェイの待受ポートを確認する間隔（秒）。 | `2`       | サーバー |
| `GPUSTACK_GATEWAY_PORT_CHECK_RETRY_COUNT`          | GPUStack サーバーが組み込みゲートウェイの待受ポートを確認する再試行回数。 | `300`     | サーバー |
| `GPUSTACK_GATEWAY_AUTH_RECONCILE_INTERVAL_SECONDS` | ゲートウェイがローカル認証する API キーを、サーバーがデータベースから再計算する間隔。ORM を介さない削除はイベントを発行しないため、公開ルートでは削除後もキーが有効であり続ける最悪時の時間を示します。 | `30`      | サーバー |
| `GPUSTACK_GATEWAY_AUTH_ALLOW_CUSTOM_KEYS`          | カスタム API キー（ユーザーがシークレットを指定したキー）をゲートウェイで認証できるか。無効でもキーは使用できますが、リクエストごとにサーバーへ問い合わせます。有効にすると、シークレット自体のソルトなし高速ハッシュを索引としてゲートウェイ設定に公開します。ハッシュはデプロイ間で同一なので、弱いシークレットは事前計算テーブルで破られます。`custom` にエントロピー要件はないため、ユーザーが独自のキーを選ぶ環境では無効にしてください。調整のたびに再読込され、新しいキーだけでなく、有効時に公開したカスタムキーも撤回されます。 | `true`    | サーバー |
| `GPUSTACK_GATEWAY_AUTH_MAX_CR_BYTES`               | サーバーがゲートウェイ認証プラグインに書き込むキーテーブルと公開ルートルールのバイト上限。etcd のオブジェクト上限約 1.5 MiB より小さく設定します。上限を超えたキーは、リクエストごとにサーバー経由で認証します。 | `1100000` | サーバー |
| `GPUSTACK_GATEWAY_TLS_MIN_PROTOCOL_VERSION`        | ゲートウェイの HTTPS リスナーが受け付ける最小 TLS バージョン。`TLSv1.0`、`TLSv1.1`、`TLSv1.2`、`TLSv1.3` のいずれかで、ほかは起動時に拒否します。空の場合、TLS 1.0 と 1.1 を受け付ける Higress のデフォルトを維持します。`incluster` モードでは無視します。下記を参照。 | （空） | サーバー |
| `GPUSTACK_GATEWAY_TLS_MAX_PROTOCOL_VERSION`        | リスナーが受け付ける最大 TLS バージョン。値と適用範囲は同じで、通常は不要です。 | （空） | サーバー |

TLS バージョンの設定は `GPUSTACK_GATEWAY_MIRROR_INGRESS_NAME` で
指定した Ingress に書き込まれ、GPUStack が生成する
各 LLM ルートの Ingress は、その上下限をコピーします。
`incluster` モードではその Ingress は Helm チャートに属するため、
代わりに `server.ingress.tls.minProtocolVersion` / `maxProtocolVersion` を設定します。
環境変数は無視され、指定すると警告を記録します。変更は対象の Ingress に即時適用され、
既存の LLM ルートには次回の調整時に反映されます。
即時反映するにはサーバーを再起動します。

ゲートウェイプラグインの設定**内部**に反映する項目は、
ここではなく設定ファイルの `gateway_plugin` に指定します。
これにより、設定ファイルの値は次回の調整時、
ここの環境変数は再起動時という適用時点が明確になります。以下の 2 項目は移動済みで非推奨です。
設定ファイルで上書き可能なデフォルト値としては引き続き有効なため、
更新時に既存のデプロイ設定が意図せず元に戻ることはありません。
どちらかを設定すると警告を記録します。

| 非推奨の変数 | 説明 | 置き換え先 | デフォルト | 適用先 |
| ----------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------ | ---------- |
| `GPUSTACK_HIGRESS_EXT_AUTH_TIMEOUT_MS`                | ゲートウェイの認証呼び出しタイムアウト（ミリ秒）。 | `gateway_plugin.gpustack-ext-auth.config.authz.timeout`    | `30000`                              | サーバー |
| `GPUSTACK_GATEWAY_AI_STATISTICS_PLUGIN_CONTENT_TYPES` | トークン使用量を計測するコンテンツタイプのコンマ区切り一覧。 | `gateway_plugin.ai-statistics.config.enable_content_types` | `application/json,text/event-stream` | サーバー |

### 使用量追跡の設定 { #usage-tracking-configuration }

| 変数 | 説明 | デフォルト | 適用先 |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ---------- |
| `GPUSTACK_TIMEZONE`                                | すべての使用量ビューのバケット分けと Last Active・イベント時刻の表示に使用する、プラットフォーム共通の IANA タイムゾーン（例：`Asia/Shanghai`。下記注記を参照）。空の場合は OS のローカルタイムゾーン（`TZ` / `/etc/localtime`）。非推奨の `GPUSTACK_USAGE_ROLLUP_TIMEZONE` を置き換えます（別名としては引き続き有効）。 | （空） | サーバー |
| `GPUSTACK_USAGE_ESTIMATED_BYTES_PER_INPUT_TOKEN`   | ゲートウェイレポートが `completed=false` かつ `input_token` が空の場合、プロンプトトークン数の推定に使う 1 トークンあたりのバイト数の除数。デフォルトは英語中心の GPT 系トークナイザー向けです。CJK 中心の処理では小さく（例：`2`）します。 | `4`          | サーバー |
| `GPUSTACK_USAGE_ESTIMATED_TOKENS_PER_OUTPUT_CHUNK` | 不完全なストリームで `output_token` が空の場合、完了トークン数の推定に使うチャンクあたりのトークン数の乗数。 | `1`          | サーバー |
| `GPUSTACK_USAGE_DETAILS_RETENTION_MONTHS`          | `model_usage_details`（リクエスト単位の監査テーブル）の保持期間。`COALESCE(completed_at, created_at)` を基準として、この期間より古い行をリーダー専用のアーカイバーが `model_usage_details_archive` に移動します。 | `13`         | サーバー |
| `GPUSTACK_USAGE_DETAILS_ARCHIVE_CRON`              | 使用量明細の定期アーカイブの Cron 式（UTC）。このスケジュールに関係なく、サーバー起動時にも 1 回実行します。 | `0 3 * * *`  | サーバー |
| `GPUSTACK_USAGE_DETAILS_ARCHIVE_BATCH_SIZE`        | アーカイブ移動のバッチあたりの行数。小さくすると複製遅延に敏感な環境でトランザクションを短く保ち、大きくすると往復のオーバーヘッドを減らします。 | `1000`       | サーバー |
| `GPUSTACK_USAGE_DETAILS_BUFFER_MAX_SIZE`           | フラッシュ間にメモリーに保持するリクエスト明細バッファーの上限。DB 停止やスキーマ差異によってフラッシュが失敗し続けても、メモリー増加を制限します。超過時は最も古い項目を破棄し、`WARNING` ログを記録します。 | `100000`     | サーバー |
| `GPUSTACK_RESOURCE_USAGE_TICK_SECONDS`             | インスタンス使用量コレクターの定期処理間隔（秒）。前回以降にライフサイクルイベントがなかった長時間実行中の GPU/CPU インスタンスについて、累積稼働時間をフラッシュします。 | `300`        | サーバー |
| `GPUSTACK_STORAGE_USAGE_TICK_SECONDS`              | 永続ボリューム（ストレージ）の使用量コレクターの定期処理間隔（秒）。 | `300`        | サーバー |
| `GPUSTACK_METERED_USAGE_SEAL_GRACE_SECONDS`        | 時間バケット終了から確定までの猶予（秒）。`now >= bucket_end + grace` で確定し、遅延イベントや時刻のずれを吸収します。定期処理間隔より大きくしてください。確定済みバケットは変更不可です。 | `900`        | サーバー |
| `GPUSTACK_METERED_USAGE_RETENTION_MONTHS`          | `metered_usage`（時間単位のリソース使用量集計）の保持期間。古い確定済み行をリーダー専用アーカイバーが `metered_usage_archive` に移動します。コレクターの精算対象期間（時間）より十分長くしてください。 | `13`         | サーバー |
| `GPUSTACK_METERED_USAGE_ARCHIVE_CRON`              | 計測使用量の定期アーカイブの Cron 式（UTC）。このスケジュールに関係なく、サーバー起動時にも 1 回実行します。 | `0 4 * * *`  | サーバー |
| `GPUSTACK_METERED_USAGE_ARCHIVE_BATCH_SIZE`        | 計測使用量のアーカイブ移動のバッチあたりの行数。小さくするとトランザクションを短く保ち、大きくすると往復のオーバーヘッドを減らします。 | `5000`       | サーバー |
| `GPUSTACK_USAGE_EVENTS_RETENTION_MONTHS`           | `resource_events`（リソースのライフサイクル・監査ログ）の保持期間。古い行をリーダー専用アーカイバーが `resource_events_archive` に移動します。 | `13`         | サーバー |
| `GPUSTACK_USAGE_EVENTS_ARCHIVE_CRON`               | リソースイベントの定期アーカイブの Cron 式（UTC）。このスケジュールに関係なく、サーバー起動時にも 1 回実行します。 | `30 3 * * *` | サーバー |
| `GPUSTACK_USAGE_EVENTS_ARCHIVE_BATCH_SIZE`         | リソースイベントのアーカイブ移動のバッチあたりの行数。 | `5000`       | サーバー |
| `GPUSTACK_USAGE_BREAKDOWN_MAX_NO_PAGINATION_ROWS`  | ページ分割しない（`page=-1`）内訳が返せる最大バケット数。傾向グラフとエクスポートは全系列を一度に取得します。超過リクエストは黙って切り詰めず、HTTP 400 で拒否します。大規模なダッシュボードでは増やし、メモリーを制限するには減らします。 | `50000`      | サーバー |

!!! note "`GPUSTACK_TIMEZONE` の範囲と夏時間"

    この 1 つのタイムゾーンが、毎日の `model_usages` トークン集計、
    GPU・ストレージの時間バケット、表示する Last Active と
    リソースイベントの時刻など、すべての使用量ビューを制御します。**夏時間（DST）**を使う地域には、
    次の 2 つの影響があります。

    - DST の切り替え付近では、イベントの表示時刻が
      集計対象のバケットと異なる日になる場合があります。
    - DST 切り替え後に同じ過去のクエリーを再実行すると、
      古い行が 1 時間ずれたバケットに再分類され、結果が変化する場合があります。

    DST がない地域（UTC、`Asia/Shanghai` など）では正確で、影響を受けません。

### クラスター設定 { #cluster-configuration }

| 変数 | 説明 | デフォルト | 適用先 |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_DEFAULT_CLUSTER_KUBERNETES` | デフォルトクラスターを自動作成する場合、この変数が設定されていれば `Kubernetes` プロバイダーを使用します。 | `false` | サーバー |

### コンテンツソース設定 { #content-source-configuration }

| 変数 | 説明 | デフォルト | 適用先 |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_BOOTSTRAP_WITH_EMBEDDED_SOURCES` | すべての種類のコンテンツ（モデルカタログ、コミュニティバックエンド、組み込みバックエンドバージョン、キャッシュプロバイダー）を、公開ドキュメントではなく組み込みコンテンツで開始し、OTA サーバーから取得しないようにします。開始時点だけを決める設定で、後から有効にしたソースは維持されます。OTA サーバーに接続できない場合や、インストールしたバージョンにコンテンツを固定する場合に便利です。 | `false`（開発ビルドは `true`。空の値は未設定として扱う） | サーバー |

### スケジューラー設定 { #scheduler-configuration }

| 変数 | 説明 | デフォルト | 適用先 |
| --------------------------------------------------- | --------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_SCHEDULER_SCALE_UP_PLACEMENT_MAX_SCORE`   | スケジューラーの配置スコアラーが使用する最大配置スコア。 | `100`   | サーバー |
| `GPUSTACK_SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE`    | モデルファイルが存在する場合、配置スコアに加える最大ローカリティスコア。 | `5`     | サーバー |
| `GPUSTACK_SCHEDULER_SCALE_DOWN_STATUS_MAX_SCORE`    | スケールダウン時の状態スコアラーの最大寄与（正規化済み）。 | `100`   | サーバー |
| `GPUSTACK_SCHEDULER_SCALE_DOWN_OFFLOAD_MAX_SCORE`   | スケールダウン時のオフロードスコアラーの最大寄与（正規化済み）。 | `10`    | サーバー |
| `GPUSTACK_SCHEDULER_SCALE_DOWN_PLACEMENT_MAX_SCORE` | スケールダウン時の配置スコアラーの最大寄与（正規化済み）。 | `1`     | サーバー |
| `GPUSTACK_SCALING_SCHEDULER_INTERVAL`               | スケジュールされたスケーリングが各モデルのレプリカ数をウィンドウから再計算する間隔（秒）。調整はレベル駆動で、正確性ではなく、境界の検出までの遅延だけを制限します。最小 `1` 秒に制限されます。 | `30`    | サーバー |

#### 複数ロールのグループスケジューリング { #multi-role-group-scheduling }

これらの重みと時間は、プリフィル・デコード分離グループやそのルーターのように、ロールを宣言するデプロイメントに適用します。通常の単一ロールのデプロイメントには影響しません。以下の各スコアはスコアチェーンへの最大寄与で、`0` にすると該当条件を完全に考慮しなくなります。

| 変数 | 説明 | デフォルト | 適用先 |
| ---------------------------------------------------- | --------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_SCHEDULER_TOPOLOGY_PROXIMITY_MAX_SCORE`    | ロールのスケールアウトやルーター配置で、候補ワーカーが配置済みメンバーと共有するトポロジー階層あたりの最大スコア。配置スコアラーの 100 点の幅より大きくし、余剰容量だけで宣言されたトポロジーの 1 階層全体を覆さないようにします。`0` はリソース適合性だけで後続メンバーを配置します。 | `150`   | サーバー |
| `GPUSTACK_SCHEDULER_PAIRING_AFFINITY_MAX_SCORE`      | 分離グループの片方のロールをスケールアウトする際、候補ワーカー上の反対ロールのメンバーあたりの最大スコア。KV 転送を同一ホスト内に保つための値です。実際の重みではなく**下限**で、スケジューラーはこの値と「チェーン内のほかのスコアラーの合計上限 +1」の大きい方を使います。どのスコアラーが有効でも「反対ロールのメンバー数が多い方を優先し、同数なら容量で判定」を保証します。`0` は無効指定で、下限へ引き上げません。 | `200`   | サーバー |
| `GPUSTACK_SCHEDULER_SCALE_DOWN_PAIRING_MAX_SCORE`    | グループの片方のロールをスケールインする際、同じワーカーにいる反対ロールのメンバーが保持スコアに与える最大寄与。余剰メンバーがすべて正常で状態・オフロードスコアが同じ場合、削除対象を決めます。**50 未満**にしてください。状態スコアラーの「starting」と「running」の差が 50 のため、それ以上では仲間の多い起動中メンバーが単独の正常メンバーを上回り、異常メンバーを先に削除できなくなります。`0` は状態と配置だけで削除対象を決めます。 | `20`    | サーバー |
| `GPUSTACK_SCHEDULER_DRAIN_WINDOW_SECONDS`            | 縮小対象のグループメンバーがルーターの登録から外れた後も実行を続ける時間。プロセスの終了猶予ではなく、そのメンバーから KV キャッシュを取得中のデコードが完了するための時間で、エンジン自身では待機できません。`0` は即時削除し、ロールのないデプロイメントでは設定によらず即時削除です。 | `60`    | サーバー |
| `GPUSTACK_SCHEDULER_GATHER_BLOCKED_DWELL_SECONDS`    | `MustGather` グループのメンバーが未配置のまま待機し、デプロイメントが `gather_blocked_scale_out` を報告するまでの時間。時間が過ぎても何も失効しません。「スケジューラー未実行」と「実行したが gather ドメイン内に配置場所がない」を区別するだけです。すべてのスケールアップは最初の状態を通るため、待機なしではすべてにマーカーが付きます。`0` は未配置メンバーがあると直ちに拒否を報告します。 | `120`   | サーバー |

### GPU インスタンス設定 { #gpu-instance-configuration }

| 変数 | 説明 | デフォルト | 適用先 |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_GPU_INSTANCE_TRANSITIONING_REQUEUE_INTERVAL`      | まだ状態遷移中（未確定）の GPU インスタンスを、コントローラーがメモリー内の再キューで再確認する間隔（秒）。PV / PVT の終了コントローラーも、終了処理中のインスタンスの再確認に使用します。Ready 行のずれは下流の watch が検出します。最小 `1` 秒に制限されます。 | `15`    | サーバー |
| `GPUSTACK_GPU_INSTANCE_READY_SWEEP_INTERVAL`                | 確定済みの Ready GPU インスタンスを再確認する、任意の低頻度スイープの間隔（秒）。再接続の空白期間に下流 watch が見逃すワーカー側のずれへの補完です。デフォルトの `0` は無効です。そのような検出漏れが観測された場合のみ低頻度で設定します。 | `0`     | サーバー |
| `GPUSTACK_GPU_INSTANCE_PV_FINALIZING_UNREACHABLE_TOLERANCE` | 永続ボリューム (PV / PVT) のファイナライザーが、すべての Worker が停止している Kubernetes クラスタを待機する秒数です。この期間を超えると、そのクラスタを諦めて該当の行を解放します。Worker がまったくないクラスタは直ちにスキップされます（そのボリュームを保持したことはありません）。到達できないままのクラスタは待機された後解放され、そのボリュームはクラスタ内に残る場合があります。`0` は直ちに諦めることを意味します。`GPUSTACK_GPU_INSTANCE_UNREADABLE_CR_TOLERANCE` の `0` とは逆の意味（上限なし）です。 | `1800` | サーバー |

### ワーカーとモデルの設定 { #worker-and-model-configuration }

| 変数 | 説明 | デフォルト | 適用先 |
|------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------|----------------|
| `GPUSTACK_WORKER_HEARTBEAT_INTERVAL`                             | ワーカーのハートビート間隔（秒）。 | `30`    | ワーカー |
| `GPUSTACK_WORKER_STATUS_SYNC_INTERVAL`                           | ワーカーの状態同期間隔（秒）。 | `30`    | ワーカー |
| `GPUSTACK_WORKER_UNREACHABLE_CHECK_MODE`                         | ワーカー到達不能チェックのモード。選択肢：`auto`、`enabled`、`disabled`。`auto` はワーカー数が 50 を超えるとチェックを無効にします。 | `auto`  | サーバー |
| `GPUSTACK_WORKER_HEARTBEAT_GRACE_PERIOD`                         | ワーカーハートビートの猶予時間（秒）。 | `150`   | サーバー |
| `GPUSTACK_MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD`                | モデルインスタンスの再スケジュールの猶予時間（秒）。 | `300`   | サーバー |
| `GPUSTACK_MODEL_EVALUATION_CACHE_MAX_SIZE`                       | モデル評価キャッシュの最大サイズ。 | `1000`  | サーバー |
| `GPUSTACK_MODEL_EVALUATION_CACHE_TTL`                            | モデル評価キャッシュの TTL（秒）。 | `3600`  | サーバー |
| `GPUSTACK_WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD`           | ワーカーの孤立ワークロードを削除するまでの猶予時間（秒）。 | `300`   | ワーカー |
| `GPUSTACK_WORKER_ORPHAN_BENCHMARK_WORKLOAD_CLEANUP_GRACE_PERIOD` | ワーカーの孤立ベンチマークワークロードを削除するまでの猶予時間（秒）。 | `300`   | ワーカー |
| `GPUSTACK_WORKER_STATUS_COLLECTION_LOG_SLOW_SECONDS`             | ワーカーの状態収集がこの秒数を超えた場合、遅い処理としてデバッグログを追加します。 | `180`   | ワーカー |
| `GPUSTACK_MODEL_INSTANCE_HEALTH_CHECK_INTERVAL`                  | モデルインスタンスのヘルスチェック間隔（秒）。 | `3`     | ワーカー |
| `GPUSTACK_DISABLE_OS_FILELOCK`                                   | OS のファイルロックを無効化。 | `false` | ワーカー |
| `GPUSTACK_ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY`               | CUDA のマイナーバージョンが低いデバイスで、より高いマイナーバージョンのイメージを実行できるようにします。ワーカー全体またはモデルごとに設定でき、モデル側が優先されます。 | `false` | ワーカーとモデル |
| `GPUSTACK_RESTART_IN_FLIGHT_LAPSE_SECONDS`                       | 最初の再起動が再構築中に、`POST /models/{id}/restart` が 2 回目の要求を HTTP 409 で拒否する期間。この保護がないと、2 回目の解体処理が最初に作った置換リソースを削除します。`running` に到達しないデプロイメントも再起動できるよう、自動失効します。イメージ取得とディスクからの重み読込を伴うコールドスタートより長く設定します。 | `900`   | サーバー |
| `GPUSTACK_SERVE_LOG_MAX_BYTES`                                   | 1 つのサービングログの 1 回の再起動あたりのバイト上限。超過後は先頭を保持し、末尾をローテーションし、間にマーカーを表示します。`0` は上限なし。 | `67108864` | ワーカー |
| `GPUSTACK_SERVE_LOG_HEAD_BYTES`                                  | 一度だけ書き込み、その後は書き換えず保持する先頭部分の容量。全体上限の半分以下に制限され、`0` は最初の 1 行だけ保持します。 | `8388608` | ワーカー |

### ベンチマーク設定 { #benchmark-configuration }

| 変数 | 説明 | デフォルト | 適用先 |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ---------- |
| `GPUSTACK_BENCHMARK_DATASET_SHAREGPT_PATH` | `dataset_name` が `ShareGPT` の場合、ベンチマークコンテナーが使用する ShareGPT データセットのパス。デフォルトのイメージに同梱されています。 | `/workspace/benchmark-runner/sharegpt_data/ShareGPT_V3_unfiltered_cleaned_split.json` | ワーカー |
| `GPUSTACK_BENCHMARK_REQUEST_TIMEOUT`       | 各ベンチマークリクエストのタイムアウト（秒）。 | `3600`                                                                                | ワーカー |

### モデルデプロイ設定 { #model-deployment-configuration }

!!! note

    これらの環境変数は GPUStack の起動時には**設定しません**。モデルのデプロイ時に **Advanced Options > Environment Variables** で設定し、モデルの提供動作をカスタマイズします。

| <div style="width:180px">変数</div>                   | 説明 | デフォルト | 適用先 |
|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|----------------|
| `GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED`          | サービングコマンドスクリプトの自動実行を無効化。`1` または `true` にすると、パッケージのインストールなどの準備を行うスクリプトを実行しません。 | `0`     | モデル |
| `PYPI_PACKAGES_INSTALL`                                   | モデルの提供環境に追加でインストールする PyPI パッケージ。複数は空白区切りで指定します。利用可能なら `uv pip install`、なければ `pip install` を使用します。 | （空） | モデル |
| `GPUSTACK_MODEL_RAM_CLAIM`                                | ユーザーが申告するモデルの RAM 要件（バイト）。スケジューラーの容量計画に使用します。 | （空） | モデル |
| `GPUSTACK_MODEL_VRAM_CLAIM`                               | ユーザーが申告するモデルの VRAM 要件（バイト）。スケジューラーの容量計画に使用します。 | （空） | モデル |
| `GPUSTACK_APPLY_QWEN3_RERANKER_TEMPLATES`                 | リクエスト本文に Qwen3 リランカーのテンプレートを適用します。https://huggingface.co/Qwen/Qwen3-Reranker-0.6B の手順を参照。 | （空） | モデル |
| `GPUSTACK_SKIP_MODEL_EVALUATION`                          | デプロイ時のモデル評価または検証を省略します。 | （空） | モデル |
| `GPUSTACK_SKIP_GPU_COUNT_CHECK`                           | 手動選択した GPU 数と並列処理の world size（tp×pp×dp）の一致チェックを省略します。GPU を手動選択した場合のみ有効で、自動スケジューリングには影響しません。並列パラメーターと選択 GPU 数を意図的に変える単一ノードの手動デプロイに使用します。 | （空） | モデル |
| `GPUSTACK_DISABLE_METRICS`                                | モデルのメトリクス公開と収集を無効化。 | （空） | モデル |
| `GPUSTACK_MODEL_HEALTH_CHECK_PATH`                        | モデルが公開する HTTP ヘルスチェックパスを指定。 | （空） | モデル |
| `GPUSTACK_MODEL_RUNTIME_UID`                              | コンテナー内で動くプロセスのユーザー権限を制御。 | （空） | モデル |
| `GPUSTACK_MODEL_RUNTIME_GID`                              | コンテナー内で動くプロセスのグループ権限を制御。 | （空） | モデル |
| `GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB`                     | コンテナーの共有メモリーサイズ（GiB）。 | `10.0`  | モデル |
| `GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED`           | このモデルの推論ヘルスチェックを有効化。有効時は、GPUStack が最小限の推論要求を定期的に送り、正常な応答を確認します。 | `false` | モデル |
| `GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_INTERVAL`          | 推論ヘルスチェック間隔（秒、最小 `60`）。この期間内に成功した推論トラフィックがあれば、能動的なチェックを省略します。 | `300`   | モデル |
| `GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_TIMEOUT`           | 各定期推論ヘルスチェック要求のタイムアウト（秒）。デフォルトは SGLang PD エンジンロールで 25、それ以外で 15 です。明示した値を優先します。 | `15/25` | モデル |
| `GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD` | インスタンスを異常と判定するまでの推論ヘルスチェックの連続失敗回数。 | `3`     | モデル |
| `GPUSTACK_ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY`        | CUDA のマイナーバージョンが低いデバイスで、より高いマイナーバージョンのイメージを実行できるようにします。ワーカー全体またはモデルごとに設定でき、モデル側が優先されます。 | `false` | ワーカーとモデル |

#### 使用例 { #usage-example }

モデルのデプロイ時に **Advanced Options > Environment Variables** に移動し、次を追加します。

```bash
# Install additional packages before model serving starts
PYPI_PACKAGES_INSTALL=torch-audio==2.0.0 transformers==4.30.0

# Disable the serving command script entirely
GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED=1
```

サービングコマンドスクリプトは、次を自動処理します。

- `PYPI_PACKAGES_INSTALL` に指定した追加の PyPI パッケージのインストール。
- パッケージインストールでの `uv pip` と `pip` の両方のサポート。
- `PIP_INDEX_URL` と `PIP_EXTRA_INDEX_URL` によるカスタム PyPI インデックスの処理。

## GPUStack Runtime の環境変数 { #gpustack-runtime-environment-variables }

これらの環境変数は GPUStack Runtime が使用し、主に Docker/Kubernetes 内の推論バックエンドの動作を調整します。

ワーカー内でのみ利用できます。適切に反映されるよう、ワーカーのコンテナーに環境変数を設定してください。

### グローバル変数 { #global-variables }

| 変数 | 説明 | デフォルト |
| -------------------------------- | -------------------------- | ------- |
| `GPUSTACK_RUNTIME_LOG_LEVEL`     | ログレベル。 | `INFO`  |
| `GPUSTACK_RUNTIME_LOG_WARNING`   | 警告のログ記録を有効化。 | `0`     |
| `GPUSTACK_RUNTIME_LOG_EXCEPTION` | 例外のログ記録を有効化。 | `0`     |

### 検出器の変数 { #detector-variables }

| 変数 | 説明 | デフォルト |
| -------------------------------------------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --------------------------------------- |
| `GPUSTACK_RUNTIME_DETECT`                          | 使用する検出器。選択肢：Auto、AMD、ASCEND、CAMBRICON、HYGON、ILUVATAR、METAX、MTHREADS、NVIDIA。 | `Auto`                                  |
| `GPUSTACK_RUNTIME_DETECT_NO_PCI_CHECK`             | 検出時の PCI チェックを省略します。WSL 環境で便利です。 | （空） |
| `GPUSTACK_RUNTIME_DETECT_PCI_CLASS_PREFIXES`       | 検出時にチェックする PCI クラスの接頭辞。`GPUSTACK_RUNTIME_DETECT_NO_PCI_CHECK` が false の場合、デバイスの PCI クラスが指定したいずれかの接頭辞で始まるかを確認します。 | 02,03,0b,12                             |
| `GPUSTACK_RUNTIME_DETECT_BACKEND_MAP_RESOURCE_KEY` | 検出バックエンドからリソースキーへのマッピング。例：`{"cuda": "nvidia.com/devices", "rocm": "amd.com/devices"}`。gpustack-runner のバックエンド名を対応リソースキーに対応付けます。 | 各ベンダーが定義するデフォルト値 |
| `GPUSTACK_RUNTIME_DETECT_PHYSICAL_INDEX_PRIORITY`  | デバイス検出時に物理インデックスを優先。 | `1`                                     |

#### ROCm 検出器固有の変数 { #rocm-detector-specific-variables }

!!! note

    Hygon など ROCm ベースのバックエンドにも適用します。

| 変数 | 説明 | デフォルト |
| -------------------- | ----------------------- | ----------- |
| `ROCM_SMI_LIB_PATH`  | ROCm SMI ライブラリのパス。 | （空） |
| `ROCM_HOME`          | ROCm のホームディレクトリ。 | （空） |
| `ROCM_PATH`          | ROCm のパス。 | `/opt/rocm` |
| `ROCM_CORE_LIB_PATH` | ROCm コアライブラリのパス。 | （空） |

### デプロイヤーの変数 { #deployer-variables }

| 変数 | 説明 | デフォルト |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `GPUSTACK_RUNTIME_DEPLOY`                                          | 使用するデプロイヤー。選択肢：Auto、Docker、Kubernetes、Podman（実験的）。 | `Auto`                                          |
| `GPUSTACK_RUNTIME_DEPLOY_API_CALL_ERROR_DETAIL`                    | デプロイ時の API 呼び出しエラーの詳細表示を有効化。 | `1`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_PRINT_CONVERSION`                         | デプロイ時の変換結果の出力を有効化。GPUStack Runtime の統一 Workload 定義 API は、個別のコンテナーランタイム API 呼び出し（Docker SDK、Kubernetes API、Podman SDK など）へ変換されます。有効にすると最終的な呼び出しを INFO ログに出力し、デバッグに使用できます。 | `0`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_ASYNC`                                    | 非同期デプロイを有効化。 | `1`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_ASYNC_THREADS`                            | スレッドプールのスレッド数。 | （空） |
| `GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME`                            | デプロイヤーの名前。 | （空） |
| `GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_ENVIRONMENTS`  | ミラーデプロイ時に無視する環境変数名。 | （空） |
| `GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES`       | ミラーデプロイ時に無視するボリュームマウント先。 | （空） |
| `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME`      | デフォルトのコンテナーレジストリのユーザー名。GPUStack ランナーイメージが認証の必要なレジストリにある場合、すべての GPUStack ワーカー（またはオールインワンの GPUStack サーバー）の起動時に設定します。実際には、コンテナーランタイム自体に認証ユーザー名を設定することを推奨します。 | （空） |
| `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD`      | デフォルトのコンテナーレジストリのパスワード。GPUStack ランナーイメージが認証の必要なレジストリにある場合、すべての GPUStack ワーカー（またはオールインワンの GPUStack サーバー）の起動時に設定します。実際には、コンテナーランタイム自体に認証パスワードを設定することを推奨します。 | （空） |
| `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE`              | デフォルトのランナーイメージの名前空間。`gpustack` 以外（例：`gpustack-ai`）にイメージがある場合、この変数を `gpustack-ai` に設定します。`awesome.com/amazing-group/amazing-subgroup/gpustack-ai` のような多階層の場合は、`--system-default-container-registry`（ワーカー設定の `system_default_container_registry` キー、または `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY` 環境変数でも可）に親名前空間を含めて指定します。例：`--system-default-container-registry=awesome.com/amazing-group/amazing-subgroup`。 | `gpustack` |
| `GPUSTACK_RUNTIME_DEPLOY_IMAGE_PULL_POLICY`                        | デプロイヤーのイメージ取得ポリシー（Always、IfNotPresent、Never など）。 | `IfNotPresent`                                  |
| `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY`                      | デプロイ時に Container Device Interface（CDI）仕様を読み込むディレクトリ、または CDI 仕様を生成する先のディレクトリのパス。 | `/var/run/cdi`                                  |
| `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI`                     | コンテナーランタイムに注入するデバイスを伝える、CDI の手動マッピング。例：`nvidia.com/devices=nvidia.com/gpu;amd.com/devices=amd.com/gpu`。キーはリソースキー、値は CDI キーです。 | 各ベンダーが定義するデフォルト値 |
| `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES` | コンテナーランタイムに注入するデバイスを伝える、ランタイム可視デバイス環境変数の手動マッピング。例：`nvidia.com/devices=NVIDIA_VISIBLE_DEVICES;amd.com/devices=AMD_VISIBLE_DEVICES`。キーはリソースキー、値は環境変数名です。 | 各ベンダーが定義するデフォルト値 |
| `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_BACKEND_VISIBLE_DEVICES` | コンテナー内で使うデバイスをデバイスランタイム（ROCm、CUDA、OneAPI など）に伝える、バックエンド可視デバイス環境変数の手動マッピング。例：`nvidia.com/devices=CUDA_VISIBLE_DEVICES;amd.com/devices=ROCR_VISIBLE_DEVICES`。キーはリソースキー、値は環境変数名の一覧です。 | 各ベンダーが定義するデフォルト値 |
| `GPUSTACK_RUNTIME_DEPLOY_RUNTIME_VISIBLE_DEVICES_VALUE_UUID`       | 指定したランタイム可視デバイス環境変数に UUID を使用。 | （空） |
| `GPUSTACK_RUNTIME_DEPLOY_BACKEND_VISIBLE_DEVICES_VALUE_ALIGNMENT`  | 指定したバックエンド可視デバイス環境変数の値の整列を有効化。 | `ASCEND_RT_VISIBLE_DEVICES,NPU_VISIBLE_DEVICES` |
| `GPUSTACK_RUNTIME_DEPLOY_CPU_AFFINITY`                             | デプロイしたワークロードの CPU アフィニティを有効化。 | `0`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_NUMA_AFFINITY`                            | デプロイしたワークロードの NUMA アフィニティを有効化。`GPUSTACK_RUNTIME_DEPLOY_CPU_AFFINITY` も暗黙に有効になります。 | `0`                                             |

#### Docker デプロイヤー固有の変数 { #docker-deployer-specific-variables }

| 変数 | 説明 | デフォルト |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `GPUSTACK_RUNTIME_DOCKER_HOST`                      | Docker 接続先ホスト。デフォルトの Docker ホストを上書きします。 | `http+unix:///var/run/docker.sock` |
| `GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE`               | Docker の pause コンテナーに使用するイメージ。 | `gpustack/runtime:pause`           |
| `GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE`   | Docker の異常時再起動コンテナーに使用するイメージ。 | `gpustack/runtime:health`          |
| `GPUSTACK_RUNTIME_DOCKER_RESOURCE_INJECTION_POLICY` | Docker デプロイヤーのリソース注入ポリシー（Env、CDI など）。<br/>`Env`：`GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES` に基づき、標準の環境変数で注入します。<br/>`CDI`：`GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI` に基づき、CDI で注入します。 | `Env`                              |
| `GPUSTACK_RUNTIME_DOCKER_CDI_SPECS_GENERATE`        | CDI リソース注入ポリシーでデプロイする際、CDI 仕様を生成します。`GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY` が存在する必要があります。`GPUSTACK_RUNTIME_DOCKER_RESOURCE_INJECTION_POLICY` が `CDI` の場合のみ有効です。内部知識に基づいてデプロイヤー用の仕様を生成するため、NVIDIA Container Toolkit などほかのツールの生成ファイルと競合する場合は、無効にして出力ファイルを手動削除してください。 | `1`                                |

#### Kubernetes デプロイヤー固有の変数 { #kubernetes-deployer-specific-variables }

| 変数 | 説明 | デフォルト |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `GPUSTACK_RUNTIME_KUBERNETES_NODE_NAME`                      | ワークロードを配置する Kubernetes ノード名。 | （空） |
| `GPUSTACK_RUNTIME_KUBERNETES_NAMESPACE`                      | ワークロードを配置する Kubernetes 名前空間。 | `default`       |
| `GPUSTACK_RUNTIME_KUBERNETES_DOMAIN_SUFFIX`                  | Kubernetes サービスのドメイン接尾辞。 | `cluster.local` |
| `GPUSTACK_RUNTIME_KUBERNETES_SERVICE_TYPE`                   | Kubernetes サービスの種類。選択肢：ClusterIP、NodePort、LoadBalancer。 | `ClusterIP`     |
| `GPUSTACK_RUNTIME_KUBERNETES_QUORUM_READ`                    | Kubernetes サービスに quorum read を使用するか。 | `0`             |
| `GPUSTACK_RUNTIME_KUBERNETES_DELETE_PROPAGATION_POLICY`      | Kubernetes リソースの削除伝播ポリシー。選択肢：Foreground、Background、Orphan。 | `Foreground`    |
| `GPUSTACK_RUNTIME_KUBERNETES_RESOURCE_INJECTION_POLICY`      | Kubernetes デプロイヤーのリソース注入ポリシー。選択肢：Auto、Env、KDP。<br/>`Auto`：環境に応じて自動選択。<br/>`Env`：`GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES` に基づき標準の環境変数で注入。基盤の Container Toolkit に依存します。<br/>`KDP`：Kubernetes Device Plugin で注入。 | `Auto`          |
| `GPUSTACK_RUNTIME_KUBERNETES_KDP_PER_DEVICE_MAX_ALLOCATIONS` | Kubernetes Device Plugin の 1 デバイスあたりの最大割り当て数。 | `10`            |
| `GPUSTACK_RUNTIME_KUBERNETES_KDP_DEVICE_ALLOCATION_POLICY`   | Kubernetes Device Plugin のデバイス割り当てポリシー。選択肢：Auto、CDI、Env、Opaque。<br/>`Auto`：環境に応じて自動選択。<br/>`Env`：`GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES` に基づきランタイム可視環境変数で割り当て。Container Toolkit の対応が必要です。<br/>`CDI`：`GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI` に基づいて生成した CDI 仕様で割り当て。デバッグや診断が容易ですが、`GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY` が必要です。<br/>`Opaque`：内部ロジックで割り当て。デプロイは容易ですが、診断は難しくなります。 | `Auto`          |
| `GPUSTACK_RUNTIME_KUBERNETES_KDP_CDI_SPECS_GENERATE`         | デプロイ時に CDI 仕様を生成。`GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY` が必要です。`GPUSTACK_RUNTIME_KUBERNETES_KDP_DEVICE_ALLOCATION_POLICY` が `CDI` の場合のみ有効です。内部知識に基づいて生成するため、NVIDIA Container Toolkit などほかのツールの生成ファイルと競合する場合は、無効にして出力ファイルを手動削除してください。 | `1`             |

#### Podman デプロイヤー固有の変数 { #podman-deployer-specific-variables }

!!! note

    Podman デプロイヤーは実験的機能で、バージョン 4.9 以降が必要です。

| 変数 | 説明 | デフォルト |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `GPUSTACK_RUNTIME_PODMAN_HOST`                    | Podman 接続先ホスト。デフォルトの Podman ホストを上書きします。 | `http+unix:///run/podman/podman.sock` |
| `GPUSTACK_RUNTIME_PODMAN_PAUSE_IMAGE`             | Podman の pause コンテナーに使用するイメージ。デフォルトは `GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE` と同じです。 | `gpustack/runtime:pause`              |
| `GPUSTACK_RUNTIME_PODMAN_UNHEALTHY_RESTART_IMAGE` | Podman の異常時再起動コンテナーに使用するイメージ。デフォルトは `GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE` と同じです。 | `gpustack/runtime:health`             |
| `GPUSTACK_RUNTIME_PODMAN_CDI_SPECS_GENERATE`      | デプロイ時に CDI 仕様を生成。`GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY` が必要です。内部知識に基づいて生成するため、NVIDIA Container Toolkit などほかのツールの生成ファイルと競合する場合は、無効にして出力ファイルを手動削除してください。デフォルトは `GPUSTACK_RUNTIME_DOCKER_CDI_SPECS_GENERATE` と同じです。 | `1`                                   |
