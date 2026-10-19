---
translation_status: translated
hide:
  - toc
---

# gpustack start { #gpustack-start }

GPUStackのサーバーまたはワーカーを起動します。

```bash
gpustack start [OPTIONS]
```

!!! note "CLI引数の指定位置"

    Dockerでは、`docker run`が実際に実行するコマンドは**ENTRYPOINT**と**COMMAND**で構成されます。

    `gpustack/gpustack`イメージには`ENTRYPOINT`が設定されており、これは次のコマンドとして
    理解できます（または同等と見なせます）。

        gpustack start

    そのため、`gpustack start`のCLI引数はイメージ名の後、
    `docker run`コマンドの末尾に指定してください。`docker run`自体のオプションとしては指定しません。

    例：

        docker run [docker options] gpustack/gpustack <start-args>

    Kubernetesでは、コンテナー仕様の`args`フィールドを使って
    これらの引数を指定してください。

## 設定 { #configurations }

### 共通オプション { #common-options }

| <div style="width:180px">フラグ</div>         | <div style="width:100px">デフォルト</div> | 説明 |
|---------------------------------------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `--advertise-address` 値                 | （空）                                | 外部アクセス向けに公開するIPアドレス。<br/>未設定の場合、適切なローカルIPアドレスを自動検出します。 |
| `--config-file` 値                       | （空）                                | YAML設定ファイルのパス。 |
| `-d` 値, `--debug` 値                 | `False`                                | デバッグモードを有効にします。短いフラグ-dはPowerShellのCommonParameters用に予約されているため、Windowsでは使用できません。 |
| `--data-dir` 値                          | （空）                                | データを保存するディレクトリー。デフォルトはOSによって異なります。 |
| `--cache-dir` 値                         | （空）                                | キャッシュ（モデルファイルなど）を保存するディレクトリー。デフォルトは<data-dir>/cacheです。 |
| `--huggingface-token` 値                 | （空）                                | Hugging Face Hubへの認証に使用するユーザーアクセストークン。`HF_TOKEN`環境変数でも設定できます。 |
| `--insecure-tls`                            | `False`                                | GPUStack自身のHTTPS接続（ワーカーからサーバー、サーバーから外部認証IdP）のTLS検証を省略します。他の方法でサーバー証明書を検証できない、信頼済みネットワーク向けです。 |
| `--bin-dir` 値                           | （空）                                | 追加バイナリー（バージョン別のバックエンド実行ファイルなど）を保存するディレクトリー。 |
| `--pipx-path` 値                         | （空）                                | バージョン別のバックエンドをインストールするためのpipx実行ファイルのパス。 |
| `--system-default-container-registry` 値 | `docker.io`                            | GPUStackがシステム用および推論用イメージを取得するデフォルトのコンテナーレジストリー。<br/>GPUStackの業務用イメージはすべて`gpustack`名前空間に配置してください。多階層の名前空間が必要な場合も、`awesome.com/amazing-group/amazing-subgroup/gpustack`のように最終階層を`gpustack`にします。このフラグ（またはサーバー/ワーカーの設定キー`system_default_container_registry`、環境変数`GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY`）には、`awesome.com/amazing-group/amazing-subgroup`のように親のパスを指定してください。<br/>イメージを`gpustack`名前空間に配置できない場合（例：`gpustack-ai`）、`--image-repo`と`--benchmark-image-repo`を適宜変更してください。 |
| `--image-repo` 値                        | `gpustack/gpustack`                    | GPUStackコンテナーのデフォルトのイメージリポジトリーを上書きします。値にタグを含めないでください。GPUStackがバージョンタグを自動で付加します。<br/>イメージ`gpustack/gpustack:<tag>`が`gpustack`名前空間の外（例：`awesome.com/amazing-group/amazing-subgroup/gpustack-ai/gpustack:<tag>`）にある場合、まず`--system-default-container-registry=awesome.com/amazing-group/amazing-subgroup`で親の名前空間を含むレジストリーを指定し、次にこのフラグ（またはサーバー/ワーカーの設定キー`image_repo`、環境変数`GPUSTACK_IMAGE_REPO`）を`gpustack-ai/gpustack`に設定してください。 |
| `--gateway-mode` 値                      | `auto`                                 | ゲートウェイの動作モード。選択肢はembedded、in-cluster、external、disabled、auto（デフォルト）です。 |
| `--gateway-kubeconfig` 値                | （空）                                | ゲートウェイ用kubeconfigファイルのパス。gateway-modeがexternalの場合のみ有効です。 |
| `--gateway-namespace` 値                 | `higress-system`                       | ゲートウェイコンポーネントをデプロイする名前空間。 |
| `--service-discovery-name` 値            | （空）                                | DNS上のサービスディスカバリーサービス名。サービスディスカバリーを使用してKubernetesにデプロイする場合のみ有効です。 |
| `--namespace` 値                         | （空）                                | GPUStackがゲートウェイのルーティングルールとモデルインスタンスをデプロイするKubernetes名前空間。 |

### サーバーオプション { #server-options }

| <div style="width:180px">フラグ</div>              | <div style="width:100px">デフォルト</div>           | 説明 |
|--------------------------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--port` 値                                   | `80`                                             | サーバーがバインドするポート。 |
| `--tls-port` 値                               | `443`                                            | TLSサーバーがバインドするポート。 |
| `--api-port` 値                               | `30080`                                          | GPUStack APIサーバーがバインドするポート。 |
| `--proxy-port` 値                             | `30079`                                          | GPUStackプロキシサーバーがバインドするポート。 |
| `--database-port` 値                          | `5432`                                           | 組み込みPostgresSQLデータベースのポート。 |
| `--metrics-port` 値                           | `10161`                                          | サーバーメトリクスを公開するポート。 |
| `--disable-metrics`                              | `False`                                          | サーバーメトリクスを無効にします。 |
| `--disable-worker`                               | （空）                                          | （非推奨）GPUStackサーバーの組み込みワーカーを無効にします。新規インストールでは、デフォルトで組み込みワーカーは存在しません。必要な場合は'--enable-worker'で有効にしてください。どちらのフラグも設定されていない場合、後方互換性のため、v2.0.1より前の既存インストールでは組み込みワーカーがデフォルトで有効になります。 |
| `--enable-worker`                                | `False`                                          | GPUStackサーバーの組み込みワーカーを有効にします。 |
| `--bootstrap-password` 値                     | 自動生成。                                  | デフォルトの管理者ユーザーの初期パスワード。 |
| `--database-url` 値                           | 組み込みPostgreSQL。                             | データベースのURL。PostgreSQL 13.0+、MySQL 8.0.36+、およびopenGauss（postgresql://スキーム）やOceanBase（mysql://スキーム）などの互換データベースをサポートします。例：postgresql://username:password@host:port/dbnameまたはmysql://username:password@host:port/dbname |
| `--redis-url` 値                              | None                                             | 共有された高速な状態を必要とする機能（LBプラグインの共有状態バックエンド）で使用する外部RedisのURL。形式：`redis://host[:port][/db]`。認証情報はゲートウェイCRに保存されてしまうため受け付けません。TLS（`rediss://`）はサポートしていません。 |
| `--ssl-keyfile` 値                            | （空）                                          | SSLキーファイルのパス。 |
| `--ssl-certfile` 値                           | （空）                                          | SSL証明書ファイルのパス。 |
| `--ssl-ca-certfile` 値                        | （空）                                          | ワーカーがGPUStackサーバーを検証するために使用するPEM形式のCAバンドルのパス。サーバー証明書がプライベートCAで署名されている場合に必要です。 |
| `--force-auth-localhost`                         | （未設定）                                          | （非推奨、効果なし）localhostからの呼び出しも常に認証されます。後方互換性のために残されており、将来のリリースで削除されます。 |
| `--disable-update-check`                         | `False`                                          | 更新チェックを無効にします。 |
| `--ota-server-url` 値                         | （空）                                          | 公式コンテンツソースを読み込むOTAサーバーのURL。インデックスとファイルは、そのURLの直下に配置します（例：https://ota.example.com/gpustack）。デフォルトは公開OTAサーバーです。ネットワークから到達できない場合は自分で用意したコピーを指定でき、独自リポジトリーから内部環境に配信する場合にも使用できます。3種類のコンテンツすべてに適用され、`Official`の状態と自動更新は維持されます。UIの`Reset to Official Source`でもこのURLに戻ります。[OTAコンテンツの更新](../user-guide/ota-content-updates.md)を参照してください。 |
| `--disable-openapi-docs`                         | `False`                                          | 自動生成されるOpenAPIドキュメントのエンドポイント（/docsのSwagger UI、/redocのReDoc、/openapi.jsonの未加工の仕様）を無効にします。 |
| `--model-catalog-file` 値                     | （空）                                          | モデルカタログファイルのパスまたはURL。 |
| `--enable-cors`                                  | `False`                                          | サーバーでCross-Origin Resource Sharing（CORS）を有効にします。 |
| `--allow-credentials`                            | `False`                                          | クロスオリジンリクエストでCookieと認証情報を許可します。 |
| `--allow-origins` 値                          | `["*"]`                                          | クロスオリジンリクエストで許可するオリジン。複数のオリジンを指定する場合はフラグを繰り返します。例：`--allow-origins https://example.com --allow-origins https://api.example.com` |
| `--allow-methods` 値                          | `["GET", "POST"]`                                | クロスオリジンリクエストで許可するHTTPメソッド。複数のメソッドを指定する場合はフラグを繰り返します。例：`--allow-methods GET --allow-methods POST` |
| `--allow-headers` 値                          | `["Authorization", "Content-Type", "X-API-Key"]` | クロスオリジンリクエストで許可するHTTPリクエストヘッダー。複数のヘッダーを指定する場合はフラグを繰り返します。例：`--allow-headers Authorization --allow-headers X-API-Key --allow-headers Content-Type` |
| `--oidc-issuer` 値                            | （空）                                          | OpenID Connectの発行者URL。 |
| `--oidc-client-id` 値                         | （空）                                          | OpenID ConnectのクライアントID。 |
| `--oidc-client-secret` 値                     | （空）                                          | OpenID Connectのクライアントシークレット。 |
| `--oidc-redirect-uri` 値                      | （空）                                          | OIDCアプリケーションに設定したリダイレクトURI。`<server-url>/auth/oidc/callback`に設定する必要があります。 |
| `--external-auth-post-logout-redirect-key` 値 | （空）                                          | 各IdPのログアウト後リダイレクトに使用する共通のパラメーター名。OIDCとSAMLの両方に適用されます。 |
| `--oidc-skip-userinfo`                           | `False`                                          | OIDCのuserinfo_endpointへのリクエストを省略し、代わりにヘッダーから直接解析を試みます。 |
| `--oidc-use-userinfo`                            | （空）                                          | ［非推奨］認証後にUserInfoエンドポイントを使用してユーザー情報を取得します。 |
| `--saml-idp-server-url` 値                    | （空）                                          | SAML Identity ProviderのサーバーURL。 |
| `--saml-idp-entity-id` 値                     | （空）                                          | SAML Identity ProviderのエンティティーID。 |
| `--saml-idp-x509-cert` 値                     | （空）                                          | SAML Identity ProviderのX.509証明書。 |
| `--saml-sp-entity-id` 値                      | （空）                                          | SAML Service ProviderのエンティティーID。 |
| `--saml-sp-acs-url` 値                        | （空）                                          | SAML Service ProviderのAssertion Consumer Service URL。`<server-url>/auth/saml/callback`に設定する必要があります。 |
| `--saml-sp-x509-cert` 値                      | （空）                                          | SAML Service ProviderのX.509証明書。 |
| `--saml-sp-private-key` 値                    | （空）                                          | SAML Service Providerの秘密鍵。 |
| `--saml-idp-logout-url` 値                    | （空）                                          | SAML Identity ProviderのSingle LogoutエンドポイントURL。 |
| `--saml-sp-slo-url` 値                        | （空）                                          | SAML Service ProviderのSingle Logout ServiceコールバックURL（例：`<server-url>/auth/saml/logout/callback`）。 |
| `--saml-security` 値                          | （空）                                          | JSON形式のSAMLセキュリティー設定。 |
| `--cas-server-url` 値                         | （空）                                          | CAS（Central Authentication Service）サーバーのベースURL（例：`https://cas.example.com/cas`）。これを設定するとCASログインが有効になります。 |
| `--cas-callback-url` 値                       | （空）                                          | CASサーバーに登録したCASコールバックURL。未設定の場合は`<server-url>/auth/cas/callback`を使用します。 |
| `--cas-validate-endpoint` 値                  | `/p3/serviceValidate`                            | `--cas-server-url`を基準とするCASチケット検証エンドポイント。CAS 3.0より前のサーバーでは`/serviceValidate`を使用してください（属性はサポートされません）。 |
| `--cas-username-attribute` 値                 | （空）                                          | GPUStackのユーザー名に使用するCAS XML属性名。未設定の場合はCASの`cas:user`要素を使用します。 |
| `--cas-full-name-attribute` 値                | （空）                                          | ユーザーの氏名に使用するCAS XML属性名。 |
| `--cas-avatar-attribute` 値                   | （空）                                          | ユーザーのアバターURLに使用するCAS XML属性名。 |
| `--external-auth-provider-name` 値            | （空）                                          | ログインページのSSOボタンに表示するIDプロバイダー名（例：`Okta`）。未設定の場合はプロトコル名（OIDC/SAML/CAS）を使用します。 |
| `--external-auth-name` 値                     | （空）                                          | 外部認証のユーザー情報をユーザー名に対応付けるマッピング（例：preferred_username）。 |
| `--external-auth-full-name` 値                | （空）                                          | 外部認証のユーザー情報をユーザーの氏名に対応付けるマッピング。`name`や`firstName+lastName`のように複数の要素を組み合わせることができます。 |
| `--external-auth-avatar-url` 値               | （空）                                          | 外部認証のユーザー情報をユーザーのアバターURLに対応付けるマッピング。 |
| `--external-auth-default-inactive`               | `False`                                          | 新規ユーザーをデフォルトで無効にする場合はTrue。 |
| `--external-auth-insecure-skip-tls-verify`       | `False`                                          | 外部認証IdPとのハンドシェイク（OIDC/CAS）のTLS検証を省略します。自己署名IdPに対するテスト専用です。本番環境では使用しないでください。 |
| `--server-external-url` 値                    | （空）                                          | ワーカー登録に使用するサーバーの外部URL。クラウドプロバイダー経由でワーカーをプロビジョニングする場合に必要で、ワーカーがサーバーに正しく接続できるようにします。 |
| `--trusted-hosts` 値                          | （空）                                          | リバースプロキシ経由で受け取る`the X-Forwarded-Host`ヘッダーの許可リスト。未設定の場合は`--server-external-url`から導出します。両方が未設定の場合、デフォルトの`*`によりすべてのホストを信頼します（信頼できるプロキシ経由でのみサーバーに到達できる場合を除き、推奨しません）。 |
| `--saml-sp-attribute-prefix` 値               | （空）                                          | SAML Service Providerの属性プレフィックス。--external-auth-\*で指定した属性を取得する際に使用します。 |
| `--gateway-concurrency` 値                    | `16`                                             | 組み込みゲートウェイの同時接続数。 |
| `--disable-builtin-observability`                | `False`                                          | 組み込みGrafanaおよびPrometheusサービスを無効にします。 |
| `--builtin-prometheus-port` 値                | `19090`                                          | 組み込みPrometheusサービスのポート。 |
| `--builtin-grafana-port` 値                   | `13000`                                          | 組み込みGrafanaサービスのポート。 |
| `--prometheus-url` 値                         | （空）                                          | クエリー先となる外部PrometheusのベースURL。未設定の場合、有効になっていれば組み込みPrometheusを使用します。可観測性を独自のスタックに委譲する構成でも、メトリクスに依存する機能（モデルとPDのダッシュボード、帯域幅のセルフチェック）を動作させるために設定します。 |
| `--grafana-url` 値                            | （空）                                          | ダッシュボードのリダイレクトとプロキシーに使用するGrafanaのベースURL。コンテナー内専用のホスト名ではなく、ブラウザーから到達できる必要があります。設定すると組み込みGrafanaとPrometheusは無効になります。外部Grafanaを使用する場合のみ必要です。 |
| `--grafana-worker-dashboard-uid` 値           | （空）                                          | ワーカーダッシュボードへのリダイレクトに使用するGrafanaダッシュボードUID。 |
| `--grafana-model-dashboard-uid` 値            | （空）                                          | モデルダッシュボードへのリダイレクトに使用するGrafanaダッシュボードUID。 |

### ワーカーオプション { #worker-options }

| <div style="width:180px">フラグ</div>      | <div style="width:100px">デフォルト</div> | 説明 |
|------------------------------------------|----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-t` 値, `--token` 値              | 自動生成。                        | ワーカー登録に使用する共有シークレット。 |
| `-s` 値, `--server-url` 値         | （空）                                | 接続先のサーバー。 |
| `--worker-name` 値                    | （空）                                | ワーカーノード名。デフォルトではホスト名を使用します。 |
| `--worker-ip` 値                      | （空）                                | ワーカーノードのIPアドレス。デフォルトでは自動検出します。 |
| `--disable-worker-metrics`               | `False`                                | メトリクスを無効にします。 |
| `--worker-metrics-port` 値            | `10151`                                | メトリクスを公開するポート。 |
| `--worker-port` 値                    | `10150`                                | ワーカーがバインドするポート。すべてのワーカーで同じ値を使用してください。 |
| `--service-port-range` 値             | `40000-40063`                          | 推論サービスのポート範囲。'N1-N2'形式の文字列で指定します。範囲の両端を含みます。 |
| `--ray-port-range` 値                 | `41000-41999`                          | Rayサービス（vLLMの分散デプロイで使用）のポート範囲。'N1-N2'形式の文字列で指定します。範囲の両端を含みます。 |
| `--log-dir` 値                        | （空）                                | ログを保存するディレクトリー。 |
| `--system-reserved` 値                | （空）                                | スケジューリング時にワーカー用として確保するシステムリソース（GiB単位）。デフォルトではリソースを予約しません。例：'{"ram": 2, "vram": 1}'。 |
| `--enable-hf-transfer`                   | `False`                                | ［非推奨］huggingface_hub v1.0でhf_transferのサポートが削除されたため、このフラグは効果を持ちません。現在のデフォルトのダウンローダーはhf_xetです。 |
| `--enable-hf-xet`                        | `False`                                | ［非推奨］Hugging Face Xetを使用したモデルファイルのダウンロードを有効にします。 |
| `--worker-ifname` 値                  | （空）                                | ワーカーノードのネットワークインターフェース名。デフォルトでは自動検出します。 |
| `--kv-transfer-ifname` 値                      | （空）                                | 分離型（prefill/decode）デプロイでKVキャッシュ転送に使用するネットワークインターフェース。ホストにルーティング可能なインターフェースが1つだけある場合は自動検出します。複数ある場合は指定が必要です。誤ったインターフェースを選ぶと、エンジンが相手から到達できないアドレスを通知するためです。 |
| `--proxy-mode` 値                     | （空）                                | モデルインスタンスへのアクセスに使用するプロキシーモード：`direct`、`worker`、`tunnel`（ワーカーがサーバーへ接続する方式。ファイアウォール/NAT配下のワーカー向け）。組み込みワーカーのデフォルトは`direct`、スタンドアロンワーカーは`worker`です。 |
| `--benchmark-image-repo` 値           | `gpustack/benchmark-runner`            | GPUStackベンチマークコンテナーのデフォルトのイメージリポジトリーを上書きします。`--image-repo`と異なり、この値はそのまま使用されるためタグを含めてください。<br/>イメージ`gpustack/benchmark-runner:<tag>`が`gpustack`名前空間の外（例：`awesome.com/amazing-group/amazing-subgroup/gpustack-ai/benchmark-runner:<tag>`）にある場合、まず`--system-default-container-registry=awesome.com/amazing-group/amazing-subgroup`で親の名前空間を含むレジストリーを指定し、次にこのフラグ（またはワーカーの設定キー`benchmark_image_repo`、環境変数`GPUSTACK_BENCHMARK_IMAGE_REPO`）を`gpustack-ai/benchmark-runner:<tag>`に設定してください。 |
| `--benchmark-dir` 値                  | `<data-dir>/benchmarks`                | ベンチマーク結果を保存するディレクトリー。 |
| `--benchmark-max-duration-seconds` 値 | （空）                                | ベンチマークがタイムアウトするまでの最大時間。空の場合は無効です。 |

### 利用可能な環境変数 { #available-environment-variables }

ほとんどのコマンドラインパラメーターは、`GPUSTACK_`プレフィックスを付け、名前を大文字にした
環境変数でも設定できます（例：`--data-dir`は`GPUSTACK_DATA_DIR`で設定できます）。

上記のコマンドラインパラメーター以外の環境変数については、
[環境変数のドキュメント](../environment-variables.md)を参照してください。

## 設定ファイル { #config-file }

GPUStackのサーバーまたはワーカーの起動時に、YAML形式の設定ファイルで起動オプションを指定できます。
以下は完全な設定例です。

```yaml
# Common Options
advertise_address: exposed_server_or_worker_ip
debug: false
data_dir: /path/to/data_dir
cache_dir: /path/to/cache_dir
benchmark_dir: /path/to/benchmark_dir
benchmark_image_repo: gpustack/benchmark-runner
token: your_token
huggingface_token: your_huggingface_token
insecure_tls: false

# Server Options
port: 80
tls_port: 443
api_port: 30080
metrics_port: 10161
disable_metrics: false
enable_worker: false
bootstrap_password: your_admin_password
database_url: postgresql://username:password@host:port/dbname
# database_url: mysql://username:password@host:port/dbname
ssl_keyfile: /path/to/keyfile
ssl_certfile: /path/to/certfile
ssl_ca_certfile: /path/to/ca-bundle.pem
disable_update_check: false
disable_openapi_docs: false
model_catalog_file: /path_or_url/to/model_catalog_file
enable_cors: false
allow_credentials: false
allow_origins: [ "*" ]
allow_methods: [ "GET", "POST" ]
allow_headers: [ "Authorization", "Content-Type", "X-API-Key" ]
oidc_issuer: https://your_oidc_issuer
oidc_client_id: your_oidc_client_id
oidc_client_secret: your_oidc_client_secret
oidc_redirect_uri: http://your_gpustack_server_url/auth/oidc/callback
saml_idp_server_url: https://your_saml_idp_server_url
saml_idp_entity_id: your_saml_idp_entity_id
saml_idp_x509_cert: your_saml_idp_x509_cert_pem
saml_sp_entity_id: your_saml_sp_entity_id
saml_sp_acs_url: http://your_gpustack_server_url/auth/saml/callback
saml_sp_x509_cert: your_saml_sp_x509_cert_pem
saml_sp_private_key: your_saml_sp_private_key_pem
saml_security: '{"wantAssertionsSigned": true, "wantMessagesSigned": true}'
cas_server_url: https://your_cas_server_url
cas_username_attribute: uid
cas_full_name_attribute: displayName
external_auth_provider_name: Okta
external_auth_name: email
external_auth_full_name: name
external_auth_avatar_url: picture
server_external_url: http://your_gpustack_server_url_for_external_access
trusted_hosts: [ "your_reverse_proxy_hostname" ]
disable_builtin_observability: false
builtin_prometheus_port: 19090
builtin_grafana_port: 13000
# Per-plugin overrides for the API gateway, keyed by the plugin's manifest name
# -- which for several of them is not the name of the WasmPlugin resource they
# are deployed as (`transformer` is deployed as `gpustack-header-transformer`,
# `ai-statistics` as `gpustack-ai-statistics`). Keying by the wrong one matches
# nothing, silently.
#
# `url` / `sha256` / `image_pull_policy` say where Envoy pulls the module from,
# replacing the address derived from the bundled plugin manifest. `config` is
# the plugin's own settings and is validated against what that plugin accepts,
# so a field it does not declare fails startup rather than being ignored.
gateway_plugin:
  gpustack-ext-auth:
    config:
      local_auth:
        # Whether the gateway authenticates API keys itself. Turned off, every
        # credential is forwarded to the server instead, as it was before.
        enabled: true
      authz:
        # How long the gateway waits for the server to authorize a request.
        # Sized for a loaded server rather than a healthy one: a server that is
        # down refuses the connection and is detected at once, so this governs
        # only the case of one too busy to answer yet -- where waiting is what
        # keeps a load spike from being read as an outage.
        timeout: 30000
      # Let a caller the gateway authenticated itself through while the server
      # is unreachable -- a rolling upgrade included. Authorization is skipped
      # for the duration; unknown keys and anonymous callers are still refused.
      failure_mode_allow_authenticated: true
  ai-statistics:
    config:
      enable_content_types: [ application/json, text/event-stream ]
  # Fields are the plugin's own camelCase config keys; only whitelisted ones
  # are accepted, so a field the server does not vouch for fails startup
  # instead of being forwarded.
  gpustack-ai-proxy:
    config:
      # Reject tool-call messages that arrive without the assistant message
      # that initiated them ("strict"), or the plugin's default ("off").
      toolCallValidation: strict

# Worker Options
server_url: http://your_gpustack_server_url
worker_name: your_worker_name
worker_ip: 192.168.1.101
disable_worker_metrics: false
worker_metrics_port: 10151
worker_port: 10150
service_port_range: 40000-40063
ray_port_range: 41000-41999
log_dir: /path/to/log_dir
system_reserved:
  ram: 2
  vram: 1
enable_hf_transfer: false
proxy_mode: worker
```
