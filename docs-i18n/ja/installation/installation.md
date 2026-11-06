---
translation_status: translated
---

# Docker によるインストール { #installation-via-docker }

## 前提条件 { #prerequisites }

**GPUStack サーバー：**

- [Docker](https://docs.docker.com/engine/install/) が必要です。Docker Desktop（Windows、macOS）にも対応しています。

**GPUStack ワーカー：**

- [Docker](https://docs.docker.com/engine/install/) が必要です。Docker Desktop には**対応していません**。
- GPUStack ワーカーノードは Linux のみに対応しています。Windows を使用する場合は WSL2 を検討し、Docker Desktop は使用しないでください。macOS は GPUStack ワーカーノードに対応していません。
- ハードウェアに適した GPU ドライバーとコンテナーツールキットがインストールされていることを確認します。詳細は[インストール要件](./requirements.md)をご覧ください。

## GPUStack サーバーのインストール { #install-gpustack-server }

次のコマンドを実行して、Docker で GPUStack サーバーをインストール・起動します。

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

!!! note

    GPUStack v2 は、すべての GPU デバイスタイプに共通のコンテナーイメージを使用します。

## 起動 { #startup }

GPUStack コンテナーのログを確認します。

```bash
sudo docker logs -f gpustack
```

サーバーが起動したら、ブラウザーで `http://your_host_ip` を開き、GPUStack UI にアクセスします。

ユーザー名 `admin` とデフォルトのパスワードでログインします。初期パスワードは次のコマンドで取得できます。

```bash
sudo docker exec -it gpustack \
    cat /var/lib/gpustack/initial_admin_password
```

## GPU クラスターとワーカーノードの追加 { #add-gpu-clusters-and-worker-nodes }

`Clusters` と `Workers` ページの案内に従って、GPU クラスターとワーカーノードを追加します。

## カスタム設定 { #custom-configuration }

以下では、GPUStack サーバーコンテナーの起動時に指定できるカスタム設定の例を説明します。使用可能なオプションの一覧は、[CLI リファレンス](../cli-reference/start.md)をご覧ください。

### カスタム証明書による HTTPS の有効化 { #enable-https-with-custom-certificate }


```diff
 sudo docker run -d --name gpustack \
     ...
     -p 80:80 \
+    -p 443:443 \
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/cert_files:/path/to/cert_files:ro \
+    -e GPUSTACK_SSL_KEYFILE=/path/to/cert_files/your_domain.key \
+    -e GPUSTACK_SSL_CERTFILE=/path/to/cert_files/your_domain.crt \
     gpustack/gpustack
     ...
```

`GPUSTACK_SSL_CERTFILE` にはサーバー証明書を格納し、その後に中間証明書を
続けます。サーバー証明書がプライベート CA によって署名されている場合は、
PEM 形式の CA バンドルも追加すると、ワーカーとベンチマークコンテナーが
サーバーを自動的に検証できます。

```diff
 sudo docker run -d --name gpustack \
     ...
+    -e GPUSTACK_SSL_CA_CERTFILE=/path/to/cert_files/ca-bundle.crt \
     gpustack/gpustack
     ...
```

自己署名サーバー証明書の場合、別の CA バンドルが設定されていなければ
証明書ファイル自体が使用されます。パブリック CA が発行した証明書には、
追加の CA 設定は不要です。

登録コマンドには、ブートストラップ CA バンドルの SHA-256 チェックサムが含まれます。
ワーカーは、TLS 検証を行わずにダウンロードしたバンドルをインストールする前に、
このチェックサムを検証します。これは証明書ピンニングではありません。
サーバーが通常の TLS 検証に成功する場合、ワーカーは既存の信頼ストアを使用し、
ブートストラップバンドルのダウンロードや確認は行いません。

ワーカーは、検証済み CA をまずシステムの信頼ストアにインストールします。
ストアを更新できない場合、またはワーカーの TLS クライアントが別のバンドルを
使用する場合は、一時的に統合したバンドルを `SSL_CERT_FILE` で指定し、
自身と子プロセスで使用します。運用担当者が提供した既存の証明書は保持されます。
注入されたブートストラップ CA を置換する場合、以前に注入したバンドルは保持されません。

ワーカーでサーバー証明書をまったく検証できない場合の最終手段として、
`--insecure-tls`（または `GPUSTACK_INSECURE_TLS=true`）で起動すると、
サーバーへの接続時の証明書検証を省略できます。通信は暗号化されますが、
傍受に対する保護は失われるため、信頼できるネットワークでのみ使用してください。
詳細は [CLI リファレンス](../cli-reference/start.md)をご覧ください。

### 外部データベースの使用 { #using-an-external-database }

GPUStack はデフォルトで組み込み PostgreSQL データベースを使用します。PostgreSQL や MySQL などの外部データベースを使用するには、GPUStack コンテナーの起動時に `GPUSTACK_DATABASE_URL` 環境変数または `--database-url` 引数を設定します。対応データベースと検証済みバージョンの一覧は、[データベース要件](requirements.md#database-requirements)をご覧ください。

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    -e GPUSTACK_DATABASE_URL="postgresql://username:password@host:port/dbname" \
     gpustack/gpustack
     ...
```

### 外部サーバー URL の設定 { #configure-external-server-url }

クラウドプロバイダーでワーカーをプロビジョニングする場合は、ワーカー登録用の外部サーバー URL を設定し、ワーカーがサーバーに正しく接続できるようにします。

```diff
sudo docker run -d --name gpustack \
    ...
+   -e GPUSTACK_SERVER_EXTERNAL_URL="https://your_external_server_url" \
    gpustack/gpustack
    ...
```

### 追加の信頼済み CA { #additional-trusted-cas }

GPUStack がプライベート CA や企業内 CA の発行した証明書を使用するサービス（自己ホストの ID プロバイダー、Hugging Face ミラー、内部 API エンドポイントなど）と通信する場合は、CA 証明書をコンテナーの `/usr/local/share/ca-certificates/` 配下にマウントします。GPUStack は起動時にマウントされた CA 証明書を自動でインポートし、システムの信頼ストアに追加します。

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/custom-root-ca.crt:/usr/local/share/ca-certificates/custom-root-ca.crt:ro \
     gpustack/gpustack
     ...
```

!!! note

    CA 証明書は PEM 形式で、拡張子を `.crt` にする必要があります。`--volume` フラグを追加すれば、複数の CA 証明書をマウントできます。

## Docker Compose によるインストール { #installation-via-docker-compose }

### 前提条件 { #prerequisites_1 }

- [Docker Compose](https://docs.docker.com/compose/install/) が必要です。
- [必要なポート](./requirements.md#port-requirements)が使用可能であることを確認します。

### デプロイ { #deployment }

Docker Compose ファイルと設定ファイルは、[GPUStack リポジトリ](https://github.com/gpustack/gpustack/tree/main/docker-compose)で管理されています。

次のコマンドを実行して、最新の安定版リリースをクローンします。

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

GPUStack サーバーを起動します。

```bash
sudo docker compose -f docker-compose.server.yaml up -d
```

サーバーが起動したら、ブラウザーで `http://your_host_ip` を開き、GPUStack UI にアクセスします。

ユーザー名 `admin` とデフォルトのパスワードでログインします。初期パスワードは次のコマンドで取得できます。

```bash
sudo docker exec -it gpustack-server cat /var/lib/gpustack/initial_admin_password
```

組み込みおよび外部の監視機能については、[オブザーバビリティ](../user-guide/observability.md)をご覧ください。
