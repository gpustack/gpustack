---
translation_status: translated
---

# エアギャップ環境へのインストール { #air-gapped-installation }

GPUStack は、インターネットに接続できないエアギャップ（オフライン）環境にインストールできます。

起動後はサーバー自体がこのドキュメントを配信します。下の[ドキュメント](#documentation)をご覧ください。

## 前提条件 { #prerequisites }

### ドライバー { #driver }

ハードウェアに適した GPU ドライバーがシステムにインストールされていることを確認します。

ドライバーの互換性については、[インストール要件](./requirements.md)をご覧ください。

### コンテナー実行環境 { #container-running-environment }

[Docker](https://docs.docker.com/engine/install/) が必要です。

システムがコンテナーツールキットに対応している場合は、必要に応じてインストール・設定します（NVIDIA Container Toolkit、AMD ROCm Container Toolkit など）。

### コンテナーイメージ { #container-images }

GPUStack の[イメージセレクター](../image-selector/index.html)を使用すると、ダウンロードするイメージを簡単に選択できます。このドキュメントに同梱されたデータを使い、すべてブラウザー内で実行するため、インターネット接続なしでも動作します。バージョン一覧には、同梱時点までのリリースのみが含まれます。高度な同期や自動同期には、次のイメージ管理コマンドも使用できます。

- `gpustack copy-images`：レジストリ間でイメージを同期します。
- `gpustack save-images`：イメージをダウンロードしてローカルに保存します。
- `gpustack load-images`：ローカルパッケージからイメージをインポートします。
- `gpustack list-images`：現在のバージョンのイメージマニフェストを表示します。

以下では、これらの CLI コマンドの使い方を説明します。

- **イメージのコピー**

GPUStack は、各種コンポーネントと推論バックエンドのコンテナーイメージを [Docker Hub](https://hub.docker.com/u/gpustack) と [Quay.io](https://quay.io/user/gpustack/) で提供しています。

インターネットに接続できるマシンから必要なコンテナーイメージを内部レジストリに転送するには、GPUStack の `copy-images` コマンドを使用します。

```bash
sudo docker run --rm -it --entrypoint "" gpustack/gpustack \
    gpustack copy-images \
    --destination <your_internal_registry> \
    --destination-username <your_username> \
    --destination-password <your_password>
```

`Docker Hub` からイメージを取得できない、またはダウンロードが非常に遅い場合は、ソースレジストリを `quay.io` に指定して `Quay.io` ミラーを使用できます。

```bash hl_lines="3"
sudo docker run --rm -it --entrypoint "" gpustack/gpustack \
    gpustack copy-images \
    --source quay.io \
    --destination <your_internal_registry> \
    --destination-username <your_username> \
    --destination-password <your_password>
```

!!! note

    デフォルトでは最新バージョンを使用します。特定のバージョンを対象にするには、gpustack/gpustack:vx.y.z のような完全なイメージタグを指定します。

`copy-images` の詳細は、[CLI リファレンス](../cli-reference/copy-images.md)をご覧ください。

- **イメージ一覧の表示**

内部レジストリに直接アクセスできない場合は、まず `gpustack/gpustack` イメージを取得し、`list-images` コマンドでダウンロードが必要なイメージを確認できます。

```bash
sudo docker run --rm -it --entrypoint "" \
    gpustack/gpustack \
    gpustack list-images
```

!!! note

    デフォルトでは最新バージョンを使用します。特定のバージョンを対象にするには、gpustack/gpustack:vx.y.z のような完全なイメージタグを指定します。

表示される一覧には、対応するすべてのアクセラレーター、推論バックエンド、バージョン、アーキテクチャが含まれます。一部のみ必要な場合は、[CLI リファレンス](../cli-reference/list-images.md)でフィルタリングのオプションを確認してください。

- **イメージのコピーと読み込み**

対象がエアギャップ環境、またはインターネットに接続できない環境の場合は、接続可能なマシンで必要なイメージをダウンロードし、オフライン環境へ転送して読み込みます。

この手順には、GPUStack の `save-images` と `load-images` コマンドを使用します。

**イメージのコピー**

インターネットに接続できるマシンで次のコマンドを実行し、必要なイメージをダウンロードしてパッケージ化します。

```bash
sudo docker run --rm -it --entrypoint "" \
 -v ./gpustack-air-gapped:/gpustack-air-gapped \
 --workdir /gpustack-air-gapped \
 gpustack/gpustack \
 gpustack save-images \
 --platform linux/amd64 \
 --backend cuda \
 --backend-version 12.9 \
 --service vllm \
 --service-version 0.15.1 \
 --max-workers 3 \
 --max-retries 3
```

このコマンドは、指定したプラットフォーム、バックエンド、サービス構成に基づいて必要なコンテナーイメージを取得し、`gpustack-air-gapped` ディレクトリにローカルパッケージとして保存します。この例では、フィルターで一部のイメージだけをダウンロードする方法を示しています。

!!! note

    デフォルトでは最新バージョンを使用します。特定のバージョンを対象にするには、gpustack/gpustack:vx.y.z のような完全なイメージタグを指定します。

フィルターを調整して、必要なイメージだけをダウンロードできます。すべてのオプションは、[CLI リファレンス](../cli-reference/save-images.md)をご覧ください。

ダウンロードが完了したら、USB ドライブや内部のファイル転送など、任意の方法で生成されたディレクトリをエアギャップ環境に移します。

**イメージの読み込み**

対象のマシンで、保存済みのイメージをローカルのコンテナーランタイムに読み込みます。

```bash
sudo docker run --rm -it --entrypoint "" \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume ./gpustack-air-gapped:/gpustack-air-gapped \
    --workdir /gpustack-air-gapped \
    gpustack/gpustack \
    gpustack load-images \
    --platform linux/amd64 \
    --max-workers 3 \
    --max-retries 3 \
    /gpustack-air-gapped
```

このコマンドは、指定したディレクトリ内のすべてのイメージパッケージをローカル Docker デーモンにインポートし、GPUStack から利用できるようにします。

!!! note

    デフォルトでは最新バージョンを使用します。特定のバージョンを対象にするには、gpustack/gpustack:vx.y.z のような完全なイメージタグを指定します。

`load-images` の詳細は、[CLI リファレンス](../cli-reference/load-images.md)をご覧ください。

## インストール { #installation }

必要なイメージを内部コンテナーレジストリに用意したら、エアギャップ環境に GPUStack をインストールできます。

```diff
 sudo docker run -d --name gpustack \
     --restart unless-stopped \
     -p 80:80 \
     --volume gpustack-data:/var/lib/gpustack \
-    gpustack/gpustack
+    <your_internal_registry>/gpustack/gpustack \
+    --system-default-container-registry <your_internal_registry>

```

### 認証が必要なレジストリからの推論バックエンドイメージの取得 { #pulling-inference-backend-images-from-a-secure-registry }

内部コンテナーレジストリで認証が必要な場合は、  
GPUStack ワーカーの起動時に次の環境変数を設定し、ランナーイメージを取得できるようにします。

```diff
 sudo docker run -d --name gpustack \
     ...
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME=<your_internal_registry_username> \
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD=<your_internal_registry_password> \
     <your_internal_registry>/gpustack/gpustack \
     --system-default-container-registry <your_internal_registry>

```

### デフォルト以外の名前空間からの推論バックエンドイメージの取得 { #pulling-inference-backend-images-from-non-default-namespace }

内部コンテナーレジストリでデフォルトの `gpustack` 以外の名前空間を使用する場合は、  
GPUStack ワーカーの起動時に次の環境変数を設定し、ランナーイメージを取得できるようにします。

```diff
 sudo docker run -d --name gpustack \
     ...
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE=<your_namespace> \
     <your_internal_registry>/gpustack/gpustack \
     --system-default-container-registry <your_internal_registry>
```

## ドキュメント { #documentation }

サーバーにはこのドキュメントが同梱され、`/help/` で配信されるため、エアギャップ環境からオンラインサイトに接続する必要はありません。

```
http://<your_gpustack_server>/help/
```

内容はサーバーイメージのビルド元バージョンのオンラインドキュメントと同じで、検索もオフラインで動作します。
