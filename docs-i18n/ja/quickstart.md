---
translation_status: translated
---

# クイックスタート { #quickstart }

このガイドでは、自己管理の GPU サーバーで GPUStack を実行する手順を説明します。[クラウド GPU](./tutorials/adding-gpucluster-using-digitalocean.md) の利用や、[既存の Kubernetes クラスター](./tutorials/adding-gpucluster-using-kubernetes.md)との統合については、対応するチュートリアルをご覧ください。

!!! info "前提条件"

    1. NVIDIA GPU を 1 基以上搭載したノード。ほかの GPU については、ワーカー追加時に GPUStack UI に表示される案内、または[インストールドキュメント](./installation/requirements.md)を確認してください。
    2. ワーカーノードに NVIDIA ドライバー、[Docker](https://docs.docker.com/engine/install/)、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) がインストールされていることを確認します。
    3. **（任意）** GPUStack サーバーを実行する CPU ノード。GPUStack サーバーに GPU は不要で、CPU のみのマシンでも実行できます。[Docker](https://docs.docker.com/engine/install/) が必要です。Docker Desktop（Windows、macOS）にも対応しています。専用の CPU ノードがない場合は、GPU ワーカーノードと同じマシンに GPUStack サーバーをインストールできます。
    4. GPUStack ワーカーノードは Linux のみに対応しています。Windows を使用する場合は WSL2 を検討し、Docker Desktop は使用しないでください。macOS は GPUStack ワーカーノードに対応していません。

## GPUStack のインストール { #install-gpustack }

次のコマンドを実行し、[Docker](https://docs.docker.com/engine/install/) で GPUStack サーバーをインストールして起動します。

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

??? Note "代替方法：Quay Container Registry ミラーを使用する"

    Docker Hub からイメージを取得できない、またはダウンロードが非常に遅い場合は、レジストリを `quay.io` に指定して Quay Container Registry ミラーを使用できます。

    ```bash
    sudo docker run -d --name gpustack \
        --restart unless-stopped \
        -p 80:80 \
        --volume gpustack-data:/var/lib/gpustack \
        quay.io/gpustack/gpustack \
        --system-default-container-registry quay.io
    ```

GPUStack の起動ログを確認します。

```bash
sudo docker logs -f gpustack
```

GPUStack が起動したら、次のコマンドでデフォルトの管理者パスワードを取得します。

```bash
sudo docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

ブラウザーで `http://your_host_ip` を開き、GPUStack UI にアクセスします。デフォルトのユーザー名 `admin` と、取得したパスワードでログインします。

## GPU クラスターのセットアップ { #set-up-a-gpu-cluster }

1. GPUStack UI の `Clusters` ページに移動します。

2. `Add Cluster` ボタンをクリックします。

3. クラスタープロバイダーとして `Docker` を選択します。

4. 新しいクラスターの `Name` と `Description` を入力し、`Save` ボタンをクリックします。

5. UI の案内に従って新しいワーカーノードを構成します。ワーカーノードで Docker コマンドを実行し、GPUStack サーバーに接続します。コマンドは次のようになります。

```bash
sudo docker run -d --name gpustack-worker \
      --restart=unless-stopped \
      --privileged \
      --network=host \
      --volume /var/run/docker.sock:/var/run/docker.sock \
      --volume gpustack-data:/var/lib/gpustack \
      --runtime nvidia \
      gpustack/gpustack \
      --server-url http://your_gpustack_server_url \
      --token your_worker_token \
      --advertise-address 192.168.1.2
```

6. ワーカーノードでコマンドを実行して GPUStack サーバーに接続します。

7. 接続が成功すると、GPUStack UI の `Workers` ページにワーカーノードが表示されます。

## モデルのデプロイ { #deploy-a-model }

1. GPUStack UI の `Catalog` ページに移動します。

2. 使用可能なモデルの一覧から `Qwen3.5-0.8B` を選択します。

3. デプロイの互換性チェックが成功したら、`Save` ボタンをクリックしてモデルをデプロイします。

![カタログから Qwen3 をデプロイ](assets/quick-start/quick-start-qwen3.png)

4. GPUStack がモデルファイルのダウンロードとデプロイを開始します。デプロイの状態が `Running` になれば、モデルのデプロイは成功です。

!!! note

    GPUStack はコンテナーでモデルを実行します。初回のデプロイでは、モデルファイルとコンテナーイメージのダウンロードに時間がかかることがあります。UI の `View Logs` をクリックして進行状況を確認できます。

![実行中のモデル](assets/quick-start/model-running.png)

5. ナビゲーションメニューの `Playground - Chat` をクリックし、右上の `Model` ドロップダウンで `qwen3.5-0.8b` が選択されていることを確認します。UI のプレイグラウンドでモデルとチャットできます。

![クイックチャット](assets/quick-start/quick-chat.png)

## API 経由でのモデル利用 { #use-the-model-via-api }

1. `Access Control` > `API Keys` ページに移動し、`New API Key` ボタンをクリックします。

2. `Name` を入力し、`Save` ボタンをクリックします。

3. 生成された API キーをコピーし、安全な場所に保存します。キーが表示されるのは作成時の 1 回だけです。

4. API キーを使用して GPUStack が提供する OpenAI 互換 API エンドポイントにアクセスできます。たとえば、次のように curl を使用します。

```bash
# Replace `your_api_key` and `your_gpustack_server_url`
# with your actual API key and GPUStack server URL.
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Tell me a joke."
      }
    ],
    "stream": true
  }'
```

## クリーンアップ { #cleanup }

デプロイしたモデルの利用が終わったら、GPUStack UI の `Deployments` ページでモデルを削除してリソースを解放できます。
