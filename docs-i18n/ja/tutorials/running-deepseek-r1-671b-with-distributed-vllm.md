---
translation_status: translated
---

# 分散 vLLM による DeepSeek R1 671B の実行 { #running-deepseek-r1-671b-with-distributed-vllm }

このチュートリアルでは、GPUStack クラスターで**分散 vLLM** を使用し、オリジナルの **DeepSeek R1 671B** を構成・実行します。非常に大規模なモデルのため、通常は複数ワーカーにまたがる分散推論が必要です。

GPUStack は vLLM の分散推論を簡単に設定・オーケストレーションでき、最小限の手動設定で DeepSeek R1 のような巨大モデルを実行できます。

## 前提条件 { #prerequisites }

開始前に、次の条件を満たしていることを確認します。

- 必要な GPU を搭載した Linux ノードを十分な台数用意します。例：

<div class="center-table" markdown>

| **GPU** | **ノード数** |
| ---------------- | ------------------- |
| H100/H800:8      | 2                   |
| A100/A800-80GB:8 | 4                   |
| A100/A800:8      | 8                   |

</div>
- 性能を最大限に引き出すため、NVLink や InfiniBand などの高速相互接続を推奨します。
- モデルファイルを各ノードの同じパスにダウンロードします。GPUStack は実行時にも取得できますが、ネットワーク速度によって時間がかかるため、事前取得を推奨します。

!!! note

    - この例では、各ノードに A800-80GB GPU を 8 基搭載し、200G InfiniBand で接続した 4 ノードを使用します。
    - A100/A800 GPU は、DeepSeek R1 が元々使用する FP8 精度に対応しないため、[Unsloth](https://huggingface.co/unsloth/DeepSeek-R1-BF16) の BF16 バージョンを使います。

## 手順 1：GPUStack サーバーのインストール { #step-1-install-gpustack-server }

[インストール](../installation/installation.md)に従い、次のコマンドで GPUStack サーバーを起動します。

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    --volume /path/to/your/model:/path/to/your/model \
    gpustack/gpustack

```

!!! note

    - `/path/to/your/model` を実際のパスに置き換えます。

GPUStack サーバーが起動したら、次のコマンドで初期管理者パスワードを取得します。

```bash
sudo docker exec gpustack \
    cat /var/lib/gpustack/initial_admin_password

```

## 手順 2：GPUStack UI へのアクセス { #step-2-access-gpustack-ui }

`admin` ユーザーと取得したパスワードで GPUStack UI にログインします。

```
http://your_gpustack_server_ip_or_hostname
```

## 手順 3：GPUStack ワーカーのインストール { #step-3-install-gpustack-workers }

GPUStack UI の `Workers` ページで `Add Worker` をクリックし、追加用のコマンドを取得します。

**各ワーカーノード**で追加コマンドを実行し、GPUStack ワーカーを起動します。

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --volume /path/to/your/model:/path/to/your/model \
    --runtime nvidia \
    gpustack/gpustack \
    --server-url http://your_gpustack_server_ip_or_hostname \
	--token your_gpustack_cluster_token

```

!!! note

    - パス、IP アドレス・ホスト名、クラスタートークンのプレースホルダーを環境に合わせて置き換えます。
    - `/path/to/your/model` を、DeepSeek R1 のモデルファイルがある実際のパスに置き換えます。

すべてのワーカーを追加したら、GPUStack UI に戻ります。

`Workers` ページで、すべてのワーカーが Ready で、GPU が一覧に表示されていることを確認します。

![初期リソース](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/initial-resources.png)

## 手順 4：DeepSeek R1 モデルのデプロイ { #step-4-deploy-the-deepseek-r1-model }

1. `Deployments` ページに移動します。
2. `Deploy Model` をクリックします。
3. ソースに `Local Path` を選択します。
4. `Name` に名前（例：`DeepSeek-R1`）を入力します。
5. `Model Path` に、各ワーカーノードで DeepSeek R1 のファイルを格納するディレクトリを指定します。
6. `Backend` が `vLLM` であることを確認します。
7. 互換性チェックが成功したら、`Save` をクリックしてデプロイします。

![モデルのデプロイ](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/deploy-model.png)

## 手順 5：デプロイの監視 { #step-5-monitor-deployment }

`Deployments` ページで状態を監視できます。`distributed across workers` にポインターを合わせると、GPU とワーカーの使用状況を確認できます。`View Logs` をクリックすると、モデル読込の進行状況がリアルタイムに表示されます。読込には数分かかる場合があります。

![モデル情報](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/model-info.png)

モデル起動後、`Workers` ページで GPU 使用率を確認します。vLLM はデフォルトで GPU メモリーの 90% を使用し、モデル設定で調整できます。

![読込後のリソース](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/resources-loaded.png)

## 手順 6：プレイグラウンドでの推論 { #step-6-run-inference-via-playground }

モデルがデプロイされ実行中になったら、GPUStack Playground でテストできます。

1. `Playground` -> `Chat` に移動します。
2. モデルが 1 つだけならデフォルトで選択されます。複数ある場合はドロップダウンから `DeepSeek-R1` を選択します。
3. プロンプトを入力してモデルと対話します。

![プレイグラウンドのチャット](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/playground-chat.png)

`Compare` タブでは同時推論のシナリオもテストできます。

![プレイグラウンドの比較](../assets/tutorials/running-deepseek-r1-671b-with-distributed-vllm/playground-compare.png)

以上で、GPUStack クラスター上の分散 vLLM による DeepSeek R1 671B のデプロイと実行が完了しました。実際のアプリケーションでモデルの性能や機能を試してください。

さらに支援が必要な場合は、GPUStack コミュニティまたはサポートチームにお問い合わせください。
