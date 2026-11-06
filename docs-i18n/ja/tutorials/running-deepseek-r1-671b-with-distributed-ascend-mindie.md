---
translation_status: translated
---

# 分散 Ascend MindIE による DeepSeek R1 671B の実行 { #running-deepseek-r1-671b-with-distributed-ascend-mindie }

このチュートリアルでは、GPUStack クラスターで**分散 Ascend MindIE** を使用し、オリジナルの **DeepSeek R1 671B** を構成・実行します。

非常に大規模なモデルのため、通常は複数ワーカーにまたがる分散推論が必要です。

GPUStack は Ascend MindIE の分散推論を簡単に設定・オーケストレーションでき、最小限の手動設定で DeepSeek R1 のような巨大モデルを実行できます。

## 前提条件 { #prerequisites }

開始前に、次の条件を満たしていることを確認します。

- 必要な Ascend Atlas AI プロセッサーを搭載した Linux ノードを十分な台数用意します。例：

<div class="center-table" markdown>

| **サーバー（NPU）** | **ノード数** |
| ------------------------- | ------------------- |
| Atlas 800T A2 (910B3 x 8) | 4                   |

</div>

- 全ノードで Huawei Collective Communication adaptive Protocol（HCCP）に対応し、Huawei Collective Communication Library（HCCL）をインストールし、Huawei Cache Coherence System（HCCS）を有効化する必要があります。
- モデルファイルを各ノードの同じパスにダウンロードします。GPUStack は実行時にも取得できますが、ネットワーク速度によって時間がかかるため、事前取得を推奨します。

!!! note

    - この例では、各ノードに 910B3 NPU を 8 基搭載し、200G Huawei Cache Coherence Network（HCCN）で接続した 4 ノードを使用します。
    - Atlas NPU は、DeepSeek R1 が元々使用する FP8 精度に対応しないため、[Unsloth](https://huggingface.co/unsloth/DeepSeek-R1-BF16) の BF16 バージョンを使います。

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

    - `/path/to/your/model` を、DeepSeek R1 のモデルファイルがある実際のパスに置き換えます。

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
    --env "ASCEND_VISIBLE_DEVICES=$(sudo ls /dev/davinci* | head -1 | grep -o '[0-9]\+' || echo "0")" \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --volume /path/to/your/model:/path/to/your/model \
    --runtime ascend \
    gpustack/gpustack \
    --server-url http://your_gpustack_server_ip_or_hostname \
    --token your_gpustack_cluster_token

```

!!! note

    - パス、IP アドレス・ホスト名、クラスタートークンのプレースホルダーを環境に合わせて置き換えます。
    - `/path/to/your/model` を、DeepSeek R1 のモデルファイルがある実際のパスに置き換えます。
    - システムに `hccn_tool` が正しくインストール・設定されていることを確認します。HCCN のネットワーク通信の検出に必要です。

すべてのワーカーを追加したら、GPUStack UI に戻ります。

`Workers` ページで、すべてのワーカーが Ready で、GPU が一覧に表示されていることを確認します。

![初期リソース](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/initial-resources.png)

## 手順 4：DeepSeek R1 モデルのデプロイ { #step-4-deploy-the-deepseek-r1-model }

1. `Deployments` ページに移動します。
2. `Deploy Model` をクリックします。
3. ソースに `Local Path` を選択します。
4. `Name` に名前（例：`DeepSeek-R1`）を入力します。
5. `Model Path` に、各ワーカーノードで DeepSeek R1 のファイルを格納するディレクトリを指定します。
6. `Backend` が `Ascend MindIE` であることを確認します。
7. `Advanced` を展開し、`Backend Parameters` に次を追加します。
   - `--data-parallel-size=4`
   - `--tensor-parallel-size=8`
   - `--moe-tensor-parallel-size=1`
   - `--moe-expert-parallel-size=32`
   - `--npu-memory-fraction=0.95`。データ並列を使用するため、全 NPU のメモリーを効率的に使うよう、割合を 0.95 にします。
8. 互換性チェックが成功したら、`Save` をクリックしてデプロイします。

![モデルのデプロイ設定](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/deploy-model-1.png)
![モデルの詳細設定](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/deploy-model-2.png)

## 手順 5：デプロイの監視 { #step-5-monitor-deployment }

`Deployments` ページで状態を監視できます。`distributed across workers` にポインターを合わせると、GPU とワーカーの使用状況を確認できます。`View Logs` をクリックすると、モデル読込の進行状況がリアルタイムに表示されます。読込には数分かかる場合があります。

![モデル情報](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/model-info.png)

モデル起動後、`Workers` ページで GPU 使用率を確認します。上記の設定では、Ascend MindIE は NPU メモリーの 95% を使用します。

![読込後のリソース](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/resources-loaded.png)

## 手順 6：プレイグラウンドでの推論 { #step-6-run-inference-via-playground }

モデルがデプロイされ実行中になったら、GPUStack Playground でテストできます。

1. `Playground` -> `Chat` に移動します。
2. モデルが 1 つだけならデフォルトで選択されます。複数ある場合はドロップダウンから `DeepSeek-R1` を選択します。
3. プロンプトを入力してモデルと対話します。

![プレイグラウンドのチャット](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/playground-chat.png)

`Compare` タブでは同時推論のシナリオもテストできます。

![プレイグラウンドの比較](../assets/tutorials/running-deepseek-r1-671b-with-distributed-ascend-mindie/playground-compare.png)

以上で、GPUStack クラスター上の分散 Ascend MindIE による DeepSeek R1 671B のデプロイと実行が完了しました。実際のアプリケーションでモデルの性能や機能を試してください。

さらに支援が必要な場合は、GPUStack コミュニティまたはサポートチームにお問い合わせください。
