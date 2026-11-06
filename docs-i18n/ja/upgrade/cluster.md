---
translation_status: translated
---

# クラスターのデプロイメントのアップグレード { #upgrade-a-cluster-deployment }

GPUStack はワーカーノードを[クラスター](../user-guide/cluster-management.md)にまとめて管理します。このページでは、GPUStack サーバーの更新後に、クラスターのワーカーデプロイメントをアップグレードする方法を説明します。

!!! note

    1. 最初に GPUStack **サーバー**をアップグレードし（[Docker によるアップグレード](docker.md)、[Helm によるアップグレード](helm.md)を参照）、次にクラスターのデプロイメントを更新します。

    2. ワーカーはサーバーと**同じバージョン**に更新します。安定版サーバーに `main`（開発版）やリリース候補（rc）のワーカーイメージを混在させないでください。

アップグレードの方法は、元のデプロイ方法に対応します（[クラスター管理](../user-guide/cluster-management.md)を参照）。Docker クラスターでは新しいイメージでデプロイを再実行し、Kubernetes クラスターでは新しいイメージを指定したマニフェストを再適用します。

## Docker クラスター { #docker-cluster }

Docker クラスターのワーカーは、クラスターの `Add Worker` ステップで生成される `docker run` コマンドで起動したコンテナーです。新しいイメージを取得し、古いコンテナーを削除して、以前と**同じ引数とボリューム**で再作成し、各ワーカーを更新します。

```bash
docker pull gpustack/gpustack:vx.y.z   # match the server version

docker stop gpustack-worker
docker rm gpustack-worker

docker run -d --name gpustack-worker \
  ... \
  gpustack/gpustack:vx.y.z \
  --server-url http://your_gpustack_server_url \
  --token your_worker_token \
  --advertise-address your_worker_ip
```

!!! tip

    対象バージョンの最新の `docker run` コマンドは GPUStack UI から取得できます。`Clusters` ページでクラスターの `Add Worker` ステップを開き、生成された `Run command` をコピーします。

## Kubernetes クラスター { #kubernetes-cluster }

Kubernetes クラスターのワーカーは、クラスター登録時に適用したマニフェストに基づき、GPUStack Operator が管理する DaemonSet としてデプロイされます。更新するには、新しいバージョン用に生成されたマニフェストを再適用します。

1. `Clusters` ページで更新する Kubernetes クラスターを見つけます。
2. 操作列の三点ボタンをクリックし、`Register Cluster` を選択します。
3. `Run command` ステップまで進み、生成されたコマンドをコピーします。
4. `kubectl` がインストールされ、対象クラスター用の `kubeconfig` が設定された環境でコマンドを実行します。更新されたマニフェストが再適用され、Operator がワーカーの DaemonSet を新しいバージョンにローリング更新します。

ワーカー Pod が `Ready` になれば、クラスターは新しいバージョンで稼働しています。

## クラウドプロバイダーのクラスター { #cloud-provider-cluster }

クラウドでプロビジョニングするクラスター（DigitalOcean など）では、各ワーカーノードを cloud-init で初期化します。設定は `/var/lib/gpustack/config.yaml` に書き込まれ、`/opt/gpustack-run-worker.sh` 経由で `gpustack-worker` コンテナーが起動します。既存ノードを更新するには、SSH で接続し、同じ設定とボリュームを再利用して新しいイメージでコンテナーを再作成します。

```bash
# SSH into the worker node, then:
NEW_IMAGE=gpustack/gpustack:vx.y.z   # match the server version

docker pull "$NEW_IMAGE"

docker rm -f gpustack-worker

docker run -d --name gpustack-worker \
  -e "GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME=gpustack-worker" \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  -v /var/lib/gpustack:/var/lib/gpustack \
  -v /var/run/docker.sock:/var/run/docker.sock \
  "$NEW_IMAGE" \
  --config-file=/var/lib/gpustack/config.yaml
```

!!! note

    ワーカープールのスケールアウトなどで**新たにプロビジョニングする**ノードも新しいバージョンで起動するように、UI でクラスターのイメージ設定も更新してください。[クラスター管理](../user-guide/cluster-management.md#operating-worker-pools)をご覧ください。
