---
translation_status: translated
---

# Kubernetes を使った GPU クラスターの追加 { #adding-a-gpu-cluster-using-kubernetes }

GPUStack は、Kubernetes クラスターを GPU クラスターとして追加できます。

## 事前準備 { #preparation }

すべてのノードで runtime class をインストール・設定した Kubernetes クラスターを用意します。この例では k3s を使用します。

## Kubernetes クラスターの作成 { #create-kubernetes-cluster }

[Kubernetes クラスターの登録](../user-guide/cluster-management.md#register-kubernetes-cluster)の手順を参照してください。

作成時には、次のオプションを設定できます。

![Kubernetes クラスターのオプション](../assets/tutorials/adding-gpucluster-using-kubernetes/k8s-options.png)

`Save` をクリックしてクラスターを作成します。

`Kubernetes` プロバイダーのクラスター作成後、必要な `GPU Vendors` を選択します。複数選択も、選択なしも可能です。

![NVIDIA GPU ベンダー](../assets/tutorials/adding-gpucluster-using-kubernetes/nvidia-gpu-vendor.png)

`Next` をクリックすると、登録の準備状態を確認する環境チェックスクリプトが表示されます。k3s をインストールしたホストで実行します。

![環境チェック](../assets/tutorials/adding-gpucluster-using-kubernetes/check-environment.png)

Kubernetes の `runtimeclass` リソースは情報のレコードにすぎず、各ノードのコンテナーランタイムが正しく設定されていることを保証しません。対象 GPU ベンダーのランタイムが設定済みでないと、ワーカーは起動できません。

`Run Command` ステップのスクリプトをコピーし、k3s をインストールしたホストで実行します。

![実行コマンド](../assets/tutorials/adding-gpucluster-using-kubernetes/run-command.png)

![マニフェストの適用](../assets/tutorials/adding-gpucluster-using-kubernetes/apply-manifests.png)

## ワーカーのプロビジョニングの待機 { #waiting-for-workers-to-be-provisioned }

Kubernetes クラスターにマニフェストを適用したら、`Workers` ページにワーカーノードが表示されるまで待ちます。

![プロビジョニング済みワーカー](../assets/tutorials/adding-gpucluster-using-kubernetes/provisioned-workers.png)

ワーカーが `Ready` になれば、モデルをデプロイできます。
