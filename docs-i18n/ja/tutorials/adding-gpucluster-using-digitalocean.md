---
translation_status: translated
---

# DigitalOcean を使った GPU クラスターの追加 { #adding-a-gpu-cluster-using-digitalocean }

クラスターの作成時に、GPUStack は DigitalOcean でワーカーを作成し、クラスターに追加できます。

## 事前準備 { #preparation }

DigitalOcean アカウントを作成し、API ページで Personal Access Token を発行する必要があります。

![DigitalOcean の個人用アクセストークン](../assets/tutorials/adding-gpucluster-using-digitalocean/digital-ocean-pat.png)

> 注意：トークンのスコープは Full Access にしてください。Custom Scopes で権限を選ぶと、Droplet を削除できない場合があります。

GPUStack サーバーの起動時に `--server-external-url` を指定します。これは Droplet 作成後のワーカーの `--server-url` に使われます。サーバーがプロキシの背後にある場合は、パブリックネットワーク上の Droplet が起動後に GPUStack Server API にアクセスできるよう、プロキシのアドレスを指定してください。

## DigitalOcean クラスターの作成 { #create-digitalocean-cluster }

[DigitalOcean クラスターの作成](../user-guide/cluster-management.md#creating-digitalocean-cluster)の手順を参照してください。

### クラウド認証情報の作成 { #create-cloud-credential }

`Cloud Credentials` ページで DigitalOcean のクラウド認証情報を作成します。

![クラウド認証情報の作成](../assets/tutorials/adding-gpucluster-using-digitalocean/create-cloud-credential.png)

### クラウド認証情報によるクラスターの作成 { #create-cluster-with-cloud-credential }

クラスター作成ページで、`Cloud Provider` に `DigitalOcean` を選択します。

![クラウドプロバイダー](../assets/tutorials/adding-gpucluster-using-digitalocean/select-cloud-provider.png)

名前を入力し、作成したクラウド認証情報と、GPU Droplet に対応するリージョンを選択します。

![リージョンの選択](../assets/tutorials/adding-gpucluster-using-digitalocean/select-region.png)

`Next` をクリックして、クラスターのワーカープールを作成します。

### ワーカープールの作成 { #create-worker-pool }

必要に応じて `Name`、`Replicas`、`Batch Size` を入力し、ワーカープールの `Instance Type`（DigitalOcean の Droplet サイズ）を選択します。

![インスタンスタイプの選択](../assets/tutorials/adding-gpucluster-using-digitalocean/select-instance-type.png)

次に `OS Image` を選択します。

![OS イメージの選択](../assets/tutorials/adding-gpucluster-using-digitalocean/select-os-image.png)

AMD GPU Droplet は利用できないことが多いため、現在は NVIDIA 系 GPU Droplet のみ対応します。OS は Debian 系のみ対応します。

- Nvidia AI/ML Ready（**推奨**）は Ubuntu 22.04 ベースで、ドライバー、CUDA、container-toolkit が導入済みのため、追加パッケージは不要です。
- Debian ディストリビューションでは、cloud-init により `nvidia-open` ドライバー、CUDA 12.8、nvidia-container-toolkit 1.17.8-1 を導入します。
- Ubuntu ディストリビューションでは、cloud-init により `nvidia-driver-570`、CUDA 12.8、nvidia-container-toolkit 1.17.8-1 を導入します。

ワーカープールではラベルとボリュームを設定できます。Droplet に作成・接続したボリュームはワーカーコンテナーへ自動マウントされません。必要に応じて `/opt/gpustack-run-worker.sh` を変更してマウントします。

設定が完了したら `Save` をクリックします。

## ワーカーのプロビジョニングの待機 { #waiting-for-workers-to-be-provisioned }

クラスター保存後、`Workers` ページで DigitalOcean ワーカーの進行状況を確認します。

![ワーカーのプロビジョニング](../assets/tutorials/adding-gpucluster-using-digitalocean/provisioning-workers.png)

プロビジョニングは、次の段階で行われます。

1. SSH キーを作成します。
2. SSH キーを使って Droplet を作成します。
3. Droplet の起動を待ちます。
4. パブリック IP の割り当てを待ちます。
5. ボリュームを作成し、Droplet に接続します。
6. ワーカーが `Initialized` になり、コンテナーの起動とサーバーへの接続を待ちます。

ワーカーが `Ready` になれば、モデルをデプロイできます。

![プロビジョニング済みワーカー](../assets/tutorials/adding-gpucluster-using-digitalocean/worker-provisioned.png)

## DigitalOcean ワーカーの安全なスケールダウン { #safely-scale-down-digitalocean-workers }

DigitalOcean ワーカーが不要になったら、次の手順で安全に Droplet を削除します。

- 削除するワーカー数に合わせて、ワーカープールのレプリカ数を調整します。この操作だけでワーカーは自動削除されません。
- 削除するワーカーにモデルインスタンスがデプロイされていないことを確認します。
- 必要なワーカーを削除します。対応する Droplet も削除されます。
