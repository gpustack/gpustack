---
translation_status: translated
---

# Shuihua を使った GPU クラスターの追加 { #adding-a-gpu-cluster-using-shuihua }

クラスターの作成時に、GPUStack は SHUIHUA FUTURE でワーカーを作成し、クラスターに追加できます。

## 事前準備 { #preparation }

SHUIHUA FUTURE のアカウントと、[Shuihua コンソール](https://hub.do.top/cn/signin)で作成した API キーが必要です。キーは `amp_live_` で始まります。

![Shuihua API キー](../assets/tutorials/adding-gpucluster-using-shuihua/shuihua-api-key.png)

GPUStack サーバーの起動時に `--server-external-url` を指定します。Shuihua インスタンスはこのアドレスで登録し、トンネル接続を維持するため、そこから到達可能でなければなりません。プロキシの背後にある場合は、そのアドレスを指定します。

## Shuihua クラスターの作成 { #create-shuihua-cluster }

### クラウド認証情報の作成 { #create-cloud-credential }

`Cloud Credentials` ページで `SHUIHUA FUTURE` のクラウド認証情報を作成します。入力するのは `API Key` だけです。

![クラウド認証情報の作成](../assets/tutorials/adding-gpucluster-using-shuihua/create-cloud-credential.png)

### クラウド認証情報によるクラスターの作成 { #create-cluster-with-cloud-credential }

クラスター作成ページで、`Cloud Provider` に `SHUIHUA FUTURE` を選択します。

![クラウドプロバイダーの選択](../assets/tutorials/adding-gpucluster-using-shuihua/select-cloud-provider.png)

名前、作成済みのクラウド認証情報、`GPUStack Server URL` を設定します。Shuihua にはリージョンがないため、選択は不要です。

ここでは `Default Container Registry` が必須です。Shuihua インスタンスは Docker Hub に接続できないため、別のレジストリからワーカーイメージを取得する必要があります。候補は `quay.io` と `swr.cn-south-1.myhuaweicloud.com` ですが、独自の Harbor やミラーなども入力できます。Docker Hub のホストに解決される値は拒否されます。

![基本設定](../assets/tutorials/adding-gpucluster-using-shuihua/basic-configuration.png)

> 注意：サーバーの `--image-name-override` はこのフィールドより優先します。上書きがある場合、そのイメージ名をそのまま使用し、レジストリ設定は無効になるため、上書き先自体に到達可能なレジストリを指定してください。

`Next` をクリックして、クラスターのワーカープールを作成します。

### ワーカープールの作成 { #create-worker-pool }

必要に応じて `Name`、`Replicas`、`Batch Size` を入力します。

![ワーカープール設定](../assets/tutorials/adding-gpucluster-using-shuihua/worker-pool-configuration.png)

`Instance Type`（Shuihua の spec template）を選択します。各候補には GPU モデル、時間単価、残数を表示します。売り切れのテンプレートも理由を確認できるよう一覧に残りますが、選択はできません。

![インスタンスタイプの選択](../assets/tutorials/adding-gpucluster-using-shuihua/select-instance-type.png)

次に `OS Image` を選択します。

Shuihua の貸し出しは NVIDIA GPU のみで、イメージにはドライバーとコンテナーツールキットが導入済みです。cloud-init は Docker の NVIDIA ランタイムを指定してワーカーを起動するだけで、インストールも再起動も行わないため、ドライバーを構築する環境より速く起動します。

Shuihua にはブロックストレージ API がないため、ボリュームには対応しません。ラベルは使用できます。

設定が完了したら `Save` をクリックします。

## ワーカーのプロビジョニングの待機 { #waiting-for-workers-to-be-provisioned }

クラスター保存後、`Workers` ページで Shuihua ワーカーの進行状況を確認します。

![ワーカーのプロビジョニング](../assets/tutorials/adding-gpucluster-using-shuihua/provisioning-workers.png)

プロビジョニングは、次の段階で行われます。

1. SSH キーを生成します。Shuihua に SSH キー API はないため、プロバイダーには登録せず、インスタンスの cloud-init に書き込みます。
2. インスタンスを作成します。非同期処理で、Shuihua は受理後 `creating`、`processing`、`active` の順に状態を報告します。
3. インスタンスの起動を待ちます。
4. アドレスの割り当てを待ちます。
5. ワーカーが `Initialized` になり、コンテナーの起動とサーバーへの接続を待ちます。

ワーカーが `Ready` になれば、モデルをデプロイできます。

![プロビジョニング済みワーカー](../assets/tutorials/adding-gpucluster-using-shuihua/worker-provisioned.png)

## SSH でのワーカーへの接続 { #connecting-to-a-worker-over-ssh }

Shuihua は全インスタンスを共有パブリック IP の背後に公開し、22 と 80 のみを個別のポートにマッピングします。ワーカーの表示 IP はインスタンス自身のプライベートアドレスであり、識別には使えますが接続できません。

ワーカーの操作メニューの `View SSH Access` を使ってください。実際に到達可能な `Host`、`Port`、`Connection Command` が表示され、`Private Key` を `worker-<id>-private_key.pem` としてダウンロードできます。

![ワーカーの SSH 情報](../assets/tutorials/adding-gpucluster-using-shuihua/worker-ssh-info.png)

ワーカー自身のポートはマッピングされないため、ゲートウェイから直接着信接続できません。推論はサーバーの WebSocket トンネル経由で提供するため、Shuihua クラスターはデフォルトで `tunnel` プロキシモードを使います。インスタンスへ直接ルーティングするネットワークでない限り、変更しないでください。

## Shuihua ワーカーの安全なスケールダウン { #safely-scale-down-shuihua-workers }

Shuihua ワーカーが不要になったら、次の手順で安全にインスタンスを削除します。

- 削除するワーカー数に合わせて、ワーカープールのレプリカ数を調整します。この操作だけでワーカーは自動削除されません。
- 削除するワーカーにモデルインスタンスがデプロイされていないことを確認します。
- 必要なワーカーを削除します。対応するインスタンスも終了されます。

## トラブルシューティング { #troubleshooting }

**ワーカーが `Initialized` のままで `Ready` にならない場合。** インスタンスは起動してもコンテナーが起動していません。多くの場合、イメージを取得できないことが原因です。SSH で接続して次を確認します。

```bash
sudo cloud-init status --long
sudo grep -iE "pull|denied|manifest" /var/log/cloud-init-output.log
docker ps -a
```

**ワーカーは `Ready` だが GPU を報告しない場合。** Docker が NVIDIA ランタイムを使用していません。

```bash
nvidia-smi
which nvidia-ctk
docker info | grep -i runtime
```

**プロビジョニングが失敗し、その場で再試行できない場合。** ワーカーを削除し、プールに代替を作成させてください。各作成要求はワーカー行に基づくリプレイ防止キーを持つため、新しいインスタンスには新しいワーカーが必要です。
