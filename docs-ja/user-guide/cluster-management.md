# クラスター管理 { #cluster-management }

GPUStack は、クラスター単位のワーカー管理と複数のクラスタータイプをサポートしています。`DigitalOcean` などの `Cloud Provider` でクラスターを作成するほか、セルフホストのクラスターを作成して `Docker` の実行コマンドでワーカーを追加できます。また、セルフホストの `Kubernetes` クラスターにあるすべてのノードを GPUStack ワーカーとして登録することもできます。

## クラスターの作成 { #create-cluster }

1. `Clusters` ページに移動します。
2. `Add Cluster` ボタンをクリックします。
3. クラスタープロバイダーを選択します。`Self-Host` プロバイダーでは `Docker` と `Kubernetes`、`Cloud Provider` では `DigitalOcean` と `SHUIHUA FUTURE` を選択できます。
4. プロバイダーに応じて、`Base Configuration` と `Add Worker` の各ステップで異なるオプションを設定します。
5. 基本設定の `Advanced` クラスター設定では、`Worker Configuration YAML` を使ってワーカーのオプションを事前設定できます。

### Docker クラスターの作成 { #create-docker-cluster }

1. `Basic Configuration` ステップでは、`Name` は必須、`Description` は任意です。
2. `Save` をクリックします。
3. `Add Worker` ステップでは、`docker run` コマンドでワーカーを追加する前に、いくつかのオプション設定と検証が必要です。
4. `Select the GPU vendor` で GPU ベンダーを選択します。検証済みのベンダーは `Nvidia`、`AMD`、`Ascend` です。実験的な対応ベンダーは `Hygon`、`Moore Threads`、`Iluvatar`、`Cambricon`、`Metax` です。選択後に `Next` をクリックします。
5. `Check Environment` で環境を確認します。ワーカーを追加する準備が整っているかを検証するシェルコマンドが表示されます。スクリプトをコピーして対象環境で実行し、OK が返されたら `Next` をクリックします。
6. `Specify arguments` で追加するワーカーの引数を指定します。次の引数を設定し、`Next` をクリックします。
   - `Specify the worker IP` でワーカー IP を指定するか、`Auto-detect the Worker IP` で自動検出させます。ワーカー IP にサーバーからアクセスできることを確認してください。
   - ワーカーコンテナーの `Additional Volume Mount` を指定します。マウント先のパスは、既存のモデルファイルの再利用に使用できます。

7. `Run command` でワーカーコンテナーを作成し、起動します。bash スクリプトをコピーして対象環境で実行します。

クラスター作成後にワーカーを追加することもできます。

1. `Clusters` ページに移動します。
2. ワーカーを追加するクラスターを探します。
3. 操作列の省略記号ボタンをクリックし、`Add Worker` を選択します。
4. ワーカーを追加するためのオプションを選択します。上記の `Select the GPU vendor` から `Run command` までと同じ手順に従います。

### Kubernetes クラスターの登録 { #register-kubernetes-cluster }

1. `Basic Configuration` ステップでは、`Name` は必須、`Description` は任意です。
2. `Save` をクリックします。
3. `Select the GPU vendor` で GPU ベンダーを選択します。検証済みのベンダーは `Nvidia`、`AMD`、`Ascend` です。実験的な対応ベンダーは `Hygon`、`Moore Threads`、`Iluvatar`、`Cambricon`、`Metax` です。`Kubernetes` クラスターでは複数の GPU ベンダーを選択できます。CPU 専用クラスターでは何も選択しません。選択したベンダーごとに 1 つのワーカー DaemonSet が生成され、各 DaemonSet の `nodeSelector` はマニフェスト生成時にそのベンダーの PCI 存在ラベルから導出されます。選択後に `Next` をクリックします。
4. `Check environment` で環境を確認します。ワーカーを追加する準備が整っているかを検証するシェルコマンドが表示されます。スクリプトをコピーして対象環境で実行し、OK が返されたら `Next` をクリックします。
5. `Run command` でワーカーのマニフェストを適用します。bash スクリプトをコピーし、`kubectl` がインストールされ、`kubeconfig` が設定された環境で実行します。

クラスター作成後に Kubernetes を登録することもできます。

1. `Clusters` ページに移動します。
2. Kubernetes クラスターを登録する対象のクラスターを探します。
3. 操作列の省略記号ボタンをクリックし、`Register Cluster` を選択します。
4. クラスターを登録するためのオプションを選択します。上記の `Select the GPU vendor` から `Run command` までと同じ手順に従います。

#### Kubernetes クラスターのオプション { #kubernetes-cluster-options }

`Kubernetes` クラスターの作成時または編集時には、`Basic Configuration` ステップで、`Name` と `Description` に加えて次のオプションを設定できます。

- `Cluster Type`（必須）— クラスターの用途を選択します。
    - `Model Service` — モデル API やトークンベースのサービスの公開など、LLM 推論と API サービングに使用します。
    - `GPU Service` — 対話型の開発、学習ジョブ、カスタム環境など、オンデマンドの GPU 計算に使用します。

`Advanced` 設定では、次の Kubernetes デプロイオプションを設定できます。

- `Namespace` — クラスターのマニフェストの生成先となる Kubernetes 名前空間です。空の場合は `gpustack-system` を使用します。
- `Volume Mounts` — すべてのワーカーポッドにマウントする追加ボリュームです。各マウントに `Volume Name`、`Container Path`、`Read Only` フラグを指定し、`Source Type` を選択します。
    - `Host Path` — ノード上のパスです。`Path Type`（例: `Directory`、`Directory (create if not exists)`、`File`、`Socket`、`Character Device`、`Block Device`）を指定します。
    - `Persistent Volume Claim (PVC)` — 既存の `PVC Name` です。読み取り専用に設定することもできます。
    - `ConfigMap` — `ConfigMap Name` です。存在を任意とする設定もできます。
- `Image Credentials` — プライベートレジストリーから GPUStack イメージを取得する際に使うイメージプルシークレットです。エントリーごとに `Registry`、`Username`、`Password` を指定します。
- `Node Selector` — このクラスターの GPUStack ワークロードを、ラベルが一致するノードに限定します。チャートの `global.nodeSelector` として渡され、各コンポーネントがフォールバック先として参照するため、ワーカー DaemonSet とオペレーターに適用されます。また、この設定に対応するオペレーターリリースでは、オペレーターがデプロイするコンポーネント（Kueue、Node Feature Discovery のコントロールプレーン、CSI コントローラー）にも適用されます。意図的に対象外となるワークロードが 2 種類あります。1 つは、対象ノード全体をカバーする必要がある Node Feature Discovery のラベラーと CSI ノードプラグインです。これらを限定すると、ほかのすべてのノードにラベルが付かなくなり、ワーカー自身が選択条件に使うラベルも失われます。もう 1 つはチャートのインストールフックです。リリースが存在する前に実行されるため、リリース自身が付けるラベルを待つことはできません。ワーカー DaemonSet は、これらのラベルに加えて GPU ベンダーラベルも保持します。ノードは両方に一致する必要があります。
- `Default Container Registry` — このクラスターの GPUStack イメージを解決する際に使うデフォルトのレジストリーです。未設定の場合はサーバーのデフォルトにフォールバックします（プレースホルダーは `docker.io`）。
- `Operator Image` — GPUStack Operator のコンテナーイメージを上書きします。空の場合はサーバーのデフォルトを使用します。
- `GPU Service Static Access Address` — `Cluster Type` が `GPU Service` の場合のみ表示されます。このクラスターの GPU インスタンスにアクセスするためにオペレーターが使用する静的アドレス（例: LoadBalancer の VIP）です。省略できます。
- `Worker Configuration YAML` — 以下の[ワーカー設定 YAML](#worker-configuration-yaml) を参照してください。

#### 名前空間と Pod Security Admission { #namespaces-and-pod-security-admission }

GPUStack は、自身が管理する名前空間として `gpustack-system`（または `Namespace` の設定値）と、クラスターを使用する組織ごとの `gpustack-<org>` を作成し、それぞれに [Pod Security Admission](https://kubernetes.io/docs/concepts/security/pod-security-admission/) の `privileged` レベルのラベルを付けます。

```yaml
pod-security.kubernetes.io/enforce: privileged
pod-security.kubernetes.io/audit: privileged
pod-security.kubernetes.io/warn: privileged
```

**必要な理由。** Pod Security Admission は Kubernetes に組み込まれており、Pod の*作成時*に適用されます。レベルに適合しない Pod は Pending のまま残るのではなく、即座に拒否されます。モデル、キャッシュサービス、ベンチマークの Pod には、`baseline` と `restricted` レベルで禁止されている次の機能が必要です。

| 要件 | 必要となる理由 |
| --- | --- |
| ホストネットワーク | RDMA が GID を NIC アドレスにバインドするため |
| hostPort | 分離されたペアが相手に渡すサイドチャネルのため |
| ホスト IPC | KV キャッシュバッファーを CUDA-IPC で共有するため |
| デバイスのマウント | アクセラレーターそのものを利用するため |

プリフィル/デコードのデプロイには 4 つすべてが同時に必要なため、このラベルがないと最も早く、深刻に失敗する構成です。

**3 つのキーをすべて設定する理由。** `warn` と `audit` をクラスターのデフォルトのままにすると、Pod を作成するたびに警告が返され、監査アノテーションが記録されます。例外を適用することが想定された名前空間ではノイズとなり、ほかの場所で重要な警告が埋もれてしまいます。

**影響範囲。** `privileged` は、その名前空間内のすべての Pod が Pod Security Admission の適用を免除されることを意味します。このため、ユーザー自身のワークロードと共有する名前空間ではなく、GPUStack が作成して管理する名前空間を使用しています。これらの名前空間には、ほかのワークロードをスケジュールしないでください。また、これらのラベルは Kyverno、Gatekeeper、OPA には影響しません。これらは別の Admission Webhook であり、ホストネットワークやデバイスマウントを禁止するポリシーがあれば、引き続き Pod は拒否されます。クラスターでこれらを使用している場合は、そちらでも GPUStack の名前空間を許可してください。

**より厳しいレベルで拒否される内容の確認。** 実行中のワークロードに対して、何も変更せずに確認できます。

```bash
kubectl label --dry-run=server --overwrite ns <namespace> \
  pod-security.kubernetes.io/enforce=baseline
```

API サーバーは、名前空間内の既存の Pod を指定したレベルで評価し、拒否される各 Pod と、違反する制御項目を警告します。

#### チャートの値 { #chart-values }

`Kubernetes` クラスターを登録すると、GPUStack Helm チャートがインストールされ、上記のオプションはそのチャートの値に変換されます。これらのオプションで扱えない設定には、同じチャートに直接渡せる `helmValues` を使用します。キーはチャート固有のものをそのまま使用するため、チャートやサブチャートが提供する設定は、この画面に専用のオプションが追加されるのを待たずに設定できます。

UI にはまだ入力欄がないため、API を通じて設定します。`PUT` はクラスター全体を置き換えるので、まずクラスターを読み取り、`k8sOptions` の下に `helmValues` を追加して返送してください。

```bash
# Read the cluster.
curl -sS -H "Authorization: Bearer <api-key>" \
  http://<server>/v2/clusters/<id> > cluster.json

# Add the values. This one keeps the chart from installing the S3 CSI driver,
# for a cluster that already provides its own storage.
jq '.k8sOptions.helmValues = {
      "gpustack-operator": {"csi-driver-s3": {"enabled": false}}
    }' cluster.json > updated.json

# Send it back.
curl -sS -X PUT -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" --data @updated.json \
  http://<server>/v2/clusters/<id>
```

値は、クラスターから導出された値に対して、キーごとに深さ優先でマージされます。リストは Helm 自体と同様に、追加ではなく置き換えとなります。キーの説明は各チャートにあります。GPUStack チャートの `values.yaml`、および `gpustack-operator` 配下の設定については [GPUStack Operator チャート](https://github.com/gpustack/gpustack-operator)を参照してください。

一部のパスはマージされず、拒否されます。これらは、このクラスターの登録情報からサーバーが導出する設定で、リリースが発行先のクラスターと一致することを保証します。具体的には、ワーカーの報告先、実行するイメージ、そのイメージの取得元レジストリー、トークンを保持する Secret などです。API は拒否したパスを返し、`helmValues` フィールドの説明には現在の一覧が記載されています。このページで重複して記載しないことで、情報の不一致を防いでいます。

!!! note

    クラスターを保存するだけでは、Kubernetes 側は変更されません。`Register Cluster` を再実行して、生成されたマニフェストを適用してください。クラスター内の Job が、マニフェストの要求とリリースのインストール元設定を比較し、異なる場合にのみアップグレードします。

!!! note

    リリースがデプロイする構成を変えるアップグレードでは、生成対象から外れたワーカー DaemonSet が自動的に削除されます。たとえば、CPU 専用ノードがコントロールプレーンを担うクラスターで CPU ワーカー DaemonSet を無効にする場合（`helmValues` で `worker.cpuEnabled=false` を指定）、またはクラスターが選択しなくなった GPU ベンダーを削除する場合です。`kubectl apply` はマニフェストから除外されたオブジェクトを削除しないため、アップグレード Job が、リリースで生成されなくなったすべての `<release>-worker[-<vendor>]` DaemonSet を削除します。過去の Helm リビジョンがそれを管理していたかどうかは問いません。

    このクリーンアップ機能が導入される前から残っているものも、同じ方法で削除されます。現在のマニフェストを一度適用すると、前回から設定が変わっていなくても Job が削除します。Job を実行できないクラスターに限り、次の手動操作が必要です。

    ```bash
    kubectl delete daemonset/gpustack-worker -n gpustack-system
    kubectl delete daemonset/gpustack-worker-<vendor> -n gpustack-system
    ```

    バージョンだけをアップグレードする場合は、`Register Cluster` のパラメーター（GPU ベンダー、オプション）を、最後に適用したマニフェストと同じにしてください。それ以外の変更は、バージョンアップグレードではなく構成変更です。

!!! warning

    Kueue と Node Feature Discovery は必須です。オペレーターはこれらに基づいてスケジューリングの仕組みを構成し、起動時に CRD を待つため、クラスターに既存のものがない状態で無効にすると、オペレーターが起動できなくなります。このリリースがインストールしたものを無効にすると、それ自体が削除されます。Kueue の CRD はそのチャートに含まれ、すべての `Workload` と `ClusterQueue` も一緒に削除されます。

#### Kubernetes クラスターのアンインストール { #uninstalling-a-kubernetes-cluster }

アンインストールでは、登録時にインストールされたリリースの `helm uninstall` に加えて、Helm が管理していないオブジェクトを削除します。正常に完了するかは、事前準備によって決まります。オペレーターの配下（Kueue、Node Feature Discovery、CSI ドライバー、デバイスマネージャー）には、カスタムリソースにファイナライザーを付けるコントローラーがあります。アンインストールするとそのコントローラーが削除されるため、先にリソースを削除しなければ、リソースがある名前空間（`gpustack-system`、`gpustack-default`）が `Terminating` のままになります。次の順に実行してください。

1. まず、このクラスターで実行するものをサーバーから削除します。デプロイ済みのモデルを削除（または移行）してから、クラスターのレコードを削除し、消失するワーカーへのスケジューリングをサーバーが停止するようにします。
2. **コントローラーがまだ動作している間に**カスタムリソースを削除し、削除の完了を待ちます。推測せず、このクラスターのリリースが管理するものを調べてください。

    ```bash
    kubectl get crd -o json | jq -r '.items[]
      | select(((.metadata.annotations // {})["meta.helm.sh/release-name"] // "")
        | . == "gpustack" or . == "gpustack-kueue"
          or . == "gpustack-node-feature-discovery"
          or . == "gpustack-csi-driver-nfs" or . == "gpustack-csi-driver-s3"
          or . == "gpustack-operator-device-manager")
      | .metadata.name'
    ```

    表示されたすべての CRD について、すべてのオブジェクトを削除し、残りがなくなるまで待ちます。

    ```bash
    kubectl delete <crd-name> --all -A          # e.g. clusterqueues.kueue.x-k8s.io
    kubectl get <crd-name> -A                   # empty before you move on
    ```

3. リリースをアンインストールします。メインのリリースは `gpustack` です。5 つの `gpustack-*` アプリケーションリリースは、チャートがそれらのオブジェクトを引き継ぐ前に登録されたクラスターに存在します。存在しないリリースに `helm uninstall` を実行するとエラーになるため、先に確認します。

    ```bash
    for release in gpustack gpustack-kueue gpustack-node-feature-discovery \
                   gpustack-csi-driver-nfs gpustack-csi-driver-s3 \
                   gpustack-operator-device-manager; do
      kubectl get secret -n gpustack-system \
        --selector "owner=helm,name=${release}" >/dev/null 2>&1 || continue
      helm uninstall "${release}" -n gpustack-system --timeout 10m
    done
    ```

    ここでは `--wait` を渡さないでください。`--wait` を指定すると、アンインストールは Node Feature Discovery の **post-delete prune フック**まで進み、フックの `node-feature-discovery-prune` ServiceAccount と ClusterRole を作成した後、削除待ちで停止します。この待機は `--timeout` を超えて継続します（`helm --debug` で確認すると、タイムアウト後も数分間 `waiting for resources to be deleted count=1` に留まり、作成された ServiceAccount が残っています）。抜け出すために Helm を終了すると prune RBAC が管理者のいない状態で残るため、ステップ 4 でこれらを手動削除する必要があります。`--wait` を指定しなければアンインストールは速やかに戻り、フックのリソースもほかと一緒にクリーンアップされ、オブジェクトは同様に削除されます。ステップ 4 では、Helm が管理していなかった残存物を削除します。

4. `helm uninstall` で削除されないものを削除します。Helm は CRD を削除しません。また、オペレーターは、どのリリースにも管理されない 3 種類のクラスター単位のオブジェクトを登録します。

    - **オペレーター自身の CRD** — `worker.gpustack.ai` 配下の `devices`、`instances`、`instancetypes` です。チャートではなくオペレーターが起動時に作成するため、`meta.helm.sh/release-name` アノテーションがなく、ステップ 2 の検出クエリーにも表示されません。`instancetype` オブジェクトには `gpustack.ai/controlled` ファイナライザーが残ります。CRD の削除前または削除中にこれを除去しないと（下記参照）、CRD が削除中のままになり、グループの API が消えません。
    - **オペレーターの Admission Webhook** — `gpustack-worker-mutation`（Mutating）と `gpustack-worker-validation`（Validating）です。オペレーターの Service に転送し、失敗時には拒否するため、削除するまで `worker.gpustack.ai` オブジェクトへのすべての書き込み（下記のファイナライザー除去も含む）が `service "gpustack-operator-worker" not found` で拒否されます。
    - **オペレーターの APIServices** — `v1.gpustack.ai` と `v1.worker.gpustack.ai` です。これらもオペレーターの Service を参照します。削除するまで、削除中の名前空間はこれらのグループについて `stale GroupVersion discovery` を報告し、完了しません。

    ```bash
    kubectl delete mutatingwebhookconfiguration gpustack-worker-mutation --ignore-not-found
    kubectl delete validatingwebhookconfiguration gpustack-worker-validation --ignore-not-found
    kubectl delete apiservice v1.gpustack.ai v1.worker.gpustack.ai --ignore-not-found
    kubectl delete crd devices.worker.gpustack.ai instances.worker.gpustack.ai \
      instancetypes.worker.gpustack.ai --ignore-not-found
    ```

    Node Feature Discovery の prune RBAC も、`helm uninstall` の後に残ることがあります。`ClusterRole/node-feature-discovery-prune` とそのバインディングは post-delete prune フックのリソースで（上記の `--wait` に関する注意を参照）、フック実行中に Helm を終了すると管理者のいない状態で残ります。その後の登録では引き継がれず、インストールが拒否されるため、ほかと一緒に削除してください。

    ```bash
    kubectl delete clusterrole node-feature-discovery-prune --ignore-not-found
    kubectl delete clusterrolebinding node-feature-discovery-prune --ignore-not-found
    ```

    次にクラスター単位のブートストラップ RBAC、その後に名前空間を削除します。`gpustack-system` にはブートストラップの ConfigMap、ServiceAccount、登録トークンの Secret も含まれ、名前空間の削除とともに削除されます。

    ```bash
    kubectl delete clusterrolebinding gpustack-bootstrap --ignore-not-found
    kubectl delete namespace gpustack-default gpustack-system
    ```

5. *（任意）* 各ノードに残るデータを削除します。ワーカーポッドはホストの `/var/lib/gpustack` をマウントしており、上記の手順ではこのディレクトリーを変更しません。ここにはダウンロード済みのモデルの重みやキャッシュが保存されています。再インストール時に問題になることはありません。同じノードにクラスターを再登録すると、このディレクトリーが再利用され、既存のファイルの再ダウンロードを省略できます。ストレージをほかの用途に必要とする場合にのみ削除してください。

    ```bash
    rm -rf /var/lib/gpustack
    ```

`kubectl get all -n gpustack-system`（`NotFound` になること）、`kubectl get clusterrolebinding | grep gpustack`、`kubectl get ds -A | grep -E 'gpustack|csi'` で確認します。ステップ 4 でのみ削除するオブジェクトについては、`kubectl get apiservice,mutatingwebhookconfiguration,validatingwebhookconfiguration 2>/dev/null | grep gpustack` と `kubectl get crd | grep -E 'gpustack|worker\.gpustack'` でも確認します（どちらも何も表示されないこと）。

名前空間がすでに `Terminating` のままの場合は、コントローラーが消えたオブジェクトにファイナライザーが残っています。対象を特定してから、ファイナライザーを手動で除去してください。この操作が安全なのは、削除するクラスターだけです。先に Admission Webhook を削除してください（ステップ 4）。そうしないと、Service が存在しなくなった Webhook がパッチ自体を捕捉します。

```bash
kubectl api-resources --verbs=list --namespaced -o name |
  xargs -n1 -I{} sh -c 'kubectl get {} -n gpustack-default -o name 2>/dev/null'
kubectl patch <kind>/<name> -n gpustack-default \
  -p '{"metadata":{"finalizers":null}}' --type=merge
```

クラスター単位の残存物も同じように除去します。`gpustack.ai/controlled` ファイナライザーを保持する `instancetype.worker.gpustack.ai` オブジェクトは、その CRD（および `worker.gpustack.ai` グループ全体）を削除中の状態に留めます。`kubectl patch instancetypes.worker.gpustack.ai <name> -p '{"metadata":{"finalizers":null}}' --type=merge` を実行してください。

### DigitalOcean クラスターの作成 { #creating-digitalocean-cluster }

1. `Basic Configuration` ステップでは、`Name` は必須、`Description` は任意です。DigitalOcean API との通信に使用するクラウド認証情報を作成または選択します。GPU Droplet をサポートするリージョンを選択します。新しく作成する DigitalOcean Droplet からアクセス可能な `GPUStack Server URL` も設定する必要があります。
2. `Next` をクリックします。
3. 1 つ以上の `Worker Pools` を追加します。プールごとに `Name`、`Instance Type`、`OS Image`、`Replicas`、`Batch Size`、`Labels`、`Volumes` を指定できます。
4. ワーカープールの設定後に `Save` をクリックします。

クラスター作成後にワーカープールを追加することもできます。

1. `Clusters` ページに移動します。
2. ワーカープールを追加する `DigitalOcean` クラスターを探します。
3. 操作列の省略記号ボタンをクリックし、`Add Worker Pool` を選択します。
4. 上記のステップ 3 のオプションを設定して、新しいワーカープールを追加します。

### SHUIHUA FUTURE クラスターの作成 { #creating-shuihua-future-cluster }

1. `Basic Configuration` ステップでは、`Name` は必須、`Description` は任意です。Shuihua API との通信に使用するクラウド認証情報を作成または選択します。Shuihua にはリージョンがないため、選択は不要です。新しく作成するインスタンスからアクセス可能な `GPUStack Server URL` も設定する必要があります。
2. このプロバイダーでは `Default Container Registry` が必須です。Shuihua インスタンスは Docker Hub にアクセスできないため、Docker Hub に解決されるレジストリーは拒否されます。入力欄には `quay.io` と `swr.cn-south-1.myhuaweicloud.com` が候補として表示されます。独自の Harbor やミラーなど、アクセス可能なほかのレジストリーも指定できます。
3. `Next` をクリックします。
4. 1 つ以上の `Worker Pools` を追加します。プールごとに `Name`、`Instance Type`、`OS Image`、`Replicas`、`Batch Size`、`Labels` を指定できます。`Instance Type` は Shuihua の仕様テンプレートで、GPU モデル、時間単価、残り在庫とともに表示されます。売り切れのテンプレートは選択できません。Shuihua にはブロックストレージ API がないため、ボリュームは提供されません。
5. ワーカープールの設定後に `Save` をクリックします。

クラスター作成後も、DigitalOcean クラスターと同じ方法でワーカープールを追加できます。

Shuihua インスタンスは NAT 配下にあり、ポート 22 と 80 だけがマッピングされています。そのため、クラスターはデフォルトで `tunnel` プロキシモードを使用し、ワーカーはサーバーの WebSocket トンネルを通じて推論を提供します。表示される IP はインスタンスのプライベートアドレスです。実際にアクセスできる接続先は、ワーカーの `View SSH Access` で確認してください。手順全体は [Shuihua を使用した GPU クラスターの追加](../tutorials/adding-gpucluster-using-shuihua.md)を参照してください。

### ワーカープールの操作 { #operating-worker-pools }

クラウドプロバイダーのクラスターでは、`Clusters` ページでワーカープールを管理できます。

1. `Clusters` ページに移動します。
2. 管理するクラウドプロバイダーのクラスターを探し、展開してワーカープールを表示します。
3. ワーカーのレプリカ数を変更するには、ワーカー列で直接編集します。
4. ワーカープールを編集するには、`Edit` ボタンをクリックし、必要に応じて `Name`、`Replica`、`Batch Size`、`Labels` を更新します。
5. ワーカープールを削除するには、そのワーカープールの操作列にある省略記号ボタンをクリックし、`Delete` を選択します。

## クラスターの更新 { #update-cluster }

1. `Clusters` ページに移動します。
2. 編集するクラスターを探します。
3. `Edit` ボタンをクリックします。
4. 必要に応じて `Name`、`Description`、`Worker Configuration YAML` を更新します。
5. `Save` ボタンをクリックします。

## クラスターの削除 { #delete-cluster }

1. `Clusters` ページに移動します。
2. 削除するクラスターを探します。
3. 操作列の省略記号ボタンをクリックし、`Delete` を選択します。
4. 削除を確認します。
5. モデルまたはワーカーが残っているクラスターは削除できません。

## ワーカー設定 YAML { #worker-configuration-yaml }

クラスターの作成時または更新時には、次の YAML の例を使ってワーカー設定を事前定義できます。

```yaml
# ========= log level & tools ===========
debug: false
tools_download_base_url: https://mirror.your_company.com
# ========= directories ===========
pipx_path: "/usr/local/bin/pipx"
cache_dir: "/var/lib/gpustack/cache"
log_dir: "/var/lib/gpustack/log"
bin_dir: "/var/lib/gpustack/bin"
# ========= container & image ===========
image_name_override: "gpustack/gpustack:main"
image_repo: "gpustack/gpustack"
# ========= service & networking ===========
worker_ifname: en0
worker_port: 10150
worker_metrics_port: 10151
disable_worker_metrics: false
service_port_range: "40000-40063"
ray_port_range: "41000-41999"
proxy_mode: worker
# ========= system reserved resources ===========
system_reserved:
  ram: 0
  vram: 0
# ========= huggingface ===========
huggingface_token: xxxxxx
enable_hf_transfer: false
# ========= TLS ===========
insecure_tls: false
```

上記の YAML は、`Worker Configuration YAML` で現在サポートしているすべてのオプションを示しています。各オプションの意味は、GPUStack の[設定ファイルのドキュメント](../cli-reference/start.md#config-file)を参照してください。`proxy_mode` オプションは、サーバーからワーカーへの接続方法を制御します。[ワーカー接続モード](#worker-connection-modes)を参照してください。

`Docker` クラスターと `Kubernetes` クラスターのどちらも、デフォルトのコンテナーレジストリーをここで設定しなくなりました。代わりに、`Advanced` 設定にあるクラスター単位の `Default Container Registry` オプションで設定してください。`Kubernetes` クラスターの名前空間も同様に、`Namespace` オプションで設定します（[Kubernetes クラスターのオプション](#kubernetes-cluster-options)を参照）。

## ワーカー接続モード { #worker-connection-modes }

ワーカー上のモデルインスタンスに推論リクエストを転送するには、サーバーと API ゲートウェイからそのワーカーへのネットワーク経路が必要です。ワーカーの `proxy_mode` オプションを、[ワーカー設定 YAML](#worker-configuration-yaml) または `--proxy-mode` フラグで設定すると、次の経路を選択できます。

- `direct` — サーバーとゲートウェイが、ワーカーの通知したアドレスとポートに直接接続します。オーバーヘッドが最も少ない方式ですが、サーバーからワーカーに直接到達できる必要があります。
- `worker` — リクエストは、ワーカー内蔵の HTTP リバースプロキシを経由し、ローカルの推論プロセスに転送されます。引き続きサーバーからワーカーに到達できる必要がありますが、公開するのはワーカーポートだけで済みます。
- `tunnel` — ワーカーはサーバーへの 1 本の**アウトバウンド** WebSocket 接続を維持し、サーバーはこのトンネルだけを通じてワーカーにアクセスします。ワーカーがサーバーからのインバウンド接続を受け付けられない場合に使用します。

デフォルトは、サーバー内の組み込みワーカーでは `direct`、スタンドアロンワーカーでは `worker` です。

### トンネルモード（ファイアウォールまたは NAT 配下のワーカー） { #tunnel-mode-worker-behind-a-firewall-or-nat }

`direct` と `worker` モードではサーバーが接続を開始するため、サーバーからワーカーに到達できる必要があります。しかし、ワーカーがファイアウォールや NAT の背後にある場合、別のネットワークにある場合、アウトバウンド通信のみ許可するプライベートサブネットにある場合など、到達できないことがあります。

`tunnel` モードでは向きが逆になり、ワーカーからサーバーへの**一方向**に接続を確立します。

1. ワーカーが、登録にも使う同じ `--server-url` エンドポイントに対して、永続的なアウトバウンド WebSocket 接続を開きます。トークンで認証し、切断された場合は自動的に再接続します。
2. サーバーは、プロキシポート（デフォルトは `30079`、`--proxy-port` で設定）で HTTP/HTTPS プロキシを実行します。
3. ゲートウェイがトンネルモードのワーカー上のモデルインスタンスにアクセスする際は、リクエストをサーバーのプロキシポートに送り、既存のトンネルを通じてワーカーに中継します。

ワーカーは外向きの接続だけを行うため、ワーカー側でインバウンドポートを開く必要はありません。サーバーの API ポートに到達できればよく、サーバーのプロキシポートにはゲートウェイから到達できる必要があります。

有効にするには、ワーカーの `proxy_mode` を `tunnel` に設定します。

```yaml
proxy_mode: tunnel
```
