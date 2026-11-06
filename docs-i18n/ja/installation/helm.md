---
translation_status: translated
---

# Helm によるインストール { #installation-via-helm }

v2.2.0 以降、公式 Helm チャートを使用し、クラスター内の [Higress](https://higress.io/) API ゲートウェイとともに Kubernetes 上へ GPUStack をデプロイできます。

!!! note

    Higress を使用した Kubernetes 上での GPUStack のデプロイは、現在**実験的機能**です。実行前に[制限事項](#limitations)を確認してください。

## 前提条件 { #prerequisites }

| コンポーネント | バージョン |
| ---------- | ---------- |
| Helm       | >= v3.18.4 |
| Kubernetes | >= v1.30.0 |
| GPUStack   | >= v2.2.0  |

さらに、次の条件が必要です。

- サーバーのデータボリューム用に、クラスターでデフォルトの `StorageClass` を設定します（k3s のデフォルトは `local-path`）。または、`server.dataVolume.hostPath` を指定し、PVC の代わりにホストパスボリュームを使用します。
- GPU ワーカー用のノードに、適切な GPU ドライバーとコンテナーツールキットがあることを確認します（[インストール要件](./requirements.md)を参照）。GPU ノードにラベルを付けるため、[Node Feature Discovery（NFD）](https://kubernetes-sigs.github.io/node-feature-discovery/) も必要です。

## 制限事項 { #limitations }

- GPUStack サーバーは `StatefulSet` としてデプロイされ、現在は複数のレプリカに**対応していません**。
- デフォルトでは、同梱の組み込み `PostgreSQL` データベースを使用します。本番環境では、`server.externalDatabaseURL` で外部データベースを指定することを推奨します。
- Higress プラグインは、GPUStack とともにインストールされる専用の `gpustack/higress-plugins` Deployment から配信されます。Higress ゲートウェイは再起動時にこのサービスからプラグインを取得します。サービスを利用できない場合は、取得できるまでゲートウェイの起動がブロックされます。
- 同梱の `higress-core` サブチャートは、Higress をクラスターの Ingress コントローラーとしてデプロイします。別の Ingress コントローラーが稼働している場合は、`higress-core.enabled=false` とし、`gateway.ingressClassname` に既存の Higress インスタンスを指定します。

## k3s のインストール（任意） { #install-k3s-optional }

以下では Kubernetes ディストリビューションとして k3s を例に使用します。上記のバージョン要件を満たすほかの CNCF 準拠ディストリビューション（RKE2、kubeadm ベースのクラスター、マネージドクラウド Kubernetes）でも動作します。

Higress を Ingress コントローラーとして使うため、Traefik を無効にして k3s をインストールします。k3s の高可用性構成は、[k3s ドキュメント](https://docs.k3s.io/datastore/ha)をご覧ください。

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.30.11+k3s1 INSTALL_K3S_EXEC="--disable=traefik" sh -
```

セットアップを確認します。

```bash
kubectl version
```

## Helm による GPUStack のインストール { #install-gpustack-with-helm }

チャートは Docker Hub 上の `oci://registry-1.docker.io/gpustack/gpustack-chart` に OCI アーティファクトとして公開されています。チャートのバージョンは GPUStack のリリースに対応します（例：チャート `2.2.0` は GPUStack `v2.2.0` を含みます）。

最新の安定版をインストールします。`--version` を省略すると、Helm は開発版やプレリリースを除く最新の安定版チャートを自動的に選択します。

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace
```

バージョンを固定するには、`--version <chart-version>` を追加します（チャートのバージョンは GPUStack のリリースに対応し、例として `2.2.0` があります）。

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --version <chart-version>
```

デフォルトでは `higress-core` サブチャートが有効で、GPUStack とともにデプロイされます。既存の Higress を使用する場合は、同梱の Higress を無効にし、既存インスタンスを GPUStack に指定します。

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set higress-core.enabled=false \
  --set gateway.ingressClassname=<your-higress-ingressclass>
```

パラメーターの変更には、インストール時に `--set key=value` または `-f your-values.yaml` を使用します。下の[チャートパラメーター](#chart-parameters)をご覧ください。

### クローンしたリポジトリからのインストール { #installing-from-a-cloned-repository }

リポジトリをクローンしてローカルのチャートディレクトリからインストールすることもできます（エアギャップ環境やチャートを変更する場合に便利です）。

```bash
git clone https://github.com/gpustack/gpustack.git
cd gpustack/charts
helm dependency update ./gpustack-chart
helm install gpustack ./gpustack-chart \
  --namespace gpustack-system --create-namespace
```

### Higress の個別インストール { #installing-higress-separately }

`higress-core.enabled=false` にする場合は、GPUStack のデプロイ前に互換性のある Higress をインストールします。

```bash
# Add the Higress Helm repository
helm repo add higress.io https://higress.io/helm-charts

# Install higress-core (match the version pinned by the GPUStack chart)
helm install higress higress.io/higress-core \
  --namespace higress-system --create-namespace \
  --version 2.1.9
```

IngressClass が使用可能で、名前が `gateway.ingressClassname` と一致することを確認します。

```bash
kubectl get ingressclass higress
# NAME      CONTROLLER                      PARAMETERS   AGE
# higress   higress.io/higress-controller   <none>       3m46s
```

Higress のカスタマイズについては、[Higress ドキュメント](https://higress.cn/en/docs/latest/ops/deploy-by-helm)をご覧ください。

## GPUStack へのアクセス { #accessing-gpustack }

サーバー Pod が Ready になるまで待ちます。

```bash
kubectl get pods -n gpustack-system -w
```

初期管理者パスワードを取得します。

```bash
kubectl exec -it -n gpustack-system gpustack-server-0 -- \
  cat /var/lib/gpustack/initial_admin_password
```

`server.ingress.hostname` を設定していない場合は、Ingress から GPUStack UI のアドレスを取得します。

```bash
kubectl get ingress -n gpustack-system gpustack \
  -o jsonpath="{.status.loadBalancer.ingress[0].ip}"
```

ブラウザーでアドレスを開き、ユーザー名 `admin` と取得したパスワードでログインします。

## よく使う設定 { #common-configuration }

### ワーカー DaemonSet の有効化 { #enabling-worker-daemonsets }

ワーカー DaemonSet はデフォルトで無効です（`worker.enabled=false`）。有効にすると、CPU ワーカーの DaemonSet（`<release>-worker`）と、`worker.gpuVendors` に指定した各 GPU ベンダーの DaemonSet（`<release>-worker-<vendor>`）が生成されます。

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set worker.enabled=true \
  --set 'worker.gpuVendors={nvidia}'
```

`worker.gpuVendors` の対応値：`nvidia`、`mthreads`、`amd`、`ascend`、`hygon`、`metax`、`iluvatar`、`cambricon`、`thead`。

!!! note

    各 GPU DaemonSet には、Node Feature Discovery が通知する PCI デバイス存在確認用の `nodeSelector` ラベル（NVIDIA では `feature.node.kubernetes.io/pci-10de.present: "true"` など）が自動付与されます。クラスターに **NFD が必要です**。ない場合は必要なラベルを持つノードがなく、すべてのワーカー Pod が `Pending` のままになります。

GPU ベンダーを 1 つ以上指定すると（`worker.gpuVendors` が空でない場合）、すべてのワーカー Pod に必須の `podAntiAffinity`（topologyKey=hostname）が追加されます。2 つのワーカーが同じノードを共有しないため、`hostNetwork: true` のポート競合を防止できます。

CPU ワーカーの DaemonSet は、`nodeSelector` の `feature.gpustack.ai/acceleratable: "false"` により、どの GPU ベンダーにも選択されないノードを対象にします。これらのノードにワーカーを配置しない場合は `worker.cpuEnabled=false` にします。たとえば、GPU ノードと同じクラスターのコントロールプレーンにワーカーを追加したくない場合です。

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set worker.enabled=true \
  --set worker.cpuEnabled=false \
  --set 'worker.gpuVendors={nvidia}'
```

この場合、`worker.gpuVendors` には対応ベンダーを 1 つ以上指定する必要があります。指定がないとワーカー DaemonSet が 1 つも生成されないため、インストールは拒否されます。GPU DaemonSet には常に `-<vendor>` 接尾辞が付き、CPU ワーカーを無効にしても `<release>-worker` に改名されることはありません。

または、インストール後に UI の **Clusters** と **Workers** ページから GPU クラスターとワーカーノードを追加できます。

### 外部データベースの使用 { #using-an-external-database }

GPUStack はデフォルトで組み込み PostgreSQL を使用します。外部 PostgreSQL または MySQL を使用するには、`server.externalDatabaseURL` を設定します。

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set server.externalDatabaseURL="postgresql://user:password@host:port/dbname"
```

### カスタム証明書による HTTPS の有効化 { #enabling-https-with-a-custom-certificate }

`server.ingress.tls` で PEM 形式の証明書と秘密鍵の内容を指定します。両方を指定すると、Ingress のスキームが HTTPS になります。

```yaml
# values.yaml
server:
  ingress:
    hostname: gpustack.example.com
    tls:
      cert: |-
        -----BEGIN CERTIFICATE-----
        MIID...
        -----END CERTIFICATE-----
      key: |-
        -----BEGIN PRIVATE KEY-----
        MIIE...
        -----END PRIVATE KEY-----
```

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  -f values.yaml
```

### プライベートレジストリからのイメージ取得 { #pulling-images-from-a-private-registry }

すべてのイメージ（GPUStack サーバー・ワーカー、higress-plugins、同梱の higress-core の gateway/controller/pilot）をプライベートミラーレジストリから取得するには、`global.hub` を上書きします。Helm のグローバル値の伝播により、1 つの設定ですべてのイメージに適用されます。

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set global.hub=myregistry.example.com
```

`global.hub` は `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY` としてサーバーにも渡され、推論エンジンのイメージ（vLLM、llama.cpp など）も同じレジストリから取得されます。

取得用の認証情報を指定すると、チャートは `gpustack-image-pull-secret` という `docker-registry` Secret を作成してすべての Pod に設定できます。

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set imagePullSecret.credentials.registry=registry.example.com \
  --set imagePullSecret.credentials.username=myuser \
  --set imagePullSecret.credentials.password=mypassword
```

既存の Secret を使用するには、`global.imagePullSecrets` を置き換えます。

```yaml
global:
  imagePullSecrets:
    - name: my-existing-secret
```

## チャートパラメーター { #chart-parameters }

よく使うパラメーターを以下に示します。完全かつ正式な一覧は、チャートの [`values.yaml`](https://github.com/gpustack/gpustack/blob/main/charts/gpustack-chart/values.yaml) と [README](https://github.com/gpustack/gpustack/blob/main/charts/gpustack-chart/README.md) をご覧ください。

| パラメーター | デフォルト | 説明 |
| -------------------------------------- | ------------------------ | -------------------------------------------------------------------------- |
| `debug`                                | `false`                  | デバッグモードを有効化。 |
| `registrationToken`                    | `null`                   | ワーカー登録トークン。`null` の場合はランダムに生成して再利用。 |
| `clusterDomain`                        | `cluster.local`          | Kubernetes クラスターサービスのドメイン接尾辞。 |
| `global.hub`                           | `docker.io`              | コンテナーレジストリのホスト。プライベートレジストリでは上書き。 |
| `global.imagePullSecrets`              | `[gpustack-image-pull-secret]` | 全 Pod に付与しサブチャートにも伝播するイメージ取得用 Secret。 |
| `global.nodeSelector`                  | `{}`                     | 全コンポーネントのデフォルト nodeSelector。コンポーネント側の値で置換。 |
| `image.repository`                     | `gpustack/gpustack`      | 名前空間を含むイメージリポジトリ。最終参照は `{global.hub}/{repository}:{tag}`。 |
| `image.tag`                            | `null`                   | イメージタグ。デフォルトはチャートの `appVersion`。 |
| `image.pullPolicy`                     | `IfNotPresent`           | イメージ取得ポリシー。 |
| `imagePullSecret.credentials.registry` | `docker.io`              | チャートが取得用 Secret を作る際のレジストリホスト。 |
| `imagePullSecret.credentials.username` | `null`                   | レジストリのユーザー名。パスワードとともに指定すると Secret を作成。 |
| `imagePullSecret.credentials.password` | `null`                   | レジストリのパスワード。ユーザー名とともに指定すると Secret を作成。 |
| `server.ingress.hostname`              | `null`                   | サーバーの Ingress ホスト名。 |
| `server.ingress.tls.cert`              | `null`                   | Ingress の TLS 証明書（PEM）。秘密鍵とともに設定すると HTTPS を有効化。 |
| `server.ingress.tls.key`               | `null`                   | Ingress の TLS 秘密鍵（PEM）。 |
| `server.externalDatabaseURL`           | `null`                   | 外部データベースの接続文字列（PostgreSQL または MySQL）。 |
| `server.dataVolume.hostPath`           | `null`                   | サーバーデータボリュームのホストパス。PVC の代わりに hostPath を使用。 |
| `server.dataVolume.size`               | `100Gi`                  | サーバーデータボリュームのサイズ（PVC）。 |
| `server.apiPort`                       | `30080`                  | API サービスポート。 |
| `server.metricsPort`                   | `10161`                  | サーバーメトリクスポート。 |
| `server.environmentConfig`             | `{}`                     | GPUStack サーバーに渡す追加の環境変数。 |
| `server.nodeSelector`                  | `{}`                     | サーバー Pod の nodeSelector。空でなければ `global.nodeSelector` を置換。 |
| `gateway.ingressClassname`             | `higress`                | Higress IngressClass の名前。見つかるとクラスター内ゲートウェイモードを有効化。 |
| `higress-core.enabled`                 | `true`                   | 同梱の Higress ゲートウェイをデプロイ。すでに導入済みなら無効化。 |
| `worker.enabled`                       | `false`                  | ワーカー DaemonSet を生成。 |
| `worker.gpuVendors`                    | `[nvidia]`               | GPU ベンダー。各ベンダーの DaemonSet に加えて CPU DaemonSet を作成。 |
| `worker.cpuEnabled`                    | `true`                   | CPU ワーカー DaemonSet を生成。`false` の場合は GPU ベンダーが必要。 |
| `worker.nodeSelector`                  | `{}`                     | 基本のワーカー nodeSelector。空でなければ `global.nodeSelector` を置換。 |
| `worker.port`                          | `10150`                  | ワーカーのサービスポート。 |
| `worker.metricsPort`                   | `10151`                  | ワーカーメトリクスポート。 |
| `worker.dataDir`                       | `/var/lib/gpustack`      | 各ワーカー Pod の `/var/lib/gpustack` にマウントするホストパス。 |
| `worker.environmentConfig`             | `{}`                     | GPUStack ワーカーに渡す追加の環境変数。 |

## アンインストール { #uninstallation }

このリリースはサーバー以外もデプロイします。`gpustack-operator` サブチャートには Kueue と Node Feature Discovery のコントローラーが含まれ、Operator は finalizer 付きのカスタムリソースを作成します。`helm uninstall` はコントローラーも削除するため、コントローラーが動作しているうちにカスタムリソースを**先に**削除する必要があります。そうしないと、オブジェクトは `Terminating` のまま残り、`gpustack-system` 名前空間も終了しません。次の順序で実行します。

1. **すべての**コントローラーが稼働している間にワークロードを終了します。Operator は自身の `Instance` の finalizer を削除します。Kueue は、受け入れ済みの `Workload` がある `ClusterQueue` を解放しないため、何かを停止する前にこれらを削除します。`Instance` はテナントの名前空間内にあるため、`-A` ですべてを列挙します。サーバーからモデルを削除するか、インスタンスを直接削除します。各インスタンスの Pod は一緒に終了し、Workload も Instance とともに削除されます。名前空間内の Pod を一括削除しないでください。クリーンアップに必要な Operator、Kueue、NFD コントローラーも停止してしまいます。

    ```bash
    kubectl delete instances.worker.gpustack.ai --all -A
    kubectl get instances.worker.gpustack.ai -A; kubectl get workloads.kueue.x-k8s.io -A   # both empty before moving on
    ```

    Kueue をほかのアプリケーションと共有している場合は、ここや手順 3 で `--all` を使わず、GPUStack のオブジェクトだけを削除します。Operator が Kueue に作成するものはすべて `gpustack` 接頭辞を持ちます（`clusterqueue/gpustack--…`、`admissioncheck/gpustack-node-devices`）。クラスター全体のリソースは名前だけで選択できますが、名前空間付きのリソースでは名前空間も保持する必要があります。`-o name` では名前空間が省略されるためです。

    ```bash
    kubectl get clusterqueues.kueue.x-k8s.io -o name | grep '/gpustack' \
      | xargs -I {} kubectl delete {}                # clusterqueues, resourceflavors, admissionchecks
    kubectl get localqueues.kueue.x-k8s.io -A -o jsonpath='{range .items[*]}{.metadata.namespace} {.metadata.name}{"\n"}{end}' \
      | awk '$2 ~ /^gpustack/' | while read ns name; do \
        kubectl -n "$ns" delete localqueues.kueue.x-k8s.io "$name"; done
    ```

2. Operator 自身のコントローラーを停止します。ただし、次の手順の削除を確定する Kueue コントローラーを稼働させるため、リリース自体はインストールしたままにします。`gpustack-operator-worker` と `gpustack-operator-device-manager-*` の DaemonSet が動作している限り、`LocalQueue`、`ClusterQueue`、`gpustack-node-devices` AdmissionCheck を削除してもすぐに再作成され、件数がゼロになりません。

    ```bash
    kubectl -n gpustack-system scale deploy/gpustack-operator-worker --replicas=0
    kubectl -n gpustack-system get ds -l app.kubernetes.io/name=gpustack-operator-device-manager -o name | xargs -I {} \
      kubectl -n gpustack-system patch {} -p '{"spec":{"template":{"spec":{"nodeSelector":{"gpustack.ai/uninstall-pause":"true"}}}}}'
    ```

3. Kueue リソースを依存順に削除します。`LocalQueue`、`ClusterQueue`、`AdmissionCheck`、`ResourceFlavor` の順です。Kueue は利用者が残っているオブジェクトの `kueue.x-k8s.io/resource-in-use` finalizer を保持します。`gpustack-node-devices` AdmissionCheck はすべての ClusterQueue、各 ClusterQueue はその LocalQueue と受け入れ済み Workload がなくなるまで削除待ちになります。種類ごとにゼロになったことを確認して次へ進んでください。共有クラスターでは、すべてのコマンドで `--all` の代わりに手順 1 の `gpustack` 接頭辞による絞り込みを適用します。

    ```bash
    kubectl delete localqueues.kueue.x-k8s.io --all -A
    kubectl delete clusterqueues.kueue.x-k8s.io --all
    kubectl delete admissionchecks.kueue.x-k8s.io --all
    kubectl delete resourceflavors.kueue.x-k8s.io --all
    ```

4. リリースをアンインストールします。`--wait` は指定しないでください。Node Feature Discovery の post-delete prune フックが `--timeout` より長く削除の待機をブロックし、抜けるために Helm を強制終了すると prune 用 RBAC が所有者なしで残ります（手順 5 で手動削除します。`--wait` を使わなくても手順 5 を実行してください）。

    ```bash
    helm uninstall gpustack -n gpustack-system
    ```

5. `helm uninstall` が削除しないものを削除します。Helm は CRD を削除しません。また、Operator はどのリリースにも所有されない admission webhook と APIService を登録します。いずれも、削除済みの Operator Service に対して fail-closed となります。Webhook を削除しないと、以下の finalizer 除去も拒否されます。

    ```bash
    kubectl delete mutatingwebhookconfiguration gpustack-worker-mutation --ignore-not-found
    kubectl delete validatingwebhookconfiguration gpustack-worker-validation --ignore-not-found
    kubectl delete apiservice v1.gpustack.ai v1.worker.gpustack.ai --ignore-not-found
    kubectl get instances.worker.gpustack.ai -A -o jsonpath='{range .items[*]}{.metadata.namespace} {.metadata.name}{"\n"}{end}' \
      | while read ns name; do [ -n "$name" ] && kubectl -n "$ns" patch instances.worker.gpustack.ai "$name" \
        --type=json -p='[{"op":"remove","path":"/metadata/finalizers"}]' 2>/dev/null; done
    kubectl get instancetypes.worker.gpustack.ai -o name | xargs -I {} \
      kubectl patch {} --type=json -p='[{"op":"remove","path":"/metadata/finalizers"}]' 2>/dev/null
    kubectl delete crd devices.worker.gpustack.ai instances.worker.gpustack.ai \
      instancetypes.worker.gpustack.ai --ignore-not-found
    kubectl delete clusterrole node-feature-discovery-prune --ignore-not-found
    kubectl delete clusterrolebinding node-feature-discovery-prune --ignore-not-found
    ```

    `worker.gpustack.ai` グループは常に GPUStack 専用で、ほかのコンポーネントは使用しません。その CRD は無条件に削除できます。

6. リリースが残した CRD のうち、**このリリースがインストールしたものだけ**を削除します。CRD を削除すると、クラスター全体のその種類のオブジェクトがすべて削除されます。GPUStack 導入前から Kueue、Node Feature Discovery、Higress が動作しており（上記 `higress-core.enabled` を参照）、ほかでも使用中の場合は、それらのグループの CRD を残し、この手順では対象外にしてください。推測せず、このリリースの所有物を確認します。

    ```bash
    kubectl get crd -o json | jq -r '.items[]
      | select(((.metadata.annotations // {})["meta.helm.sh/release-name"] // "") == "gpustack")
      | .metadata.name'
    ```

    標準構成では Kueue の CRD、およびチャートがインストールした場合は NFD と Higress の CRD が表示されます。

    ```bash
    kubectl get crd -o name | grep 'kueue\.x-k8s\.io' | xargs -I {} kubectl delete {} --ignore-not-found
    kubectl delete crd nodefeatures.nfd.k8s-sigs.io nodefeaturerules.nfd.k8s-sigs.io \
      nodefeaturegroups.nfd.k8s-sigs.io --ignore-not-found
    kubectl delete crd envoyfilters.networking.istio.io http2rpcs.networking.higress.io \
      mcpbridges.networking.higress.io wasmplugins.extensions.higress.io --ignore-not-found
    ```

    クラスタープロバイダー自身の CRD は GPUStack が削除すべきものではありません。たとえば k3s クラスターの `*.k3s.cattle.io` や `helm.cattle.io` はクラスターに属します。

7. *（任意）* 名前空間を削除すると、Higress オブジェクト、ブートストラップ ConfigMap、ServiceAccount、登録トークン Secret も削除されます。削除されるデータを把握してください。この名前空間には、ユーザー、モデル、クラスターを格納する組み込み PostgreSQL などのサーバー永続データを持つ PVC（`gpustack-data-dir-gpustack-server-0`、`<volumeClaimTemplate>-<statefulset>-<ordinal>`）があります。回収ポリシーが `Delete`（`local-path` のデフォルト）の場合、基盤のボリュームも削除されます。再インストール用にデータを残す場合は名前空間を保持し、不要なら名前空間を削除します（回収ポリシーで削除されない場合は、先に PVC も削除します）。

    ```bash
    kubectl delete pvc gpustack-data-dir-gpustack-server-0 -n gpustack-system
    kubectl delete namespace gpustack-system
    ```
