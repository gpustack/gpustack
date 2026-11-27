---
translation_status: translated
---

# Helm によるアップグレード { #upgrade-via-helm }

Kubernetes ベースの GPUStack は、`helm upgrade` でアップグレードできます。

以下の手順は、Helm によるデプロイに初めて対応した GPUStack v2.2.0 以降にのみ適用されます。

!!! note

    Higress を使用した Kubernetes 上での GPUStack のデプロイとアップグレードは、現在**実験的機能**とされています。

!!! warning

    **最初にバックアップ：** アップグレードを始める前に、データベースのバックアップを強く推奨します。

    同梱の組み込み PostgreSQL を使用する場合、データはサーバーの PVC（`gpustack-data-dir-<release-name>-server-0`。デフォルトの `gpustack-server` リリースでは `gpustack-data-dir-gpustack-server-server-0`）に保存されます。これはサーバー Pod 内の `/var/lib/gpustack` にマウントされ、PostgreSQL のデータディレクトリはその下の `postgresql/data` です。アップグレード前にバックアップしてください。

    外部データベース（`server.externalDatabaseURL`）を使用する場合は、そのデータベースプロバイダーのバックアップ手順に従ってください。

## Helm アップグレード時のトレースストレージ { #trace-storage-on-helm-upgrades }

組み込み Tempo は独立したワークロードで実行されます。サーバー Pod 内で Tempo が稼働していたリリースをアップグレードすると、新しい Tempo ワークロードは独自のボリュームを使用します。Helm は、サーバーのデータボリュームにある既存の `<data_dir>/tempo` ファイルをそのまま残します。過去のトレースを保持する必要がある場合は、そのファイルを別途バックアップしてください。

新しい Tempo Pod はデフォルトで `emptyDir` を使い、Pod の置き換えごとに空のトレースストレージで起動します。以降の Pod の置き換えでトレースを保持するには、アップグレード前に `tracing.tempo.existingClaim` に既存の PVC を指定します。[Helm の設定](../user-guide/observability.md#configure-helm)を参照してください。

## リリースのアップグレード { #upgrade-the-release }

既知の再現可能なリリースにアップグレードするため、必ず `--version <chart-version>` で対象バージョンを固定します（チャートのバージョンは GPUStack のリリースに対応し、例として `2.2.0` があります）。また、インストール時と同じ `-f values.yaml` や `--set` フラグを指定するか、`--reuse-values` で以前の設定値を再利用します。リリース名はインストール時に従ったドキュメントによって異なります（旧ドキュメントでは `gpustack`、現在のドキュメントでは `gpustack-server`）。そのため、下記のコマンドは名前空間内の StatefulSet の `meta.helm.sh/release-name` アノテーションから自動的に導出し（GPUStack クラスター経由で登録されたワーカー専用リリースは StatefulSet を所有しないためパターンで除外されます）、一致するリリースがちょうど 1 つである場合にのみアップグレードを実行します。

```bash
RELEASE_NAME=$(kubectl get statefulset -n gpustack-system \
  -o jsonpath='{range .items[*]}{.metadata.annotations.meta\.helm\.sh/release-name}{"\n"}{end}' \
  | grep -E '^gpustack(-server)?$' | sort -u)
[ "$(printf '%s\n' "$RELEASE_NAME" | grep -c .)" -eq 1 ] \
  || { echo "expected one GPUStack server release in gpustack-system, got: '$RELEASE_NAME'" >&2; false; } \
  && helm upgrade "$RELEASE_NAME" oci://registry-1.docker.io/gpustack/gpustack-chart \
    --namespace gpustack-system \
    --version <chart-version> \
    -f values.yaml
```

!!! warning

    `--version` を省略すると、Helm は最新の安定版チャートを自動選択します。アップグレード先を制御できなくなるため、アップグレードでは避けてください。必ず `--version` で特定のリリースを指定します。

## サーバーとワーカーの更新順序 { #server-and-worker-ordering }

GPUStack サーバーは `StatefulSet`、チャート管理下のワーカーは `DaemonSet` としてデプロイされ、リリースのアップグレード時に両方が新しいイメージへ更新されます。両者は**同じリリース**に属するため、1 回の `helm upgrade` で「サーバーを先に、次にワーカー」という順序を強制することはできません。厳密な順序が必要な環境では、ワーカー Pod の更新を許可する前に、サーバー Pod が Ready になるまで待ってください。

```bash
kubectl rollout status statefulset/${RELEASE_NAME}-server -n gpustack-system
```

チャート外で GPUStack クラスター経由で登録したワーカー（UI から追加した Docker や Kubernetes クラスターなど）は、この Helm リリースの管理対象ではありません。[クラスターのデプロイメントのアップグレード](cluster.md)の手順に従って更新してください。
