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

    同梱の組み込み PostgreSQL を使用する場合、データはサーバーの PVC（`gpustack-data-dir-<release-name>-server-0`。デフォルトの `gpustack` リリースでは `gpustack-data-dir-gpustack-server-0`）に保存されます。これはサーバー Pod 内の `/var/lib/gpustack` にマウントされ、PostgreSQL のデータディレクトリはその下の `postgresql/data` です。アップグレード前にバックアップしてください。

    外部データベース（`server.externalDatabaseURL`）を使用する場合は、そのデータベースプロバイダーのバックアップ手順に従ってください。

## リリースのアップグレード { #upgrade-the-release }

既知の再現可能なリリースにアップグレードするため、必ず `--version <chart-version>` で対象バージョンを固定します（チャートのバージョンは GPUStack のリリースに対応し、例として `2.2.0` があります）。インストール時と同じ `-f values.yaml` や `--set` フラグを指定するか、`--reuse-values` で以前の設定値を再利用します。

```bash
helm upgrade gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system \
  --version <chart-version> \
  -f values.yaml
```

!!! warning

    `--version` を省略すると、Helm は最新の安定版チャートを自動選択します。アップグレード先を制御できなくなるため、アップグレードでは避けてください。必ず `--version` で特定のリリースを指定します。

## サーバーとワーカーの更新順序 { #server-and-worker-ordering }

GPUStack サーバーは `StatefulSet`、チャート管理下のワーカーは `DaemonSet` としてデプロイされ、リリースのアップグレード時に両方が新しいイメージへ更新されます。両者は**同じリリース**に属するため、1 回の `helm upgrade` で「サーバーを先に、次にワーカー」という順序を強制することはできません。厳密な順序が必要な環境では、ワーカー Pod の更新を許可する前に、サーバー Pod が Ready になるまで待ってください。

```bash
kubectl rollout status statefulset/<release-name>-server -n gpustack-system
```

チャート外で GPUStack クラスター経由で登録したワーカー（UI から追加した Docker や Kubernetes クラスターなど）は、この Helm リリースの管理対象ではありません。[クラスターのデプロイメントのアップグレード](cluster.md)の手順に従って更新してください。
