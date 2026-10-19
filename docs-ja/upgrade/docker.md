---
translation_status: translated
---

# Docker によるアップグレード { #upgrade-via-docker }

Docker ベースの GPUStack は、新しいイメージを取得してコンテナーを再作成することでアップグレードできます。

以下のアップグレード手順は、GPUStack v2.0 以降にのみ適用されます。

v0.7 より前のインストールについては、[移行ガイド](../migration.md)をご覧ください。

!!! note

    1. アップグレード時は、最初に GPUStack サーバー、次にワーカーをアップグレードします。

    2. main（開発版）またはリリース候補（rc）版には破壊的変更が含まれる場合があるため、これらのバージョンから、またはこれらへのアップグレードは**行わないでください**。main 版や rc 版を試す場合は、新規にインストールしてください。

!!! warning

    **最初にバックアップ：** アップグレードを始める前に、データベースのバックアップを強く推奨します。

    組み込み PostgreSQL を使用するデフォルトの構成では、GPUStack サーバーを停止し、コンテナー内の次の PostgreSQL データベースディレクトリをバックアップします。

    ```
    /var/lib/gpustack/postgresql/data
    ```

    外部データベースを使用する場合は、そのデータベースプロバイダーのバックアップ手順に従ってください。

最初に**サーバー**、次に**ワーカー**をアップグレードします。サーバーは、新しいイメージ（特定のバージョンタグまたは `latest` タグ）を取得し、古いコンテナーを削除して、以前と**同じ引数とボリューム**で新しいイメージのコンテナーを起動することでアップグレードします。

## サーバーのアップグレード { #upgrade-the-server }

```bash
docker pull gpustack/gpustack:latest  # or: docker pull gpustack/gpustack:vx.y.z

docker stop gpustack
docker rm gpustack

docker run -d --name gpustack \
  ... \
  gpustack/gpustack:latest
  ...
```

## ワーカーのアップグレード { #upgrade-the-workers }

新しいバージョンのサーバーが起動したら、各クラスターのワーカーをアップグレードします。Docker、Kubernetes、クラウドクラスターの手順は、[クラスターのデプロイメントのアップグレード](cluster.md)をご覧ください。

## Docker Compose によるアップグレード { #upgrade-via-docker-compose }

クローンしたリポジトリから Docker Compose でデプロイした場合は、新しいリリースタグをチェックアウトし、更新したイメージでコンテナーを再作成します。

```bash
cd gpustack

# Fetch the tags and check out the target stable release
git fetch --tags
git checkout <new-tag>   # e.g. vx.y.z

cd docker-compose

# Pull the new image and recreate the containers
sudo docker compose -f docker-compose.server.yaml pull
sudo docker compose -f docker-compose.server.yaml up -d
```
