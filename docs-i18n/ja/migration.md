---
translation_status: translated
---

# v0.7 以前から v2 への移行 { #migrating-from-v07-and-earlier-versions-to-v2 }

!!! note

    v2.0.0 以降、GPUStack ワーカーが正式にサポートする OS は Linux のみです。Windows または macOS を使用している場合は、データディレクトリを Linux システムに移して移行を実行してください。

    Windows と macOS では、Docker Desktop を使用して GPUStack サーバー（組み込みワーカーなし）を引き続き実行できます。

## 移行前の準備 { #before-migration }

### 破壊的変更 { #breaking-changes }

#### 1. Ollama モデルソースの削除（v0.7.x 以降） { #1-removal-of-ollama-model-source-since-v07x }

- **変更：** バージョン 0.7 以降、GPUStack はモデルソースとして `ollama` をサポートしていません。
- **影響：** ソースが `ollama` のモデル、モデルファイル、モデルインスタンスは、アップグレード時に保持されません。
- **必要な対応：** v0.7 より前のバージョンから更新し、現在 `ollama` ソースからデプロイしたモデルがある場合は、アップグレード前に手動で移行する必要があります。  
  影響を受けるモデルは、次の対応ソースのいずれかを使用して再デプロイすることを推奨します。
    - Hugging Face
    - ModelScope
    - ローカルパス

    アップグレードを開始する前に、**GPUStack UI** からモデルを再デプロイして移行できます。

### データのバックアップ { #backup-your-data }

!!! warning

      **最初にバックアップ：** サーバーの移行を始める前に、データベースのバックアップを強く推奨します。

      v0.7 以前のデフォルトの構成では、GPUStack サーバーを停止し、コンテナー内の次のデータディレクトリをバックアップします。

      ```
      /var/lib/gpustack
      ```

移行を開始する前に、[インストール要件](./installation/requirements.md)を確認してください。

v0.7.1 より前に **Docker を使用せずに** GPUStack を導入した場合（pip install やインストールスクリプトなど）は、移行前に Docker Engine の[インストールガイド](https://docs.docker.com/engine/install/)に従って Docker をインストールしてください。

v0.7.1 より前の GPUStack で GPU による推論を使用していた場合、Docker のインストール後に、対応するアクセラレーターランタイムの Container Toolkit または Container Runtime が必要か確認してください。**インストール要件**の手順で確認・インストールできます。

## 移行手順 { #migration-steps }

### 既存のデータディレクトリの確認 { #identify-your-legacy-data-directory }

以前の GPUStack インストールで使用していたデータディレクトリを探します。デフォルトのパスは次のとおりです。

```
/var/lib/gpustack
```

ほかのインストール方法では、[こちら](faq.md/#where-are-gpustacks-data-stored)を参照してデータディレクトリを確認してください。以下の手順では、このパスを `${your-data-dir}` と表記します。

v2.0.0 のアーキテクチャ変更により、`postgres` や `s6-supervisor` などの新しいコンポーネントが、root 以外のユーザーで `${your-data-dir}` 内のファイルを読み書きする場合があります。必要なすべてのユーザーが `${your-data-dir}` にアクセスできるようにしてください。次のスクリプトで権限を調整できますが、実行前にセキュリティ上のリスクとスクリプトの動作を十分に理解してください。

```bash
dir="${your-data-dir}"
chmod a+rx "${dir}/log" || true
chmod a+rx $dir
while [ "$dir" != "/" ]; do
  chmod a+x "$dir"
  dir=$(dirname "$dir")
done
```

### Docker によるサーバーの移行 { #migrate-server-using-docker }

v2.0.0 以降は、サーバーやワーカーのイメージに GPU コンピューティングプラットフォームやバージョン（`-cpu`、`-cuda12.8`、`-rocm` など）を指定する必要がありません。共通イメージ `gpustack/gpustack:latest` を使用します。

#### 組み込みデータベースの移行（SQLite → PostgreSQL） { #embedded-database-migration-sqlite-postgresql }

v0.7 以前では、GPUStack は管理データの保存に組み込み SQLite データベースをデフォルトで使用していました。v2.0.0 以降は SQLite のサポートを終了し、性能と拡張性を高めるため、デフォルトで組み込み PostgreSQL を使用します。

`GPUSTACK_DATA_MIGRATION=true` を指定して GPUStack を起動すると、組み込みデータベースの移行が有効になります。`${your-data-dir}` を、元の SQLite データベースと関連ファイルを含む既存のデータディレクトリに置き換えてください。

```bash
sudo docker run -d --name gpustack \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  --env GPUSTACK_DATA_MIGRATION=true \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --volume ${your-data-dir}:/var/lib/gpustack \
  --runtime nvidia \
  gpustack/gpustack:latest
```

Docker から `docker: Error response from daemon: unknown or invalid runtime name: nvidia` が返る場合は、まず[アクセラレーターランタイム要件](./installation/requirements.md#accelerator-runtime-requirements)を確認し、[ほかの GPU アーキテクチャ](#other-gpu-architectures)に従って `--runtime nvidia` を使用環境のランタイムに変更してください。

データベース移行時も、`--data-dir` または `GPUSTACK_DATA_DIR` によるデータディレクトリの変更に対応しています。コマンドを次のように調整します。

```diff
...
-  --volume ${your-data-dir}:/var/lib/gpustack
+  --volume ${your-data-dir}:${your-data-dir}
...
+   gpustack/gpustack:latest \
+   --data-dir ${your-data-dir}
```

#### 外部データベースの移行 { #external-database-migration }

GPUStack は管理データの保存に外部データベースを使用できます。以前に外部データベースを使用していた場合は、次のコマンドでサーバーを起動します。

```bash
sudo docker run -d --name gpustack-server \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --volume ${your-data-dir}:/var/lib/gpustack \
  --runtime nvidia \
  gpustack/gpustack \
  --database-url ${your-database-url}
```

### Docker によるワーカーの移行 { #migrate-workers-using-docker }

ワーカーノードでは、`${your-data-dir}` を既存のワーカーデータディレクトリのパスに置き換え、次のコマンドを使用します。

```bash
sudo docker run -d --name gpustack-worker \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  --volume ${your-data-dir}:/var/lib/gpustack \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --runtime nvidia \
  gpustack/gpustack \
  --server-url ${server-url} \
  --token ${token}
```

Docker コマンドには `--volume /var/run/docker.sock:/var/run/docker.sock` と `--runtime nvidia` の両方を必ず追加してください。以前のバージョンではこれらは不要でした。別のアクセラレーターランタイムでは、[ほかの GPU アーキテクチャ](#other-gpu-architectures)を参照し、`--runtime nvidia` の代わりに適切なオプションを使用します。

これにより、既存のデータを使用して GPUStack ワーカーが起動し、指定したサーバーに接続します。

### ほかの GPU アーキテクチャ { #other-gpu-architectures }

NVIDIA 以外のアーキテクチャ（AMD、Ascend など）でも移行手順は同じです。[アクセラレーターランタイム要件](./installation/requirements.md#accelerator-runtime-requirements)で、ご使用の GPU アーキテクチャが対応していることを確認してください。

たとえば、AMD GPU を搭載したホストでサーバーとワーカーを実行する場合は、docker run コマンドを次のように変更します。

```diff
  sudo docker run -d --name gpustack \
  ...
- --runtime nvidia \
+ --runtime amd \
  ...
```

### モデルインスタンスの再作成 { #recreating-model-instances }

アップグレード後、既存のモデルインスタンスが `Starting` のままになることがあります。この場合は、モデルインスタンスを再作成すると正常に動作します。

### llama-box からの移行 { #migration-from-llama-box }

以前のバージョンで llama-box を推論バックエンドとして使用していた場合、v2.0.0 以降ではサポートされないことに注意してください。代わりにカスタム推論バックエンドで llama.cpp を使用します。

1. Inference Backend ページで `llama.cpp` カスタムバックエンドを作成します。llama.cpp の設定については、[こちら](./user-guide/inference-backend-management.md#enable-community-inference-backend)をご覧ください。
2. Deployment ページで、元々 llama-box で起動したモデルを編集し、バックエンドを llama.cpp に変更します。
3. 保存後にモデルインスタンスを再作成します。

!!! Note

    現在、カスタム推論バックエンドでは複数ワーカー間の分散推論に対応していません。

### カスタムバックエンドバージョンからの移行 { #migration-from-custom-backend-versions }

GPUStack v2.0.0 より前にカスタムバックエンドバージョンを使用していた場合、それらは Python 仮想環境に依存しており、v2.0.0 以降では**サポートされない**ことに注意してください。現在、すべての推論バックエンドはコンテナー環境で動作します。

モデルを引き続き使用するには、次のいずれかの方法で**再作成する**必要があります。

**方法 1 - 組み込みバックエンドバージョンを使用：**

GPUStack v2.0.0 以降は、組み込み推論バックエンドに事前設定された複数のバージョンを提供します。モデルのデプロイメントを再作成し、モデルの要件に最も合う組み込みバージョンを選択してください。

**方法 2 - 組み込みバックエンドにカスタムバージョンを追加：**

組み込みバージョンが要件に合わない場合は、カスタムバージョンを追加して組み込み推論バックエンドを拡張できます。詳しい手順は、[組み込み vLLM 推論バックエンドへのカスタムバージョンの追加](user-guide/inference-backend-management.md#example-add-a-custom-version-to-the-built-in-vllm-inference-backend)をご覧ください。
