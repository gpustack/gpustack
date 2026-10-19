---
translation_status: translated
---

# アンインストール { #uninstallation }

GPUStack は通常コンテナーとしてインストールするため、
アンインストールでは主にコンテナーと関連するデータボリュームを削除します。

たとえば、`gpustack` という名前の Docker コンテナーで GPUStack を実行している場合は、次を実行します。

```bash
docker rm -f gpustack

```

必要に応じて、次のコマンドで関連するデータボリュームも削除できます。

```bash
docker volume rm <data_volume_name>

```
