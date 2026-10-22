---
translation_status: translated
---

# トラブルシューティング { #troubleshooting }

## GPUStack ログの表示 { #view-gpustack-logs }

デフォルトの構成では、次のコマンドで GPUStack のログを確認できます。

```bash
docker logs -f gpustack
```

## デバッグモードの有効化 { #enable-debug-mode }

GPUStack の実行時に `--debug` フラグを指定すると、`DEBUG` モードを有効にできます。

```diff
sudo docker run -d --name gpustack \
    ...
    gpustack/gpustack \
+    --debug
    ...
```

**サーバーコンテナー**内で次のコマンドを実行すると、実行中の GPUStack でもデバッグモードを有効にできます。

```bash
gpustack reload-config --set debug=true
```

`--debug` は詳細ログを有効にしますが、Python のメモリ割り当て追跡は開始しません。メモリプロファイリングを有効にするには、GPUStack の起動前に `GPUSTACK_DEBUG_ENABLE_TRACEMALLOC=true` を設定してください。既定値は `false` で、`--debug` とは独立しており、変更後は再起動が必要です。追跡には CPU とメモリの追加負荷がかかるため、メモリ使用量を調査するときに有効にしてください。

## ログレベルの設定 { #configure-log-level }

ログレベルのエンドポイントには認証が必要です。

**サーバーコンテナー**内で次のコマンドを実行すると、GPUStack サーバーのログレベルを実行時に変更できます。管理者 API キーで認証してください。

```bash
curl -X PUT http://localhost/debug/log_level \
    -H "Authorization: Bearer <YOUR_API_KEY>" \
    -d "debug"
```

GPUStack ワーカーでも同様です。ローカルのワーカートークン（デフォルトは `/var/lib/gpustack/worker_token`）で認証してください。

```bash
curl -X PUT http://localhost:10150/debug/log_level \
    -H "Authorization: Bearer $(cat /var/lib/gpustack/worker_token)" \
    -d "debug"
```

使用可能なログレベルは `trace`、`debug`、`info`、`warning`、`error`、`critical` です。

## 管理者パスワードのリセット { #reset-admin-password }

管理者パスワードを忘れた場合は、**サーバーコンテナー**内で次のコマンドを実行してリセットできます。

```bash
gpustack reset-admin-password
```

GPUStack の起動時に `--port` でデフォルトのポートを変更した場合は、`--server-url` パラメーターで GPUStack の URL を指定します。サーバー上でローカルに実行し、`localhost` 経由でアクセスする必要があります。

```bash
gpustack reset-admin-password --server-url http://localhost:9090
```

## アクセラレーター検出の診断への協力 { #assist-in-accelerators-detection-diagnosis }

[インストールガイド](./installation/requirements.md)に従って GPUStack ワーカーを正常にデプロイした後、  
ワーカーがデバイスを検出できない場合は、  
該当するワーカーコンテナーに入り、次のコマンドを実行して結果を [GPUStack](https://github.com/gpustack/gpustack/issues) に報告してください。

```bash
time GPUSTACK_RUNTIME_LOG_LEVEL=debug GPUSTACK_RUNTIME_LOG_EXCEPTION=1 gpustack-runtime detect --format json
```

## モデルデプロイの診断への協力 { #assist-in-model-deployment-diagnosis }

モデルのデプロイ後に問題が発生した場合は、
該当するワーカーコンテナーに入り、次のコマンドを実行して結果を [GPUStack](https://github.com/gpustack/gpustack/issues) に報告してください。

```bash
gpustack-runtime inspect <model instance name>
```
