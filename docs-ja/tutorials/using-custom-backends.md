---
translation_status: translated
---

# カスタム推論バックエンドの利用 { #using-custom-inference-backends }

このガイドでは、GPUStack にカスタム推論バックエンドを追加する方法を説明します。検証済みのコミュニティ設定の利用と、独自の設定の作成を扱います。

パラメーターの説明は、[ユーザーガイド](../user-guide/inference-backend-management.md)をご覧ください。

## バックエンドの種類 { #backend-types }

GPUStack は、3 種類の推論バックエンドに対応します。

- **組み込み**：GPUStack が管理する事前設定済みのバックエンド（vLLM、MindIE、VoxBox、SGLang など）。ハードウェアに応じて自動最適化されます。
- **コミュニティ**：検証済みのカスタムバックエンド設定。実体は「community」ラベル付きの CustomBackend で、手動セットアップを簡単にします。
- **カスタム**：独自の Docker イメージとコマンドで設定するバックエンド。

## コミュニティバックエンドの利用 { #using-community-backends }

コミュニティバックエンドなら、一般的な推論エンジンを素早く追加できます。

**手順：**

1. Inference Backend ページで「Add Backend」をクリックします。
2. 「Community」を選択します。
3. 「Community Backend Marketplace」を参照し、必要なバックエンドを有効化します。

## カスタムバックエンドの作成 { #creating-custom-backends }

### 基本手順 { #core-steps }
1. 必要な推論バックエンドの Docker イメージを用意します。
2. イメージの ENTRYPOINT または CMD を理解し、起動コマンドを決定します。
3. Inference Backend ページで設定を追加します。
4. モデルのデプロイ時に、新しいバックエンドを選択します。

### 例：TensorRT-LLM { #example-tensorrt-llm }
TensorRT-LLM を例に、バックエンドの追加と利用を説明します。
> 以下は機能の実演用であり、性能を最適化した設定ではありません。性能調整は各バックエンドの公式ドキュメントを確認してください。

1. TensorRT-LLM ドキュメントからリンクされた[リリースページ](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release)で、必要なイメージを探します。
2. TensorRT-LLM イメージは、`trtllm-serve` で推論サービスを起動する必要があります。指定しないと対話シェルになります。`run_command` は `{{model_path}}`、`{{port}}`（必要に応じて `{{model_name}}`、`{{worker_ip}}`、`{{gpu_count}}`、`{{gpu_ids}}`）などのプレースホルダーに対応し、ワーカーへの配置時に実際の値へ置換されます。`{{gpu_count}}` と `{{gpu_ids}}` は配置先ワーカーに割り当てられた GPU を示します（カスタムバックエンドは単一ワーカーで動作します）。
3. Inference Backend ページで設定を追加します。YAML のインポートにも対応しています。例：
```yaml
backend_name: TensorRT-LLM-custom
default_version: 1.2.0rc0
version_configs:
  1.2.0rc0:
    image_name: nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc0
    run_command: 'trtllm-serve {{model_path}} --host 0.0.0.0 --port {{port}}'
    custom_framework: cuda
```

4. Deployments ページで新しいバックエンドを選び、モデルをデプロイします。
![カスタムバックエンドの画面](../assets/tutorials/using-custom-backend/deploy-by-custom-backend.png)

**結果**

推論バックエンドのサービスが起動すると、model_instance の状態が RUNNING になります。
![カスタムバックエンドの画面](../assets/tutorials/using-custom-backend/custom-backend-running.png)
Playground で会話できます。
![カスタムバックエンドの画面](../assets/tutorials/using-custom-backend/use-custom-backend-in-playground.png)

## 高度な設定 { #advanced-configuration }

### 環境変数の使用 { #using-environment-variables }

環境変数を使うと、コマンドに値を埋め込まず柔軟に設定できます。

```yaml
backend_name: advanced-backend-custom
default_env:
  CACHE_DIR: /models/cache
  LOG_LEVEL: info
version_configs:
  v1:
    image_name: my-backend:v1
    custom_framework: cuda
    run_command: 'serve {{model_path}} --cache {{CACHE_DIR}} --log-level {{LOG_LEVEL}} --port {{port}}'
    env:
      LOG_LEVEL: debug  # Override for this version
```

この例では：
- バックエンドレベルで `CACHE_DIR` と `LOG_LEVEL` を定義します。
- バージョン `v1` が `LOG_LEVEL` を `debug` に上書きします。
- コマンド内では両方とも `{{VAR_NAME}}` 構文で参照します。

### カスタムエントリーポイント { #custom-entrypoint }

独自の初期化が必要なイメージでは、デフォルトのエントリーポイントを上書きします。バックエンドとバージョンの両レベルで設定できます。

```yaml
backend_name: custom-entry-backend-custom
default_entrypoint: /usr/local/bin/default-init
version_configs:
  v1:
    image_name: my-backend:v1
    custom_framework: cuda
    run_command: 'serve {{model_path}} --port {{port}}'
  v2:
    image_name: my-backend:v2
    custom_framework: cuda
    entrypoint: /usr/local/bin/v2-init  # Version-specific entrypoint overrides default
    run_command: 'serve {{model_path}} --port {{port}}'
```
