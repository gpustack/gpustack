# モデルカタログ { #model-catalog }

モデルカタログは、GPUStack 向けに調整されたモデルの索引です。

## モデルの閲覧 { #browse-models }

`Catalog` ページに移動すると、モデルカタログを閲覧できます。名前とカテゴリーでモデルを絞り込めます。次のスクリーンショットは、モデルカタログページです。

![モデルカタログ](../assets/model-catalog.png)

## カタログからのモデルのデプロイ { #deploy-a-model-from-the-catalog }

モデルカードをクリックすると、カタログからモデルをデプロイできます。モデルデプロイの構成ページが開くので、構成を確認・調整し、`Save` ボタンをクリックしてデプロイします。

## モデルカタログのカスタマイズ { #customize-model-catalog }

GPUStack Server の構成で `--model-catalog-file` フラグに YAML ファイルを指定すると、モデルカタログをカスタマイズできます。ローカルファイルパスまたは URL を指定できます。スキーマは、[組み込みのモデルカタログファイル](https://github.com/gpustack/gpustack/blob/main/gpustack/assets/model-catalog.yaml)を参照してください。

Server を再起動せずにカタログをカスタマイズするには、カタログソースを設定します。独自のドキュメントが同梱カタログを置き換えます。ドロワーから、編集の出発点となるカタログを取得できます。[OTA コンテンツ更新](ota-content-updates.md)を参照してください。

カスタムモデルカタログの YAML ファイルの例：

```yaml
draft_models:
- name: Qwen3-8B-EAGLE3
  algorithm: eagle3
  source: huggingface
  huggingface_repo_id: Tengyunw/qwen3_8b_eagle3
model_sets:
- name: Deepseek R1 0528
  description: DeepSeek-R1-0528 is a minor version of the DeepSeek R1 model that features enhanced reasoning depth and inference capabilities. These improvements are achieved through increased computational resources and algorithmic optimizations applied during post-training. The model delivers strong performance across a range of benchmark evaluations, including mathematics, programming, and general logic, with overall capabilities approaching those of leading models such as O3 and Gemini 2.5 Pro.
  home: https://www.deepseek.com
  icon: /static/catalog_icons/deepseek.png
  categories:
    - llm
  capabilities:
    - context/128K
  size: 671
  licenses:
    - mit
  release_date: "2025-05-28"
  specs:
    - mode: throughput
      quantization: FP8
      gpu_filters:
        vendor: nvidia
        compute_capability: ">=9.0" # Hopper or later
      source: huggingface
      huggingface_repo_id: deepseek-ai/DeepSeek-R1-0528
      backend: SGLang
      backend_parameters:
        - --enable-dp-attention
        - --context-length=32768
    - mode: standard
      quantization: FP8
      source: huggingface
      huggingface_repo_id: deepseek-ai/DeepSeek-R1-0528
      backend: vLLM
      backend_parameters:
        - --max-model-len=32768
```

### オフライン環境でのモデルカタログの使用 { #using-model-catalog-in-air-gapped-environments }

組み込みのモデルカタログは、Hugging Face または ModelScope からモデルを取得します。インターネットに接続できない隔離環境で GPUStack を使用する場合は、ローカルパスのモデルソースを使用するようにカタログをカスタマイズできます。例：

```yaml
model_sets:
- name: Deepseek R1 0528
  description: DeepSeek-R1-0528 is a minor version of the DeepSeek R1 model that features enhanced reasoning depth and inference capabilities. These improvements are achieved through increased computational resources and algorithmic optimizations applied during post-training. The model delivers strong performance across a range of benchmark evaluations, including mathematics, programming, and general logic, with overall capabilities approaching those of leading models such as O3 and Gemini 2.5 Pro.
  home: https://www.deepseek.com
  icon: /static/catalog_icons/deepseek.png
  categories:
    - llm
  capabilities:
    - context/128K
  size: 671
  licenses:
    - mit
  release_date: "2025-05-28"
  specs:
    - mode: throughput
      quantization: FP8
      gpu_filters:
        vendor: nvidia
        compute_capability: ">=9.0" # Hopper or later
      source: local_path
      local_path: /path/to/DeepSeek-R1-0528
      backend: SGLang
      backend_parameters:
        - --enable-dp-attention
        - --context-length=32768
    - mode: standard
      quantization: FP8
      source: local_path
      # assuming you have /path/to/DeepSeek-R1-0528 directory
      local_path: /path/to/DeepSeek-R1-0528
      backend: vLLM
      backend_parameters:
        - --max-model-len=32768
```

### モデルカタログのスキーマ { #model-catalog-schema }

モデルカタログの YAML ファイルには、`draft_models` と `model_sets` の 2 つの主要セクションがあります。

- `draft_models`：投機的デコーディング用のドラフトモデルの一覧。
- `model_sets`：テストおよび最適化済みのモデルセットの一覧。

各ドラフトモデルには、次のフィールドがあります。

| フィールド | 型 | 説明 |
| --- | --- | --- |
| name | string | ドラフトモデルの名前。 |
| algorithm | string | モデルの投機的デコーディングのアルゴリズム。現在は eagle3 のみに対応しています。 |
| source | string | モデルのソース（huggingface、model_scope など）。 |
| huggingface_repo_id | string | モデルの Hugging Face リポジトリ ID（source が huggingface の場合）。 |
| model_scope_model_id | string | モデルの ModelScope リポジトリ ID（source が model_scope の場合）。 |

各モデルセットには、次のフィールドがあります。

| フィールド | 型 | 説明 |
| --- | --- | --- |
| name | string | モデルの名前。 |
| description | string | モデルの簡単な説明。 |
| home | string | モデルのホームページ URL。 |
| icon | string | モデルのアイコン URL。 |
| categories | list of str | モデルが属するカテゴリーの一覧。 |
| capabilities | list of str | モデルの能力の一覧。 |
| size | int | 10 億パラメーター単位のモデルサイズ。 |
| licenses | list of str | モデルのライセンスの一覧。 |
| release_date | string | YYYY-MM-DD 形式のモデルのリリース日。 |
| specs | list of spec | モデルのデプロイ仕様の一覧。 |

各デプロイ仕様には、次のフィールドがあります。

| フィールド | 型 | 説明 |
| --- | --- | --- |
| mode | string | GPUStack は、スループット、レイテンシー、標準などの用途に応じて、通常のモードと最適化モードを提供します。必要に応じてカスタムモードも定義できます。 |
| quantization | string | 量子化の種類（FP16、FP8、INT8 など）。 |
| gpu_filters | dict | 互換性のある GPU を指定する GPU フィルター。 |

デプロイ仕様のその他のフィールドは、モデル API のフィールドに準じます。詳細は、[API リファレンス](../api-reference.md)を参照してください。
