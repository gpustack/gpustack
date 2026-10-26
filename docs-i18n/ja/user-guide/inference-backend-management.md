# 推論バックエンドの管理 { #inference-backend-management }

GPUStack では、管理者が推論バックエンドとそのバージョンを構成できます。

本記事は、Inference Backend ページの操作ガイドです。対応する組み込みバックエンドとその機能については、[組み込み推論バックエンド](built-in-inference-backends.md)を参照してください。

カスタムバックエンドの構成方法と、動作確認済みの例については、[カスタム推論バックエンド](../tutorials/using-custom-backends.md)を参照してください。

## バックエンドのソース { #backend-sources }

GPUStack は 3 種類の推論バックエンドに対応しています。

- **組み込み**：GPUStack が保守する構成済みバックエンド（vLLM、MindIE、VoxBox など）。削除できません。
- **コミュニティ**：コミュニティバックエンドマーケットプレイスで共有されるバックエンド。必要に応じて有効化できます。
- **カスタム**：独自の構成で作成するバックエンド。自由に追加、編集、削除できます。

組み込みとコミュニティの一覧は、GPUStack をアップグレードせずに OTA で更新されます。どちらの一覧も、独自のドキュメントを参照するように設定できます。[OTA コンテンツ更新](ota-content-updates.md)を参照してください。

## パラメーターの説明 { #parameter-description }

| パラメーター名 | 説明 | 必須 |
| --- | --- | --- |
| 名前 | 推論バックエンド名 | はい |
| ヘルスチェックパス | バックエンドが起動して応答していることを確認するパス。既定値は /v1/models（OpenAI 互換）です。 | いいえ |
| 既定の実行コマンド | コンテナの起動コマンドと引数。vLLM の例：`vllm serve {{model_path}} --port {{port}} --served-model-name {{model_name}} --host {{worker_ip}}`。デプロイが Worker にスケジュールされる際に、`{{model_path}}`、`{{model_name}}`、`{{port}}`、`{{worker_ip}}`、`{{gpu_count}}`、`{{gpu_ids}}`、`{{VAR_NAME}}`（環境変数）のプレースホルダーを自動的に置き換えます。その後、POSIX 形式で引数を分割します。`{{gpu_count}}` と `{{gpu_ids}}` は、スケジュール先の単一 Worker に割り当てられた GPU を反映します（カスタムバックエンドは複数 Worker の推論に対応していません）。空白を含む値は引用符で囲み、シェル演算子は使用しないでください。 | いいえ |
| 既定のエントリーポイント | コンテナのエントリーポイントの上書き。設定すると、このバックエンドのイメージのエントリーポイントを置き換えます。引数は POSIX 形式で分割します。 | いいえ |
| 既定の環境変数 | このバックエンドのすべてのバージョンに設定する環境変数。コマンド内で `{{VAR_NAME}}` 構文により参照できます。バージョン固有の環境変数が優先されます。 | いいえ |
| 既定のバックエンドパラメーター | デプロイ時の高度なバックエンドパラメーター欄を事前入力します。起動前に調整できます。 | いいえ |
| フラグ形式 | バックエンドの起動コマンドを組み立てる際の `--key value` ペアの形式を制御します。`Space Separated (--key value)` または `Equal Sign (--key=value)` を指定します。空の場合は正規化を無効にし、入力したパラメーターをそのまま保持します。複数値のオプションは常に空白区切りのままです。[フラグ形式](#flag-format)を参照してください。 | いいえ |
| よく使うバックエンドパラメーター | デプロイ時の `Backend Parameters` 入力欄に、すばやく選択する候補として表示するパラメーター一覧。`Default Backend Parameters` と異なり、選択用の候補だけで、フォームを開いた際には事前入力しません。実行時の動作には影響しません。 | いいえ |
| 説明 | 説明 | いいえ |
| バージョン構成 | このバックエンドで利用可能なバージョンを構成します。 | はい |
| 既定のバージョン | デプロイ時に事前選択するバージョン。バージョンを選択しない場合は、そのイメージを使用します。 | いいえ |

## 環境変数 { #environment-variables }

環境変数は 2 つのレベルで定義できます。

- **バックエンドレベル**（`Default Environment Variables`）：バックエンドのすべてのバージョンに適用します。
- **バージョンレベル**（`Environment Variables`）：特定のバージョンに適用し、バックエンドレベルの変数を上書きします。

環境変数は、コマンド内で `{{VAR_NAME}}` 構文を使って参照できます。

**例：**

```yaml
backend_name: my-backend-custom
default_env:
  MODEL_CACHE_DIR: /cache
  LOG_LEVEL: info
version_configs:
  v1:
    image_name: my-image:v1
    custom_framework: cuda
    run_command: "serve {{model_path}} --cache {{MODEL_CACHE_DIR}} --log-level {{LOG_LEVEL}} --port {{port}}"
    env:
      MODEL_CACHE_DIR: /custom-cache # Overrides backend-level value
```

この例では、次のようになります。

- バックエンドレベルで `MODEL_CACHE_DIR` を `/cache` に設定します。
- バージョン `v1` では `/custom-cache` に上書きします。
- `LOG_LEVEL` はすべてのバージョンで `info` のままです。
- 両方の変数を、コマンド内で `{{VAR_NAME}}` 構文により参照します。

バージョン構成のパラメーターの説明：

| パラメーター名 | 説明 | 必須 |
| --- | --- | --- |
| バージョン | デプロイ時に Backend Version ドロップダウンに表示するバージョン名。 | はい |
| イメージ名 | バックエンドのコンテナイメージ名（`ghcr.io/org/image:tag` など）。 | はい |
| フレームワーク<br/>(custom_framework) | バックエンドのフレームワーク（内部識別子：`custom_framework`）。対応フレームワークに基づいてデプロイとスケジューリングを絞り込みます。 | はい |
| 環境変数 | このバージョン固有の環境変数。バックエンドレベルの `Default Environment Variables` を上書きします。`Execution Command` 内で `{{VAR_NAME}}` 構文により参照できます。 | いいえ |
| エントリーポイント | バージョン固有のコンテナエントリーポイントの上書き。省略すると `Default Entrypoint` を使用します。引数は POSIX 形式で分割します。 | いいえ |
| 実行コマンド | バージョン固有の起動コマンド。省略すると既定の実行コマンドを使用します。解析と分割のルールは `Default Execution Command` と同じです。 | いいえ |

## フラグ形式 { #flag-format }

`Flag Format` は、バックエンドの起動コマンドを組み立てる際に、GPUStack が各 `--key value` ペアをどのように出力するかを制御します。バックエンドレベルで構成し、そのバックエンドを使用するすべてのモデルデプロイに適用します。Worker がコンテナの起動コマンドを組み立てる際に有効になります。

指定できる値：

- `Space Separated (--key value)`：空白で区切り、`--key value` として出力します。
- `Equal Sign (--key=value)`：`=` で結び、`--key=value` として出力します。
- 空（既定）：正規化せず、`Backend Parameters` に入力したとおりに保持します。

推論エンジンのコマンドラインパーサーが片方の形式だけを受け入れる場合、このフィールドを使うと、各パラメーターを手動で書き換えずに統一した形式を適用できます。

**例：**

バックエンドの `Flag Format` を `Equal Sign (--key=value)` に設定し、このバックエンドを使用するデプロイの `Backend Parameters` を次のように指定した場合：

```bash
--tensor-parallel-size 2 --max-model-len 8192 --lora-modules a=/path1 b=/path2
```

起動コマンドでは次のように正規化されます。

```bash
vllm serve Qwen/Qwen3.5-0.8B --tensor-parallel-size=2 --max-model-len=8192 --lora-modules a=/path1 b=/path2
```

`--tensor-parallel-size` と `--max-model-len` は等号形式に変換されますが、`--lora-modules` は複数の値を持つため、空白区切りのままです。

!!! note

    - **複数値のオプション**（`--lora-modules` など）は、選択した形式に関係なく常に空白区切りのままです。`--key=value` に変換すると、後続の値が通常は前の値を上書きするため、argparse の意味が変わります。
    - **ハイフン 1 つの短いオプション**（llama.cpp の `-n`、`-ngl`、`-m` など）はそのまま保持し、`--n` のようなハイフン 2 つの形式には変換しません。

## カスタム推論バックエンドの追加 { #add-custom-inference-backend }

1. 右上の「Add Backend」ボタンをクリックします。
2. フォームに入力するか YAML 定義を貼り付けて、カスタム推論バックエンドを追加できます。各項目の意味は、上のパラメーターの説明を参照してください。
3. 作成後にバックエンド名は変更できません。カスタムバックエンド名は「-custom」で終わる必要があります（フォームに事前入力済み）。
4. 「Save」をクリックして送信します。

カスタム推論バックエンドを追加する方法は、2 つあります。

- UI フォーム：**Model Service > Inference Backends** ページに移動し、**Add Custom Inference Backend** ボタンをクリックします。
- YAML 構成：バックエンド構成を含む YAML ファイルをインポートします。

### コミュニティ推論バックエンドの有効化 { #enable-community-inference-backend }

これらは実質的には「community」ソースラベルの付いたカスタムバックエンドであり、手動の構成なしですばやく作成できます。

1. Inference Backend ページで、右上の「Add Backend」ボタンをクリックします。
2. 「Community」を選択し、マーケットプレイスで利用可能なコミュニティバックエンドを閲覧します。
3. 使用するバックエンドを探し、カードの操作メニューから「Enable」をクリックします。
4. 有効にすると、モデルのデプロイで利用できます。無効にするには、Inference Backend ページからそのバックエンドを削除します。

## 推論バックエンドの編集とカスタムバージョンの追加 { #edit-inference-backend-or-add-custom-version }

1. Inference Backend ページで対象のバックエンドを探し、カード右上のドロップダウンメニューから「Edit」を選択します。
2. バックエンドのプロパティを変更するか、新しいバージョンを追加します。名前は変更できません。
3. 組み込みバックエンドに追加するカスタムバージョンは「-custom」で終わる必要があります（フォームに事前入力済み）。
4. 「Save」をクリックして送信します。

### 例：組み込み vLLM 推論バックエンドへのカスタムバージョンの追加 { #example-add-a-custom-version-to-the-built-in-vllm-inference-backend }

1. Inference Backend ページで vLLM バックエンドを探し、カード右上のドロップダウンメニューから「Edit」を選択します。
2. Version Configs 欄の「Add Version」をクリックします。
3. 次のように各項目を入力します。

- バージョン：`0.19.0`
- イメージ名：`vllm/vllm-openai:v0.19.0`
- フレームワーク：`cuda`
- イメージエントリーポイントの上書き：`vllm serve`
- 実行コマンド：`{{model_path}} --host {{worker_ip}} --port {{port}} --served-model-name {{model_name}}`

4. 「Save」をクリックして送信します。

!!! note

    vLLM は v0.11.1 以降で Docker イメージのエントリーポイントを変更しています。そのため vLLM v0.11.1 以降のカスタムバージョンを追加する際は、`Override Image Entrypoint` と `Execution Command` を指定する必要があります。指定しないとモデルの起動に失敗します。新しい `gpustack/runner` イメージを使用する場合、`Execution Command` の設定は不要です。

    vLLM や SGLang のように、`Override Image Entrypoint` で実行ファイルを指定するバックエンドでは、`Execution Command` には引数だけを含めてください。`Execution Command` に実行ファイル（`vllm serve` や `python -m vllm ...` など）を入れると、エントリーポイントの後ろに追加され、`vllm serve python -m vllm ...` のように接頭辞が重複します。

### 例：組み込み SGLang 推論バックエンドへのカスタムバージョンの追加 { #example-add-a-custom-version-to-the-built-in-sglang-inference-backend }

1. Inference Backend ページで SGLang バックエンドを探し、カード右上のドロップダウンメニューから「Edit」を選択します。
2. Version Configs 欄の「Add Version」をクリックします。
3. 次のように各項目を入力します。

- バージョン：`0.5.10`
- イメージ名：`lmsysorg/sglang:v0.5.10`
- フレームワーク：`cuda`
- イメージエントリーポイントの上書き：`sglang serve`
- 実行コマンド：`--model-path {{model_path}} --host {{worker_ip}} --port {{port}}`

4. 「Save」をクリックして送信します。

## カスタム推論バックエンドの削除 { #delete-custom-inference-backend }

1. Inference Backend ページで対象のバックエンドを探し、カード右上のドロップダウンメニューから「Delete」を選択します。
2. 組み込みバックエンドは削除できません。
3. 確認ダイアログで「Delete」をクリックします。

## 推論バックエンドのバージョン一覧 { #list-versions-of-inference-backend }

Inference Backend ページで、操作ボタン以外のバックエンドカードの任意の場所をクリックすると、モーダルが開き、組み込みおよび追加したカスタムバージョンをすべて閲覧できます。

## 柔軟なテスト用デプロイ { #flexible-testing-deployment }

バックエンドの定義を編集せずに、イメージと起動コマンドをすばやく検証または調整するためのモードです。

1. Deployments ページに移動し、「Deploy Model」ボタンをクリックして、任意のモデルソースを選択します。
2. Basic タブで「Backend」ドロップダウンを開き、「Built-in」セクションの「Custom」を選択します。
3. `image_name` と `run_command` の 2 つのフィールドが表示されます。このデプロイに限り、バックエンド構成を上書きします。
4. 残りの必須設定を確認し、デプロイを送信します。

## カスタム推論バックエンドの制限 { #limitations-of-custom-inference-backends }

カスタム推論バックエンドは、複数の Worker にまたがる分散推論には対応していません。
