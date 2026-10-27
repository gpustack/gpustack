---
translation_status: translated
---

# 視覚言語モデルの利用 { #using-vision-language-models }

**視覚言語モデル**は、視覚（画像）と言語（テキスト）のデータを同時に処理でき、画像キャプションの生成や視覚的な質問応答など、多様な用途に使えます。このガイドでは、GPUStack に視覚言語モデル（VLM）をデプロイし、利用する方法を説明します。

GPUStack でのデプロイや利用の手順は、各モデルで共通しています。主な違いはデプロイ時に設定するパラメーターです。設定可能な項目は、[バックエンドパラメーター](../user-guide/built-in-inference-backends.md#parameters-reference_1)をご覧ください。

このガイドでは、次のモデルのデプロイを扱います。

- **Qwen3-VL**
- **Llama3.2-Vision**
- **Pixtral**
- **Phi3.5-Vision**

## 前提条件 { #prerequisites }

開始前に、次の条件を満たしていることを確認します。

- 1 基以上の GPU と合計 30 GB 以上の VRAM を備えた Linux マシン。ここでは Linux のみに対応する vLLM バックエンドを使用します。
- モデルファイルを取得するための Hugging Face へのアクセスと、[Hugging Face API キー](https://huggingface.co/settings/tokens)。
- 上記モデルへのアクセスが Hugging Face で許可されていること。[Llama3.2-VL](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) と [Pixtral](https://huggingface.co/mistralai/Pixtral-12B-2409) はアクセス制限付きのため、利用申請が必要です。

!!! note

    このガイドでは、H100（80GB）GPU を 1 基搭載した Ubuntu ノードを使用します。

## 手順 1：GPUStack のインストール { #step-1-install-gpustack }

[インストールドキュメント](../installation/requirements.md)に従って GPUStack をインストールします。

## 手順 2：GPUStack UI へのログイン { #step-2-log-in-to-gpustack-ui }

サーバーが起動したら、次のコマンドでデフォルトの管理者パスワードを取得します。

```bash
docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

ブラウザーで `http://your_host_ip` を開き、GPUStack UI にアクセスします。デフォルトのユーザー名 `admin` と取得したパスワードでログインします。

## 手順 3：vLLM による視覚言語モデルのデプロイ { #step-3-deploy-vision-language-models-with-vllm }

### Qwen3-VL のデプロイ { #deploy-qwen3-vl }

1. GPUStack UI の `Deployments` ページに移動します。
2. `Deploy Model` ボタンをクリックし、ドロップダウンから `Hugging Face` を選択します。
3. 検索バーで `Qwen/Qwen3-VL-4B-Instruct` を検索します。
4. `Save` ボタンをクリックします。十分な GPU リソースがあれば、デフォルト設定で動作します。

### Llama3.2-Vision のデプロイ { #deploy-llama32-vision }

1. GPUStack UI の `Deployments` ページに移動します。
2. `Deploy Model` ボタンをクリックし、ドロップダウンから `Hugging Face` を選択します。
3. 検索バーで `meta-llama/Llama-3.2-11B-Vision-Instruct` を検索します。
4. 設定の `Advanced` を展開し、`Backend Parameters` までスクロールします。
5. `Add Parameter` ボタンを複数回クリックし、次のパラメーターを追加します。

- `--enforce-eager`
- `--max-num-seqs=16`
- `--max-model-len=8192`

6. `Save` ボタンをクリックします。

### Pixtral のデプロイ { #deploy-pixtral }

1. GPUStack UI の `Deployments` ページに移動します。
2. `Deploy Model` ボタンをクリックし、ドロップダウンから `Hugging Face` を選択します。
3. 検索バーで `mistralai/Pixtral-12B-2409` を検索します。
4. 設定の `Advanced` を展開し、`Backend Parameters` までスクロールします。
5. `Add Parameter` ボタンを複数回クリックし、次のパラメーターを追加します。

- `--tokenizer-mode=mistral`
- `--limit-mm-per-prompt=image=4`

6. `Save` ボタンをクリックします。

### Phi3.5-Vision のデプロイ { #deploy-phi35-vision }

1. GPUStack UI の `Deployments` ページに移動します。
2. `Deploy Model` ボタンをクリックし、ドロップダウンから `Hugging Face` を選択します。
3. 検索バーで `microsoft/Phi-3.5-vision-instruct` を検索します。
4. 設定の `Advanced` を展開し、`Backend Parameters` までスクロールします。
5. `Add Parameter` ボタンをクリックし、次のパラメーターを追加します。

- `--trust-remote-code`

6. `Save` ボタンをクリックします。

## 手順 4：視覚言語モデルの利用 { #step-4-interact-with-vision-language-models }

1. GPUStack UI の `Chat` ページに移動します。
2. 右上のドロップダウンからデプロイしたモデルを選択します。
3. 入力欄の上の `Upload Image` ボタンをクリックし、画像をアップロードします。
4. 入力欄にプロンプトを入力します。例：「画像を説明してください。」
5. `Submit` ボタンをクリックし、出力を生成します。

![視覚言語モデルのプレイグラウンド](../assets/using-models/using-vision-language-models/playground-vl.png)
