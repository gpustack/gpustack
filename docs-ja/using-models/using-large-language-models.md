---
translation_status: translated
---

# 大規模言語モデルの利用 { #using-large-language-models }

**大規模言語モデル（LLM）**は、人間のようなテキストを理解・生成できる強力な AI モデルです。チャットボット、コンテンツ生成、コード補完などの用途で重要な役割を果たします。

このガイドでは、GPUStack に LLM をデプロイし、対話する方法を説明します。

## 前提条件 { #prerequisites }

開始前に、次の条件を満たしていることを確認します。

- 1 基以上の GPU と合計 30 GB 以上の VRAM を備えた Linux マシン。ここでは Linux のみに対応する vLLM バックエンドを使用します。
- モデルファイルを取得するために [Hugging Face](https://huggingface.co/) または [ModelScope](https://www.modelscope.cn/) にアクセスできること。
- GPUStack がインストール済みで、稼働していること。未導入の場合は[クイックスタート](../quickstart.md)をご覧ください。

## 手順 1：大規模言語モデルのデプロイ { #step-1-deploy-large-language-models }

### カタログからのデプロイ { #deploy-from-catalog }

カタログの大規模言語モデルには `LLM` カテゴリーが付いています。十分な GPU リソースがあり、バックエンドが環境に適合していれば、カタログから選択したモデルはデフォルト設定で動作します（たとえば vLLM には amd64 Linux ワーカーが必要です）。

ここでは `Qwen3 0.6B` を例に説明します。

カタログからモデルをデプロイするには、次の手順を実行します。

1. GPUStack UI の `Catalog` ページに移動します。
2. モデル一覧のドロップダウンで `LLM` を選択して絞り込みます。
3. モデルの説明、最大コンテキスト長、対応サイズを確認します。

![モデル一覧](../assets/using-models/using-large-language-models/model-list.png)

#### vLLM によるデプロイ { #deployment-using-vllm }

1. カタログから `Qwen3 0.6B` を選択します。
2. 必要に応じてモデル名と設定を変更します。
3. `Save` ボタンをクリックしてモデルをデプロイします。

![モデルのデプロイ](../assets/using-models/using-large-language-models/deploy-model.png)

デプロイ後は、`Deployments` ページで状態を確認し、実行が始まるのを待ちます。

## 手順 2：LLM によるテキスト生成 { #step-2-use-the-llms-for-text-generation }

1. GPUStack UI の `Playground` > `Chat` ページに移動します。
2. 右上の `Model` ドロップダウンで、デプロイしたモデルが選択されていることを確認します。
3. テキスト生成用のプロンプトを入力します。
4. 必要に応じて右側の `Parameters` を調整します。
5. `Submit` ボタンをクリックしてテキストを生成します。

生成された思考過程と結果が UI に表示されます。

![生成結果](../assets/using-models/using-large-language-models/playground-chat.png)

この手順で、GPUStack の LLM を使ったテキスト生成や自然言語タスクを実行できます。さまざまなプロンプトや設定を試し、LLM の機能を活用してください。
