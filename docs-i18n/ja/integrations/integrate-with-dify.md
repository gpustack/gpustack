---
translation_status: translated
---

# Dify との連携 { #integrate-with-dify }

Dify を GPUStack と連携させると、ローカルにデプロイした LLM、埋め込み、リランキング、画像生成、音声認識、音声合成を利用できます。

## モデルのデプロイ { #deploying-models }

1. GPUStack UI の `Deployments` ページで `Deploy Model` をクリックし、必要なモデルをデプロイします。モデルの例：

- qwen3-8b
- qwen2.5-vl-3b-instruct
- bge-m3
- bge-reranker-v2-m3

![GPUStack のモデル](../assets/integrations/integration-gpustack-models.png)

2. モデルの操作メニューで `API Access Info` を開き、連携方法を確認します。

![GPUStack の API アクセス情報](../assets/integrations/integration-gpustack-api-access-info.png)

## API キーの作成 { #create-an-api-key }

1. `Access Control` > `API Keys` ページで `New API Key` をクリックします。

2. 名前を入力し、`Save` をクリックします。

3. API キーをコピーし、後で使えるよう保存します。

## Dify への GPUStack の統合 { #integrating-gpustack-into-dify }

1. Dify UI の右上で `PLUGINS` をクリックし、`Install from Marketplace` を選択します。GPUStack プラグインを検索してインストールします。

![Dify の GPUStack プラグイン導入](../assets/integrations/integration-dify-install-gpustack-plugin.png)

2. インストール後、`Settings > Model Provider > GPUStack` で `Add Model` を選択し、次を入力します。

- Model Type：モデルに応じた種類を選択します。

- Model Name：GPUStack にデプロイしたモデル名と一致させます。

- Server URL：`http://your-gpustack-url`。`localhost` はコンテナー内部を指すため使用しないでください。カスタムポートを使う場合はポートも含めます。Dify コンテナー内からアクセス可能かも確認してください（`curl` でテストできます）。

- API Key：前の手順でコピーした API キーを入力します。

`Save` をクリックしてモデルを追加します。

![Dify にモデルを追加](../assets/integrations/integration-dify-add-model.png)

必要に応じてほかのモデルも追加し、`System Model Settings` で追加したモデルを選択して保存します。

![Dify のシステムモデル設定](../assets/integrations/integration-dify-system-model-settings.png)

`Studio` と `Knowledge` でモデルを利用できます。簡単な例を示します。

1. `Knowledge` でナレッジを作成し、文書をアップロードします。

![Dify のナレッジ作成](../assets/integrations/integration-dify-create-knowledge.png)

2. Chunk Settings と Retrieval Settings を設定します。埋め込みモデルで文書の埋め込みを生成し、リランクモデルで検索結果の順位を調整します。

![Dify の埋め込み・リランクモデル設定](../assets/integrations/integration-dify-set-embedding-and-rerank-model.png)

3. 文書のインポート後、`Studio` でアプリケーションを作成します。作成済みのナレッジを追加し、チャットモデルを選択して対話します。

![Dify でのモデルとのチャット](../assets/integrations/integration-dify-chat-with-model.png)

4. モデルを `qwen2.5-vl-3b-instruct` に切り替え、追加済みのナレッジベースを削除し、`Vision` を有効にします。チャットに画像をアップロードすると、マルチモーダル入力を利用できます。

![Dify での視覚言語モデルとのチャット](../assets/integrations/integration-dify-chat-with-vlm.png)
