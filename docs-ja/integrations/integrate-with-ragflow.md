---
translation_status: translated
---

# RAGFlow との連携 { #integrate-with-ragflow }

RAGFlow を GPUStack と連携させると、ローカルにデプロイした LLM、埋め込み、リランキング、音声認識、音声合成を利用できます。

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

## RAGFlow への GPUStack の統合 { #integrating-gpustack-into-ragflow }

1. RAGFlow UI の右上のアバターをクリックし、`Model Providers > GPUStack` で `Add the model` を選択して次を入力します。

- Model type：モデルに応じた種類を選択します。

- Model name：GPUStack にデプロイしたモデル名と一致させます。

- Base URL：`http://your-gpustack-url/v1`。余分なパスを含めず、コンテナー内部を指す `localhost` は使用しないでください。カスタムポートを使う場合はポートも含めます。RAGFlow コンテナー内からアクセス可能かも確認してください（`curl` でテストできます）。

- API-Key：前の手順でコピーした API キーを入力します。

- Max Tokens：現在のモデル設定で対応する最大トークン数を入力します。

`OK` をクリックしてモデルを追加します。

![RAGFlow のモデル追加](../assets/integrations/integration-ragflow-add-model.png)

2. 必要に応じてほかのモデルも追加し、`Set default models` で追加したモデルを選択して保存します。

![RAGFlow のデフォルトモデル設定](../assets/integrations/integration-ragflow-set-default-models.png)

`Chat` と `Knowledge Base` でモデルを利用できます。簡単な例を示します。

1. `Knowledge base` で新しいナレッジベースを作成し、ファイルを追加します。

![RAGFlow のナレッジベース作成](../assets/integrations/integration-ragflow-create-knowledge-base.png)

2. `Retrieval testing` でリランクモデルを `bge-reranker-v2-m3` に設定します。

![RAGFlow のリランクモデル設定](../assets/integrations/integration-ragflow-set-rerank-model.png)

3. `Chat` でアシスタントを作成し、作成済みのナレッジベースを関連付け、チャットモデルを選択します。

![RAGFlow のアシスタント作成](../assets/integrations/integration-ragflow-create-assistant.png)

4. チャットセッションを作成すると、モデルと対話し、ナレッジベースを検索できます。

![RAGFlow のモデルとのチャット](../assets/integrations/integration-ragflow-chat-with-model.png)

5. アシスタントを編集し、モデルを `qwen2.5-vl-3b-instruct` に切り替えます。保存後に新しいチャットを作成し、画像をアップロードするとマルチモーダル入力を利用できます。

![RAGFlow の視覚言語モデルとのチャット](../assets/integrations/integration-ragflow-chat-with-vlm.png)
