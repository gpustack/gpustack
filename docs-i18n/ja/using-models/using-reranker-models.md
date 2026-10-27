---
translation_status: translated
---

# リランカーモデルの利用 { #using-reranker-models }

**リランカーモデル**は、クエリーとの関連性に基づいて一覧の順位を改善する専用モデルです。情報検索や検索システムの初期結果を調整し、ユーザーの意図に合う可能性が高い項目を優先します。最初の文書一覧を並べ替え、検索エンジン、推薦システム、質問応答などの精度を向上させます。

このガイドでは、GPUStack でリランカーモデルをデプロイして利用する方法を説明します。

## 前提条件 { #prerequisites }

開始前に、次の条件を満たしていることを確認します。

- GPUStack がインストール済みで、稼働していること。未導入の場合は[クイックスタート](../quickstart.md)をご覧ください。
- モデルファイルをダウンロードするために Hugging Face にアクセスできること。

## 手順 1：モデルのデプロイ { #step-1-deploy-the-model }

カタログからモデルをデプロイするには、次の手順を実行します。

1. GPUStack UI の `Catalog` ページに移動します。
2. モデル一覧のドロップダウンで `Reranker` を選択して絞り込みます。
3. モデルの説明、最大コンテキスト長、対応サイズを確認します。

![モデルのデプロイ](../assets/using-models/using-reranker-models/deploy-model.png)

デプロイ後は、`Deployments` ページでモデルのデプロイ状態を確認できます。

![モデル一覧](../assets/using-models/using-reranker-models/model-list.png)

## 手順 2：API キーの生成 { #step-2-generate-an-api-key }

GPUStack API でモデルを利用します。まず API キーを生成します。

1. `Access Control` > `API Keys` ページに移動します。
2. `New API Key` ボタンをクリックします。
3. API キーの名前を入力し、`Save` ボタンをクリックします。
4. 生成された API キーをコピーします。表示されるのは 1 回だけなので、安全に保存してください。

## 手順 3：リランキング { #step-3-reranking }

モデルをデプロイし API キーを用意したら、GPUStack API で文書の一覧を並べ替えられます。次は `curl` の例です。

```bash
export SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $SERVER_URL/v1/rerank \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -d '{
        "model": "qwen3-reranker-4b",
        "query": "What is a panda?",
        "top_n": 3,
        "documents": [
            "hi",
            "it is a bear",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
        ]
    }' | jq
```

`<your-server-url>` を GPUStack サーバーの URL、`<your-api-key>` を前の手順で生成した API キーに置き換えます。

レスポンスの例：

```json
{
  "model": "qwen3-reranker-4b",
  "object": "list",
  "results": [
    {
      "index": 0,
      "document": {
        "text": "hi",
        "multi_modal": null
      },
      "relevance_score": 0.9996911287307739
    },
    {
      "index": 2,
      "document": {
        "text": "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        "multi_modal": null
      },
      "relevance_score": 0.8206241726875305
    },
    {
      "index": 1,
      "document": {
        "text": "it is a bear",
        "multi_modal": null
      },
      "relevance_score": 0.7244728803634644
    }
  ],
  "usage": {
    "total_tokens": 51
  }
}
```
