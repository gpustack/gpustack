---
translation_status: translated
---

# 埋め込みモデルの利用 { #using-embedding-models }

**テキスト埋め込み**は、テキストの意味を捉えた数値表現で、コンピューターが異なるテキスト間の関係や類似性を理解するために使用します。テキストを連続空間上のベクトルに変換し、意味が似ているものを近くに配置します。自然言語処理、情報検索、推薦システムなどで広く活用されています。

このガイドでは、GPUStack に埋め込みモデルをデプロイし、テキスト埋め込みを生成する方法を説明します。

## 前提条件 { #prerequisites }

開始前に、次の条件を満たしていることを確認します。

- GPUStack がインストール済みで、稼働していること。未導入の場合は[クイックスタート](../quickstart.md)をご覧ください。
- モデルファイルをダウンロードするために Hugging Face にアクセスできること。

## 手順 1：モデルのデプロイ { #step-1-deploy-the-model }

カタログからモデルをデプロイするには、次の手順を実行します。

1. GPUStack UI の `Catalog` ページに移動します。
2. モデル一覧のドロップダウンで `Embedding` を選択して絞り込みます。
3. モデルの説明、最大コンテキスト長、対応サイズを確認します。

![モデルのデプロイ](../assets/using-models/using-embedding-models/deploy-model.png)

デプロイ後は、`Deployments` ページでモデルのデプロイ状態を確認できます。

![モデル一覧](../assets/using-models/using-embedding-models/model-list.png)

## 手順 2：API キーの生成 { #step-2-generate-an-api-key }

GPUStack API でテキスト埋め込みを生成するには API キーが必要です。

1. `Access Control` > `API Keys` ページに移動します。
2. `New API Key` ボタンをクリックします。
3. API キーの名前を入力し、`Save` ボタンをクリックします。
4. 生成された API キーをコピーします。表示されるのは 1 回だけなので、安全に保存してください。

## 手順 3：テキスト埋め込みの生成 { #step-3-generate-text-embeddings }

モデルをデプロイし API キーを用意したら、GPUStack API でテキスト埋め込みを生成できます。次は `curl` の例です。

```bash
export SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $SERVER_URL/v1/embeddings \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ${GPUSTACK_API_KEY}" \
-d '{
  "model": "qwen3-embedding-4b",
  "input": "The food was delicious and the waiter...",
  "encoding_format": "float"
}'
```

`<your-server-url>` を GPUStack サーバーの URL、`<your-api-key>` を前の手順で生成した API キーに置き換えます。

レスポンスの例：

```json
{
  "data": [
    {
      "embedding": [
        -0.012189436703920364, 0.016934078186750412, 0.003965042531490326,
        -0.03453584015369415, -0.07623119652271271, -0.007116147316992283,
        0.11278388649225235, 0.019714849069714546, 0.010370955802500248,
        -0.04219457507133484, -0.029902394860982895, 0.01122555136680603,
        0.022912170737981796, 0.031186765059828758, 0.006303929258137941,
        # ... additional values
      ],
      "index": 0,
      "object": "embedding"
    }
  ],
  "model": "qwen3-embedding-4b",
  "object": "list",
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 9,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```
