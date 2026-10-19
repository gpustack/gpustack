---
translation_status: translated
---

# 推論 API { #inference-apis }

## OpenAI 互換 API { #openai-compatible-apis }

GPUStack は `/v1` エンドポイントで [OpenAI 互換 API](https://platform.openai.com/docs/api-reference) を提供します。

OpenAI 互換 API をサポートするアプリケーションやフレームワークで、接続先を GPUStack の OpenAI 互換エンドポイントに指定するだけで、デプロイしたモデルを統合・利用できます。

### 対応エンドポイント { #supported-endpoints }

次の API エンドポイントに対応しています。

- [x] [モデル一覧](https://platform.openai.com/docs/api-reference/models/list)
- [x] [補完の作成](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [チャット補完の作成](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [レスポンスの作成](https://platform.openai.com/docs/api-reference/responses/create)
- [x] [埋め込みの作成](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [画像の作成](https://platform.openai.com/docs/api-reference/images/create)
- [x] [画像編集の作成](https://platform.openai.com/docs/api-reference/images/createEdit)
- [x] [音声の作成](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [x] [文字起こしの作成](https://platform.openai.com/docs/api-reference/audio/createTranscription)

### 使用方法 { #usage }

#### cURL の例 { #curl-example }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'
```

## Anthropic 互換 API { #anthropic-compatible-apis }

GPUStack は Anthropic 互換の [`/v1/messages` API](https://platform.claude.com/docs/en/api/messages/create) を提供します。

### 使用方法 { #usage_1 }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "max_tokens": 1024
  }'
```

## Jina 互換 Rerank API { #jina-compatible-rerank-api }

検索拡張生成（RAG）におけるリランキングとは、検索した文書や知識ソースから、ユーザーへの提示や回答生成に利用する前に、最も関連性の高い情報を選ぶ処理です。

OpenAI 互換 API は `rerank` エンドポイントを**提供しません**。GPUStack はこれを補うため、`/v1/rerank` で [Jina 互換 Rerank API](https://jina.ai/reranker/) を提供します。

### 使用方法 { #usage_2 }

#### cURL の例 { #curl-example_1 }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/rerank \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -d '{
        "model": "bge-reranker-v2-m3",
        "query": "What is a panda?",
        "top_n": 3,
        "documents": [
            "hi",
            "it is a bear",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
        ]
    }' | jq
```

出力例：

```json
{
  "model": "bge-reranker-v2-m3",
  "object": "list",
  "results": [
    {
      "document": {
        "text": "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
      },
      "index": 2,
      "relevance_score": 1.951932668685913
    },
    {
      "document": {
        "text": "it is a bear"
      },
      "index": 1,
      "relevance_score": -3.7347371578216553
    },
    {
      "document": {
        "text": "hi"
      },
      "index": 0,
      "relevance_score": -6.157620906829834
    }
  ],
  "usage": {
    "prompt_tokens": 69,
    "total_tokens": 69
  }
}
```

## その他の API { #other-apis }

ほかの種類の API には、モデルのデプロイ時に **Generic Proxy** を有効にできます。

有効化後、対象モデルは次の 2 つの方法で指定できます。

**パスによる指定（推奨）**：モデルルート ID を `/model/proxy/<model_route_id>/<upstream-path>` のように URL に直接含めます。追加ヘッダーは不要です。

```bash
export GPUSTACK_API_KEY=your_api_key
# Assume the model route id is 42.
curl http://your_gpustack_server_url/model/proxy/42/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{"inputs": ["What is Deep Learning?", "Deep Learning is not..."]}'
```

**ヘッダーによる指定（非推奨）**：`/model/proxy/<upstream-path>` に送信し、`X-GPUStack-Model` ヘッダーまたは JSON 本文の `model` フィールドで対象モデルを指定します。

```bash
curl http://your_gpustack_server_url/model/proxy/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -H "X-GPUStack-Model: bge-m3" \
  -d '{"inputs": ["What is Deep Learning?", "Deep Learning is not..."]}'
```

ヘッダーによる指定は後方互換性のために保持されていますが、将来のリリースで削除される予定です。新しい連携にはパスによる指定を使用してください。

詳細は、[汎用プロキシの有効化](../user-guide/model-deployment-management.md#enable-generic-proxy)をご覧ください。
