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

## System One 判定 API { #system-one-decision-api }

判定モデル（`decision` カテゴリのデプロイ）は、テキストを生成する代わりに System One（Jev）判定契約に応答します。state と一連の questions を渡すと、補完ではなく型付きの回答を返します。GPUStack はこの契約を `/v1/systemone` パスで提供します。

`model` フィールドは必須で、`/v1/chat/completions` とまったく同じくモデルルート名を指定します。GPUStack はモデルを解決し、リクエスト本文の残りとレスポンス全体を判定エンジンへそのまま透過します。

### 使用方法 { #usage_3 }

#### cURL の例 { #curl-example_2 }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/systemone \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
      "model": "qwen3-decision-8b",
      "state": {
          "messages": [
              { "role": "user", "content": "Write a 200-word poem in Hungarian." }
          ]
      },
      "questions": {
          "model_selection": {
              "type": "choice",
              "instructions": "Pick the model best suited.",
              "criteria": {
                  "qwen3-8b": "Strong multilingual generation.",
                  "llama3-8b": "Faster, weaker multilingual."
              }
          }
      }
  }' | jq
```

出力例:

```json
{
  "model": "qwen3-decision-8b",
  "usage": {
    "input_tokens": 189,
    "output_tokens": 12
  },
  "answers": {
    "model_selection": {
      "type": "choice",
      "choice": "qwen3-8b",
      "confidence": 0.93,
      "probabilities": {
        "qwen3-8b": 0.93,
        "llama3-8b": 0.07
      }
    }
  }
}
```

`state` は判定の対象となる状況で、チャットメッセージのようなプレーンテキストまたは構造化 JSON です。`questions` の各エントリは質問 ID を質問定義に対応付けるもので、次の 3 種類が一般的です。

| 型 | 意味 | `criteria` | 回答フィールド |
| --- | --- | --- | --- |
| `choice` | 選択肢から 1 つ選ぶ | 選択肢名 → 説明 | `choice`——選ばれた選択肢名 |
| `score` | 順序付きレベルで state を評価 | レベル説明の順序付きリスト | `score`——レベルのインデックス |
| `noul` | 二値（はい/いいえ）の質問 | `true` / `false` 両側の任意の説明 | `noul`——真偽値 |

質問への回答は同じ ID で `answers` に返り、判定結果とともに `confidence` と `probabilities` が付きます。契約上の質問セットはオープンで、エンジンはさらに多くのフィールドを受け付けます。GPUStack が検証するのは `model` のみで、残りはそのまま転送します。判定モデルのデプロイについては、[判定モデルの利用](../using-models/using-decision-models.md)を参照してください。

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
