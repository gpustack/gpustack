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

### 動画生成 { #video-generation }

動画生成には非同期タスクを使用します。SGLang Diffusion や vLLM-Omni など、OpenAI 動画プロトコルを実装した推論エンジンが対象です。

| メソッド | エンドポイント | 用途 |
| --- | --- | --- |
| POST | `/v1/videos` | `model`、`prompt`、`seconds`、`size`、オプションの参照画像などを multipart フォームで送信して動画タスクを作成します。 |
| GET | `/v1/videos/{id}` | タスクの状態と、エンジンが提供する進捗を取得します。 |
| GET | `/v1/videos/{id}/content` | 生成が完了した動画をダウンロードします。 |

動画生成のデプロイは `video` モデルカテゴリーを使用します。`GET /v1/models?categories=video` で利用可能な動画ルートを取得できます。動画理解モデルは引き続き `llm` カテゴリーに属し、Chat Completions API を使用します。

新しいカテゴリーを使用する前に Server と Worker を一緒にアップグレードしてください。既知の Diffusers 動画パイプラインは、デプロイ名に依存せず `model_index.json` から識別されます。バックグラウンドの調整処理は、`image` に分類された既存のローカル SGLang/vLLM デプロイも確認し、動画モデルと確認できたものを更新します。自動作成されたルートのカテゴリーは通常のモデル同期で更新され、手動設定されたルートは保持されます。このカテゴリー更新はインスタンスの再起動を要求しません。取得できないメタデータは再試行され、不明なパイプラインは元のカテゴリーを維持します。カスタムパイプラインでは、デプロイのモデルカテゴリー設定で `Video` を明示的に選択してください。

カテゴリー自体は参照画像や参照動画への対応を示しません。入力への対応は選択したモデルに依存します。このカテゴリー変更には外部プロバイダーの動画タスク処理は含まれません。

タスクを作成した後、`completed` または `failed` になるまで状態をポーリングし、その後コンテンツをダウンロードします。進捗は推論エンジンが提供し、限られた段階でしか更新されない場合があります。返されたタスク ID は不透明な文字列として扱ってください。GPUStack は ID を作成ユーザー、組織コンテキスト、推論インスタンスに結び付けます。状態確認とダウンロードには同じ認証主体と組織コンテキストを使用してください。

HTTP リクエストの中断やクライアント側のポーリング停止は、受け付け済みの生成タスクをキャンセルしません。GPUStack は現在、エンジンへのキャンセル要求を転送する `DELETE /v1/videos/{id}` を公開していません。

タスクハンドルの有効期間は 24 時間ですが、エンジンがそれより早く出力を削除する場合があります。タスクの状態と生成ファイルは作成元のエンジンプロセスに属するため、別のレプリカへ切り替えることはできません。そのインスタンスを再起動すると、タスクを利用できなくなる場合があります。`/v1/videos/sync` を実装したエンジンでは、このオプションの拡張を引き続き使用できます。

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
