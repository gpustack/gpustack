---
translation_status: translated
---

# 推論 API { #inference-apis }

## OpenAI 互換 API { #openai-compatible-apis }

GPUStack は `/v1` エンドポイントで [OpenAI 互換 API](https://platform.openai.com/docs/api-reference) を提供します。

OpenAI 互換 API をサポートするアプリケーションやフレームワークで、接続先を GPUStack の OpenAI 互換エンドポイントに指定するだけで、デプロイしたモデルを統合・利用できます。

GPUStack Server が音声の文字起こしや画像編集などの multipart 推論リクエストをプロキシする際、共通パーサーはテキストのみのフォームも multipart のまま扱い、同じ名前のフィールドを保持します。

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

SGLang Diffusion や vLLM-Omni など、OpenAI 動画プロトコルを実装した推論エンジンでは非同期の動画タスクを利用できます。

| メソッド | エンドポイント | 用途 |
| --- | --- | --- |
| POST | `/v1/videos` | `application/json` または `multipart/form-data` で非同期タスクを作成します。`model` にモデルルート、`prompt` にプロンプトを指定します。 |
| POST | `/v1/videos/sync` | 同期生成を実装しているエンジン向けの multipart 互換エンドポイントです。`model` にモデルルートを指定し、エンジンが対応するネイティブパラメーターを送ると、同じレスポンスで生成済みの動画を取得できます。 |
| GET | `/v1/videos/{video_id}` | タスクの状態と、エンジンが提供する進捗を取得します。 |
| GET | `/v1/videos/{video_id}/content` | 生成が完了した動画をダウンロードします。 |
| DELETE | `/v1/videos/{video_id}` | タスクを作成したインスタンスでキャンセルまたは削除します。キャンセルとファイルの削除は推論エンジンが処理します。 |

動画生成のデプロイは `video` モデルカテゴリーを使用します。`GET /v1/models?categories=video` で利用可能な動画ルートを取得できます。動画理解モデルは引き続き `llm` カテゴリーに属し、Chat Completions API を使用します。

新しいカテゴリーを使用する前に Server と Worker を一緒に更新してください。Auto デプロイはデプロイ名に依存せず、`model_index.json` から既知の Diffusers 動画パイプラインを識別します。既存の `image` を含む明示的なカテゴリー選択は保持されます。バックグラウンド処理は `video` デプロイの機能メタデータを更新し、取得失敗時はバックオフして再試行します。既存のデプロイや未知のパイプラインには、必要に応じて `Video` を明示的に選択してください。

カテゴリー自体は参照画像や参照動画への対応を示しません。入力への対応は選択したモデルに依存します。

非同期動画エンドポイント（`/v1/videos`、`/v1/videos/{video_id}`、`/v1/videos/{video_id}/content`）と同期エンドポイント（`/v1/videos/sync`）は、いずれも Higress のモデル Ingress 経由でルーティングされます。利用するには Higress ゲートウェイモードを有効にする必要があります。GPUStack 管理インスタンスと OpenAI 動画プロトコルに対応する外部プロバイダーの双方をモデルルート経由で提供できます。

状態確認、コンテンツのダウンロード、タスクの削除には、GPUStack が返した完全な動画 ID を使用します。作成レスポンスが Higress を通過する際、ゲートウェイはエンジンのネイティブタスク ID を自己ルーティング ID としてラップします（管理インスタンスは `gpustack_<route_id>-<instance_id>_<origin_video_id>`、外部プロバイダーは `gpustack_<route_id>-provider-<provider_id>_<origin_video_id>`）。後続リクエストはルートを解決し、ゲートウェイの `ext-auth` で認証・認可したうえで作成元のインスタンスまたはプロバイダーへ固定転送します。インスタンスが削除された場合や上流エンジンがタスクを失った場合は、HTTP 404 が返ることがあります。

MiniMax H3 のデプロイでは、API 呼び出し元は対象エンジンのネイティブパラメーターを直接送信します。SGLang Diffusion は `task`、`conditions`、`target` を使用し、vLLM-Omni は `extra_params` とエンジン固有の参照フィールド（`image_reference`、`video_reference`、`audio_reference`、`frame_indices`）を使用します。Video Playground は `POST /v1/videos` を送信する前に、画面の入力内容を選択したエンジンのネイティブパラメーターへ変換します。

Video Playground のクライアント側アップロード上限は、画像 1 MiB、動画 2 MiB です。API リクエストは選択した推論エンジンの制約に従います。HTTP(S) 参照は推論エンジンが取得して検証し、プライベートアドレスにアクセスできるかどうかはエンジンのネットワークアクセスとポリシーによって決まります。

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
