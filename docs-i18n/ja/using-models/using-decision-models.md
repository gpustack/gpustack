---
translation_status: translated
---

# 判定モデルの利用 { #using-decision-models }

**判定モデル（Decision Models）**は、テキストを生成する代わりに構造化された型付きの質問に答えます。判定の対象となる *state* と一連の *questions* を渡すと、判定モデルは質問ごとに型付きの判定結果（選ばれた選択肢、レベルのスコア、はい/いいえの答え。多くの場合は信頼度付き）を返します。完全な生成が無駄になる場面、たとえばモデルの選択とルーティング、分類、軽量な判定タスクで利用されます。

GPUStack では、判定モデルは System One（Jev）判定契約により `/v1/systemone` エンドポイントで提供されます。このガイドでは、GPUStack での判定モデルのデプロイと利用を説明します。

## 前提条件 { #prerequisites }

始める前に、以下を確認してください。

- GPUStack がインストールされ稼働していること。未インストールの場合は[クイックスタートガイド](../quickstart.md)を参照してください。
- モデルファイルのダウンロードのため Hugging Face にアクセスできること。

## 手順 1: モデルをデプロイする { #step-1-deploy-the-model }

GPUStack は現在、判定 API を提供する 2 種類のデプロイをサポートしています。

### vLLM: テキスト生成モデルを転用する { #vllm-repurpose-a-text-generation-model }

vLLM `0.32.0` 以降は、通常のサービングポートで System One 判定契約を提供します。任意のテキスト生成モデルをいつも通り `vLLM` バックエンドでデプロイし、その後モデルのカテゴリを `Decision` に変更します。このカテゴリこそが、そのデプロイが判定サービスであることを GPUStack に伝えます。

1. `vLLM` バックエンド（バージョン `0.32.0` 以上）でテキスト生成モデルをデプロイします。[モデルデプロイ管理](../user-guide/model-deployment-management.md)を参照してください。
2. モデルを編集し、モデル設定で `Category` を `Decision` に設定します。

![モデルカテゴリ](../assets/using-models/using-decision-models/model-category.png)

### llama.cpp: Laya 判定 GGUF を提供する { #llamacpp-serve-a-laya-decision-gguf }

Laya 判定 GGUF ファイルは、汎用バックボーン（modern-bert、qwen、llama）の上に判定メタデータキーを保持しています。他の GGUF モデルと同じようにデプロイすると llama.cpp で提供されます。GPUStack はデプロイ時に判定メタデータを検出し、モデルを自動的に `decision` カテゴリに分類します。このキーがなければ、同じファイルはリランカーや埋め込みモデルと誤認されます。

![モデル一覧](../assets/using-models/using-decision-models/model-list.png)

どちらの方法でも、デプロイが起動すると `Models` ページの `Category` 列に `Decision` と表示されます。カテゴリは手動で修正することもできます。カテゴリ値は検証されるため、誤入力されたカテゴリは黙って保存されるのではなく拒否されます。

## 手順 2: 判定を依頼する { #step-2-ask-a-decision }

判定の依頼は UI でも API でも行えます。

### UI で行う { #via-the-ui }

1. `Playground` > `Decision` に移動するか、`Models` ページのモデル行にある `Open in Playground` をクリックします。
2. モデルセレクターで判定モデルを選択します。リストに表示されるのは `decision` カテゴリのモデルだけであり、モデルが判定の唯一のパラメーターです。
3. `State` エディターに判定の対象となる状況を入力します（チャットメッセージのサンプルが初期値で、プレーンテキストも構造化 JSON も入力できます）。`Questions` エディターには質問を入力します（`choice` 質問が初期値で、`+ choice`、`+ score`、`+ noul` ボタンで追加できます）。
4. `Run` をクリック（または `Ctrl`/`Cmd` + `Enter` を押下）し、型付きの回答を確認します。

エディターと回答カードの詳細は[判定 Playground](../user-guide/playground/decision.md)を参照してください。

### API で行う { #via-the-api }

`POST /v1/systemone` でモデルに state と questions を送ります（`API Keys` ページで作成した API キーで認証します）。`model` フィールドは必須で、OpenAI 互換エンドポイントと同様に、デプロイ済みモデルの名前（またはモデルルート名）を指定します。

以下は `curl` を使ったスクリプトの例です。

```bash
export SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $SERVER_URL/v1/systemone \
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

`<your-server-url>` を GPUStack サーバーの URL に、`<your-api-key>` を `API Keys` ページで作成した API キーに置き換えてください。

応答の例:

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

`questions` の各エントリは質問 ID を質問定義に対応付けるもので、次の 3 種類が一般的です。質問への回答は同じ ID で `answers` に返ります。

| 型 | 意味 | `criteria` | 回答フィールド |
| --- | --- | --- | --- |
| `choice` | 選択肢から 1 つ選ぶ | 選択肢名 → 説明 | `choice`——選ばれた選択肢名 |
| `score` | 順序付きレベルで state を評価する | レベル説明の順序付きリスト | `score`——レベルのインデックス |
| `noul` | 二値（はい/いいえ）の質問 | `true` / `false` 両側の任意の説明 | `noul`——真偽値 |

GPUStack は `model` フィールドを解決し、リクエスト本文の残りとレスポンス全体を判定エンジンへそのまま透過します。契約上の質問セットはオープンであり、エンジンはさらに多くのフィールドを受け付けます。

## 次のステップ { #next-steps }

- デプロイ済みの判定モデルは、ルートの[判定サービスルーティング](../user-guide/model-route-management.md#decision-service-routing)ポリシーの判定サービスとして、リクエストごとにルートターゲットを評価できます。
- エンドポイントの完全なリファレンスは[推論 API](../integrations/inference-apis.md)を参照してください。
