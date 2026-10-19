---
translation_status: translated
---

# ツール呼び出しを使った推論 { #inference-with-tool-calling }

ツール呼び出しにより、モデルを外部のツールやシステムに接続できます。AI アシスタントへの機能追加や、アプリケーションとモデルの深い連携などに役立ちます。

このチュートリアルでは、GPUStack でツール呼び出しを設定・利用し、AI の機能を拡張する方法を説明します。

!!! note

    1. ツール呼び出しは、[vLLM](../user-guide/built-in-inference-backends.md#vllm) と [SGLang](../user-guide/built-in-inference-backends.md#sglang) 推論バックエンドに対応しています。
    2. ツール呼び出しは基本的にプロンプトエンジニアリングで実現します。この機能には、対応テンプレートを学習済みのモデルが必要なため、すべての LLM が対応するわけではありません。

## 前提条件 { #prerequisites }

次の条件を満たしていることを確認します。

- GPUStack がインストール済みで稼働していること。
- GPU を搭載した Linux ワーカーノードが利用できること。この例では [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) を使い、18GB 以上の VRAM が必要です。
- モデルファイルをダウンロードするために Hugging Face にアクセスできること。

## 手順 1：モデルのデプロイ { #step-1-deploy-the-model }

### カタログからのデプロイ { #deploy-from-catalog }

ツール呼び出し対応の LLM には、カタログで `tools` 能力が表示されます。このモデルをカタログから選ぶと、デフォルトでツール呼び出しが有効になります。

### llama-box によるカスタムデプロイの例 { #example-of-custom-deployment-using-llama-box }

llama-box で GGUF モデルをデプロイする場合、対応するモデルではデフォルトでツール呼び出しが有効です。

1. GPUStack UI の `Deployments` ページで `Deploy Model` をクリックし、ソースとして `Hugging Face` を選択します。
2. `GGUF` にチェックを入れ、GGUF 形式に絞り込みます。
3. 検索バーで `Qwen/Qwen2.5-7B-Instruct-GGUF` を探します。
4. `Save` をクリックしてデプロイします。

![GGUF モデルのデプロイ](../assets/tutorials/inference-with-tool-calling/deploy-model-gguf.png)

### vLLM によるカスタムデプロイの例 { #example-of-custom-deployment-using-vllm }

vLLM でデプロイする場合は、追加パラメーターでツール呼び出しを有効にします。

1. GPUStack UI の `Deployments` ページで `Deploy Model` をクリックし、ソースとして `Hugging Face` を選択します。
2. 検索バーで `Qwen/Qwen2.5-7B-Instruct` を探します。
3. 設定の `Advanced` を展開し、`Backend Parameters` までスクロールします。
4. `Add Parameter` をクリックし、次のパラメーターを追加します。

- `--enable-auto-tool-choice`
- `--tool-call-parser=hermes`

5. `Save` をクリックしてモデルをデプロイします。

![モデルのデプロイ](../assets/tutorials/inference-with-tool-calling/deploy-model.png)

デプロイ後は、`Deployments` ページでモデルの状態を確認できます。

## 手順 2：API キーの生成 { #step-2-generate-an-api-key }

GPUStack API でモデルを利用するために、API キーを生成します。

1. `Access Control` > `API Keys` ページに移動します。
2. `New API Key` をクリックします。
3. API キーの名前を入力し、`Save` をクリックします。
4. 生成した API キーを、後で使用するためにコピーします。

## 手順 3：推論の実行 { #step-3-do-inference }

モデルをデプロイし API キーを用意したら、GPUStack API で呼び出せます。次は `curl` の例です（`<your-server-url>` をサーバー URL、`<your-api-key>` を前の手順の API キーに置き換えます）。

```bash
export GPUSTACK_SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $GPUSTACK_SERVER_URL/v1-openai/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $GPUSTACK_API_KEY" \
-d '{
  "model": "qwen2.5-7b-instruct",
  "messages": [
    {
      "role": "user",
      "content": "What'\''s the weather like in Boston today?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}'
```

レスポンスの例：

```json
{
  "model": "qwen2.5-7b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "chatcmpl-tool-b99d32848b324eaea4bac5a5830d00b8",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\": \"Boston, MA\", \"unit\": \"fahrenheit\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 212,
    "total_tokens": 242,
    "completion_tokens": 30
  }
}
```
