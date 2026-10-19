# モデルプロバイダーの管理 { #model-provider-management }

GPUStack は Higress の機能を使って、パブリック MaaS と連携します。`Model` - `Provider` ページで、パブリックなモデルサービスとの連携を管理できます。

現在、次のプロバイダーに対応しています。

- AI360
- Microsoft Azure
- Baichuan Intelligent Technology
- Baidu AI Cloud
- Amazon Bedrock
- Anthropic Claude
- Cloudflare AI
- Cohere
- Coze
- DeepL
- DeepSeek
- Dify
- Doubao
- Fireworks AI
- Galadriel
- Google Gemini
- Generic Provider
- GitHub Copilot
- xAI Grok
- Groq
- Tencent Hunyuan
- LongCat AI
- MiniMax
- Mistral AI
- Moonshot AI
- Ollama
- OpenAI
- OpenRouter
- Alibaba Qwen
- iFLYTEK SparkDesk
- StepFun
- Together AI
- NVIDIA Triton Inference Server
- 01 AI
- Zhipu AI

## プロバイダーの作成 { #create-provider }

1. `Providers` ページに移動します。
2. `Add Provider` をクリックします。
3. `Name`、`Type`、`API Key` などの必須項目を入力します。
4. `Add Model` をクリックし、このプロバイダーのモデルを 1 つ以上設定します。
5. `Save` ボタンをクリックします。

## Anthropic 互換エンドポイントの使用 { #use-an-anthropic-compatible-endpoint }

`Anthropic Claude` プロバイダーは、既定で `api.anthropic.com` に接続します。同じ API を提供するセルフホストまたはプロキシ経由のサービスを指定するには、`claudeCustomUrl` にポートとパスプレフィックスを含む絶対 `http(s)` URL を設定します。たとえば `http://192.168.50.14:8080` または `https://gateway.example.com/anthropic` です。

API のベース URL ではなく、サービスがマウントされているオリジンとプレフィックスを指定してください。Anthropic のパスは設定値に追加されるため、多くの Anthropic 互換サービスが `/v1` 付きのベース URL を案内していても、`claudeCustomUrl` の末尾にそれを付けないでください。`http://192.168.50.14:8080/v1` を設定すると、GPUStack は `/v1/v1/models` と `/v1/v1/messages` を参照します。この場合、推論からは接続できるエンドポイントに対して `Test Model` が失敗する形で問題が現れることがよくあります。これは、`/v1` を含む API の完全なベース URL を指定する `openaiCustomUrl` とは逆です。認証情報、クエリー、フラグメントを含む URL は拒否されます。API キーはプロバイダー専用のキー欄に入力してください。

このようなプロバイダーも、既定では OpenAI API を通じて公開されます。エンドポイントへ送る際にリクエストを Anthropic プロトコルへ変換し、返された応答を再変換します。変換せずに転送するには、プロバイダー構成に `protocol: original` を追加します。パススルーでは、プロンプトキャッシュ、thinking ブロック、tool-use ブロックなど、OpenAI スキーマでは表現できない要素を保持できます。ただし、そのプロバイダーは **Anthropic プロトコルのみ**を提供するようになります。`/v1/chat/completions` への OpenAI 形式のリクエストは変換されなくなるため、Anthropic のパス（`/v1/messages`）を使用してください。

## プロバイダーへのルートの追加 { #add-route-for-provider }

1. `Providers` ページに移動します。
2. ルートを作成するプロバイダーを探します。
3. そのプロバイダーの `Operations` にある `Add Route` をクリックします。
4. 必要に応じて、プロバイダー用のルートの `Name` と `Route Targets` を変更します。
5. `Save` ボタンをクリックします。

## プロバイダーの編集 { #edit-provider }

1. `Providers` ページに移動します。
2. 編集するプロバイダーを探します。
3. 必要に応じて名前、種類、API キーを変更し、モデル一覧でモデルを追加または削除します。
4. `Save` ボタンをクリックします。

## プロバイダーの削除 { #delete-provider }

1. `Providers` ページに移動します。
2. 削除するプロバイダーを探します。
3. `Operations` 列の `Delete` ボタンをクリックします。
4. 削除を確認します。
