---
translation_status: translated
---

# NewAPI との連携 { #integrate-with-newapi }

NewAPI を GPUStack と連携させると、ローカルにデプロイした LLM、埋め込み、リランキング、音声認識、音声合成を、従業員向けの統一された OpenAI 互換 API ゲートウェイに集約できます。

## GPUStack でのモデルのデプロイ { #deploying-models-in-gpustack }

1. GPUStack UI の `Deployments` ページで `Deploy Model` をクリックし、必要なモデルをデプロイします。モデルの例：

- qwen3-8b
- qwen2.5-vl-3b-instruct
- bge-m3
- bge-reranker-v2-m3

![GPUStack のモデル](../assets/integrations/integration-gpustack-models.png)

2. モデルの操作メニューで `API Access Info` を開き、連携方法を確認します。

![GPUStack の API アクセス情報](../assets/integrations/integration-gpustack-api-access-info.png)

## GPUStack での API キーの作成 { #create-an-api-key-in-gpustack }

1. GPUStack の `Access Control` > `API Keys` ページで `New API Key` をクリックします。

2. 名前を入力し、`Save` をクリックします。

3. API キーをコピーし、後で使えるよう保存します。

## NewAPI への GPUStack の統合 { #integrating-gpustack-into-newapi }

1. NewAPI 管理コンソールの左サイドバーで `Channels` を開き、`Add Channel` をクリックします。

2. チャネルのパラメーターを次のように設定します。

- **Type**：`OpenAI` または `Custom` を選択します（GPUStack は標準の OpenAI 互換および Anthropic エンドポイントを提供します）。

- **Name**：チャネル名を入力します。通常は `qwen3-8b` のようなモデル ID と同じ名前です。

- **Base URL (Base URL)**：`http://your-gpustack-url`。OpenAI 互換と Anthropic のどちらも GPUStack のアクセス URL またはドメイン名を指定します（NewAPI が自動追加するため v1 は除きます）。NewAPI を Docker で実行する場合、`localhost` はコンテナー内部を指すため使用しないでください。GPUStack サーバーで TLS を有効にした場合は https も使用できます。

- **Key (密钥)**：前の手順でコピーした GPUStack API キーを入力します。

- **Models (模型)**：GPUStack にデプロイしたモデル名（`qwen3-8b`、`bge-m3`、`bge-reranker-v2-m3` など）を正確に入力・選択します。デプロイメント名との一致を確認してください。

![NewAPI のチャネル設定](../assets/integrations/newapi1.png)
![NewAPI のチャネル設定](../assets/integrations/newapi2.png)
![NewAPI のチャネル設定](../assets/integrations/newapi3.png)
![NewAPI のチャネル設定](../assets/integrations/newapi4.png)

3. モデルの料金を設定します。

![NewAPI のチャネル設定](../assets/integrations/newapi5.png)
![NewAPI のチャネル設定](../assets/integrations/newapi6.png)
![NewAPI のチャネル設定](../assets/integrations/newapi7.png)

4. チャネル一覧の `Test` をクリックして、NewAPI と GPUStack の接続を確認できます。

![NewAPI のチャネル設定](../assets/integrations/newapi8.png)

---

## NewAPI での連携モデルの検証と利用 { #verifying-and-using-the-integrated-models-in-newapi }

1. NewAPI の `api keys` ページで `create api key` をクリックします。

* 注意：API キーにはチャネル設定と同じグループを選択してください。一致しないとモデルにアクセスできません。

2. 生成された NewAPI キー（`sk-` で始まる）をコピーします。

3. NewAPI の統一エンドポイントから、GPUStack 上のすべてのモデルにアクセスできます。

- **統一 Base URL**：`http://your-newapi-url/v1`
- **統一 API キー**：`sk-xxxxxx`（NewAPI キー）

`curl` によるリクエスト例：

```bash
curl http://your-newapi-url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-newapi-token" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
