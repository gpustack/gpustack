---
translation_status: translated
---

# LiteLLM との連携 { #integrate-with-litellm }

LiteLLM を GPUStack と連携させると、ローカルにデプロイした LLM、埋め込み、リランキング、音声認識、音声合成を、従業員向けの統一された OpenAI 互換・Anthropic API ゲートウェイに集約できます。

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

## LiteLLM への GPUStack の統合 { #integrating-gpustack-into-litellm }

1. LiteLLM の管理 UI を開きます。

```
http://your-litellm-url:4000/ui
```

![LiteLLM ダッシュボード](../assets/integrations/litellm-1.png)

2. models+endpoints メニューを開き、モデルを追加します。

![LiteLLM のモデル追加](../assets/integrations/litellm-2.png)
![LiteLLM のモデル設定](../assets/integrations/litellm-3.png)

3. LiteLLM のプレイグラウンドでテストします。

![LiteLLM のチャット](../assets/integrations/litellm-4.png)
