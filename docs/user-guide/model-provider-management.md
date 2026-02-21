# Model Provider Management

GPUStack leverages Higress capabilities to provide Public MaaS integration. On the `Model` - `Provider` page, you can manage the integration of public model services.

The currently supported providers are:

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

## Create Provider

1. Go to `Providers` Page.
2. Click the `Add Provider`.
3. Fill the required options like `Name`, `Type`, `API Key`.
4. Click `Add Model` to configure at lease one model for this provider.
5. Click the `Save` button.

## Add Route for Provider

1. Go to `Providers` Page.
2. Find the Provider for which you want to create a route.
3. Click `Add Route` in the `Operations` of this provider.
4. Modify the route `Name` and `Route Targets` for the provider as needed.
5. Click the `Save` button.

## Edit Provider

1. Go to `Providers` Page.
2. Find the Provider you want to edit.
3. Modify the name, type, API key attributes as needed. Add/remove models from model list as needed.
4. Click the `Save` button.

## Delete Provider

1. Go to `Providers` Page.
2. Find the Provider you want to delete.
3. Click the `Delete` button in the `Operations` column.
4. Confirm the deletion.
