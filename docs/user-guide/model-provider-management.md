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

## Create Provider

1. Go to `Providers` Page.
2. Click the `Add Provider`.
3. Fill the required options like `Name`, `Type`, `API Key`.
4. Click `Add Model` to configure at least one model for this provider.
5. Click the `Save` button.

## Use an Anthropic-Compatible Endpoint

An `Anthropic Claude` provider talks to `api.anthropic.com` by default. To point it at a
self-hosted or proxied service that speaks the same API, set `claudeCustomUrl` to an
absolute `http(s)` URL, including the port and any path prefix — for example
`http://192.168.50.14:8080` or `https://gateway.example.com/anthropic`.

Give the origin and the prefix the service is mounted under, **not** its API base: the
Anthropic paths are appended to what you set, so `claudeCustomUrl` should not end in
`/v1` even though most Anthropic-compatible services advertise a base URL that does.
Setting `http://192.168.50.14:8080/v1` makes GPUStack look for `/v1/v1/models` and
`/v1/v1/messages`, which usually shows up as `Test Model` failing against an endpoint
that inference itself reaches. Note this is the opposite of `openaiCustomUrl`, which is
the full API base and does include `/v1`. Credentials, a query, or a fragment in the URL
are rejected — the API key belongs in the provider's own key field.

Such a provider is still exposed through the OpenAI API by default: requests are
converted to the Anthropic protocol on the way to the endpoint and back on the way
home. To forward them unchanged instead, add `protocol: original` to the provider
config. Passthrough preserves everything the OpenAI schema has no place for (prompt
caching, thinking blocks, tool-use blocks), but the provider then serves the
**Anthropic protocol only** — OpenAI-style requests to `/v1/chat/completions` are no
longer translated for it, so use the Anthropic paths (`/v1/messages`) with it.

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
