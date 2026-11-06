---
translation_status: translated
---

# 模型提供商管理 { #model-provider-management }

GPUStack 利用 Higress 的能力集成公共 MaaS 服务。在 `Model` - `Provider` 页面中，可以管理公共模型服务的集成。

目前支持的提供商包括：

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

## 创建提供商 { #create-provider }

1. 进入 `Providers` 页面。
2. 单击 `Add Provider`。
3. 填写 `Name`、`Type`、`API Key` 等必填选项。
4. 单击 `Add Model`，为该提供商配置至少一个模型。
5. 单击 `Save` 按钮。

## 使用兼容 Anthropic 的端点 { #use-an-anthropic-compatible-endpoint }

`Anthropic Claude` 提供商默认连接 `api.anthropic.com`。要将其指向使用相同 API 的自托管或代理服务，请将 `claudeCustomUrl` 设置为绝对 `http(s)` URL，其中包括端口及路径前缀，例如 `http://192.168.50.14:8080` 或 `https://gateway.example.com/anthropic`。

请提供服务的源地址和挂载前缀，**不要**提供其 API 基础路径：Anthropic 路径会追加到配置值后，因此 `claudeCustomUrl` 不应以 `/v1` 结尾，即使大多数兼容 Anthropic 的服务公布的基础 URL 都以此结尾。设置为 `http://192.168.50.14:8080/v1` 会导致 GPUStack 查找 `/v1/v1/models` 和 `/v1/v1/messages`；常见表现是推理本身能够访问该端点，但 `Test Model` 失败。请注意，这与 `openaiCustomUrl` 相反，后者是完整的 API 基础路径，包含 `/v1`。URL 中包含凭据、查询参数或片段都会被拒绝，API 密钥应填写在提供商自身的密钥字段中。

此类提供商默认仍通过 OpenAI API 对外提供服务：请求发往端点时会转换为 Anthropic 协议，响应返回时再转换回来。要改为原样转发，请在提供商配置中添加 `protocol: original`。透传模式保留 OpenAI schema 无法表示的所有内容（提示词缓存、思考块、工具使用块），但此时提供商**仅提供 Anthropic 协议**；不会再为它转换发往 `/v1/chat/completions` 的 OpenAI 风格请求，因此请使用 Anthropic 路径（`/v1/messages`）。

## 通过提供商使用 Qwen 模型 { #use-qwen-models-through-a-provider }

`Alibaba Qwen` 提供商默认连接 `dashscope.aliyuncs.com`，支持：

- **LLM** — `/v1/chat/completions`
- **Embedding** — `/v1/embeddings`
- **Rerank** — `/v1/rerank`
- **图像生成** — `/v1/images/generations`（如 qwen-image 系列模型）

图像生成是默认域名上的唯一例外：DashScope **仅在按工作空间分配的 `maas.aliyuncs.com` 域名**上提供该 API，`dashscope.aliyuncs.com` 对其返回 404。要使用图像模型，请将 `qwenDomain` 设置为你的工作空间域名（可在 DashScope，即百炼的工作空间页面查看），其余类别在同一入口继续可用。自托管或经代理的 Qwen 兼容服务的配置方式相同。

`qwenDomain` 填主机名，非 443 端口时附带端口，如 `maas.aliyuncs.com`：它会同时成为 GPUStack 的连接地址和发往上游的 Host 头。协议固定为 https。

图像响应携带 `data[].url` —— 24 小时有效的 OSS 预签名链接 —— 而非内嵌 base64，请在该时间窗口内下载或转存图片。`usage` 对象是 DashScope 特有格式（`width`、`height`、`image_count`），不是 OpenAI 的 token 用量。

## 为提供商添加路由 { #add-route-for-provider }

1. 进入 `Providers` 页面。
2. 找到要创建路由的提供商。
3. 在该提供商的 `Operations` 中单击 `Add Route`。
4. 根据需要修改提供商路由的 `Name` 和 `Route Targets`。
5. 单击 `Save` 按钮。

## 编辑提供商 { #edit-provider }

1. 进入 `Providers` 页面。
2. 找到要编辑的提供商。
3. 根据需要修改名称、类型和 API 密钥属性，并在模型列表中添加或移除模型。
4. 单击 `Save` 按钮。

## 删除提供商 { #delete-provider }

1. 进入 `Providers` 页面。
2. 找到要删除的提供商。
3. 单击 `Operations` 列中的 `Delete` 按钮。
4. 确认删除。
