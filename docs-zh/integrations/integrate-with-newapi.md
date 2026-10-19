---
translation_status: translated
---

# 与 NewAPI 集成 { #integrate-with-newapi }

NewAPI 可接入 GPUStack，将本地部署的 LLM、向量嵌入、重排序、语音转文本和文本转语音能力汇聚到统一的 OpenAI 兼容 API 网关，供企业员工使用。

## 在 GPUStack 中部署模型 { #deploying-models-in-gpustack }

1. 在 GPUStack 界面中打开 `Deployments` 页面，点击 `Deploy Model` 部署所需模型，例如：

- qwen3-8b
- qwen2.5-vl-3b-instruct
- bge-m3
- bge-reranker-v2-m3

![GPUStack 模型列表](../assets/integrations/integration-gpustack-models.png)

2. 在模型的操作菜单中打开 `API Access Info`，查看接入信息。

![GPUStack API 访问信息](../assets/integrations/integration-gpustack-api-access-info.png)

## 在 GPUStack 中创建 API 密钥 { #create-an-api-key-in-gpustack }

1. 在 GPUStack 中打开 `Access Control` > `API Keys` 页面，点击 `New API Key`。
2. 填写名称，点击 `Save`。
3. 复制 API 密钥并保存，供后续使用。

## 在 NewAPI 中接入 GPUStack { #integrating-gpustack-into-newapi }

1. 打开 NewAPI 管理控制台，在左侧选择 `Channels`，点击 `Add Channel`。
2. 按以下说明配置渠道参数：

- **Type**：选择 `OpenAI` 或 `Custom`。GPUStack 提供标准 OpenAI 兼容端点和 Anthropic 端点。
- **Name**：填写渠道名称，通常与模型 ID 相同，例如 `qwen3-8b`。
- **Base URL (Base URL)**：`http://your-gpustack-url`，填写 GPUStack 的访问 URL 或域名，可用于 OpenAI 兼容或 Anthropic 端点。去掉 `v1`，因为 NewAPI 会自动添加。如果 NewAPI 运行在 Docker 容器中，不要使用 `localhost`，它指向容器内部网络。如果 GPUStack Server 启用了 TLS，也可以使用 HTTPS。
- **Key（密钥）**：填写上一步复制的 GPUStack API 密钥。
- **Models（模型）**：填写或选择 GPUStack 中部署的准确模型名称，例如 `qwen3-8b`、`bge-m3`、`bge-reranker-v2-m3`。模型名称必须与 GPUStack 中的部署名称一致。

![在 NewAPI 中添加渠道](../assets/integrations/newapi1.png)
![在 NewAPI 中添加渠道](../assets/integrations/newapi2.png)
![在 NewAPI 中添加渠道](../assets/integrations/newapi3.png)
![在 NewAPI 中添加渠道](../assets/integrations/newapi4.png)

3. 配置模型定价。

![配置 NewAPI 模型定价](../assets/integrations/newapi5.png)
![配置 NewAPI 模型定价](../assets/integrations/newapi6.png)
![配置 NewAPI 模型定价](../assets/integrations/newapi7.png)

4. 在渠道列表中点击 `Test`，验证 NewAPI 与 GPUStack 之间的连接。

![测试 NewAPI 渠道连接](../assets/integrations/newapi8.png)

---

## 在 NewAPI 中验证和使用接入的模型 { #verifying-and-using-the-integrated-models-in-newapi }

1. 在 NewAPI 中打开 `api keys` 页面，点击 `create api key`。

* 注意：API 密钥所选的分组必须与渠道配置中的分组一致，否则无法访问模型。

2. 复制新生成的 NewAPI 密钥，其前缀为 `sk-`。
3. 现在可以通过 NewAPI 的统一端点访问 GPUStack 提供的所有模型：

- **统一 Base URL**：`http://your-newapi-url/v1`
- **统一 API 密钥**：`sk-xxxxxx`，即您的 NewAPI 密钥。

使用 `curl` 的请求示例：

```bash
curl http://your-newapi-url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-newapi-token" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
