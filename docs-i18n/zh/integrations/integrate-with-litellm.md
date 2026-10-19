---
translation_status: translated
---

# 与 LiteLLM 集成 { #integrate-with-litellm }

LiteLLM 可接入 GPUStack，将本地部署的 LLM、向量嵌入、重排序、语音转文本和文本转语音能力汇聚到统一的 OpenAI 兼容／Anthropic API 网关，供企业员工使用。

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

## 在 LiteLLM 中接入 GPUStack { #integrating-gpustack-into-litellm }

1. 打开 LiteLLM 管理界面：

```
http://your-litellm-url:4000/ui
```

![LiteLLM 管理界面](../assets/integrations/litellm-1.png)

2. 打开 models+endpoints 菜单，选择 add model 添加模型。

![在 LiteLLM 中添加模型](../assets/integrations/litellm-2.png)
![LiteLLM 模型配置](../assets/integrations/litellm-3.png)

3. 在 LiteLLM 的 Playground 中测试。

![在 LiteLLM 中与模型对话](../assets/integrations/litellm-4.png)
