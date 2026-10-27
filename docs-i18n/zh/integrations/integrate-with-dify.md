---
translation_status: translated
---

# 与 Dify 集成 { #integrate-with-dify }

Dify 可接入 GPUStack，使用本地部署的 LLM、向量嵌入、重排序、图像生成、语音转文本和文本转语音能力。

## 部署模型 { #deploying-models }

1. 在 GPUStack 界面中打开 `Deployments` 页面，点击 `Deploy Model` 部署所需模型，例如：

- qwen3-8b
- qwen2.5-vl-3b-instruct
- bge-m3
- bge-reranker-v2-m3

![GPUStack 模型列表](../assets/integrations/integration-gpustack-models.png)

2. 在模型的操作菜单中打开 `API Access Info`，查看接入信息。

![GPUStack API 访问信息](../assets/integrations/integration-gpustack-api-access-info.png)

## 创建 API 密钥 { #create-an-api-key }

1. 打开 `Access Control` > `API Keys` 页面，点击 `New API Key`。
2. 填写名称，点击 `Save`。
3. 复制 API 密钥并保存，供后续使用。

## 在 Dify 中接入 GPUStack { #integrating-gpustack-into-dify }

1. 打开 Dify 界面，点击右上角 `PLUGINS`，选择 `Install from Marketplace`，搜索并安装 GPUStack 插件。

![在 Dify 中安装 GPUStack 插件](../assets/integrations/integration-dify-install-gpustack-plugin.png)

2. 安装后，进入 `Settings > Model Provider > GPUStack`，选择 `Add Model`，填写：

- Model Type：按模型选择对应类型。
- Model Name：必须与 GPUStack 中部署的模型名称一致。
- Server URL：`http://your-gpustack-url`。不要使用 `localhost`，因为它指向容器内部网络。如使用自定义端口，请包含端口号。还需确认 Dify 容器内可以访问该 URL，可使用 `curl` 测试。
- API Key：填写前面复制的 API 密钥。

点击 `Save` 添加模型：

![在 Dify 中添加模型](../assets/integrations/integration-dify-add-model.png)

按需添加其他模型，然后在 `System Model Settings` 中选择已添加的模型并保存：

![Dify 系统模型设置](../assets/integrations/integration-dify-system-model-settings.png)

现在可以在 `Studio` 和 `Knowledge` 中使用模型。以下是一个简单示例：

1. 进入 `Knowledge` 创建知识库，并上传文档：

![在 Dify 中创建知识库](../assets/integrations/integration-dify-create-knowledge.png)

2. 配置 Chunk Settings 和 Retrieval Settings，使用向量嵌入模型生成文档向量，使用重排序模型对检索结果排序。

![在 Dify 中设置向量嵌入和重排序模型](../assets/integrations/integration-dify-set-embedding-and-rerank-model.png)

3. 文档导入成功后，在 `Studio` 创建应用，添加前面创建的知识库，选择对话模型并开始交互：

![在 Dify 中与模型对话](../assets/integrations/integration-dify-chat-with-model.png)

4. 将模型切换为 `qwen2.5-vl-3b-instruct`，移除先前添加的知识库，启用 `Vision`，然后在聊天中上传图片，使用多模态输入：

![在 Dify 中使用视觉语言模型对话](../assets/integrations/integration-dify-chat-with-vlm.png)
