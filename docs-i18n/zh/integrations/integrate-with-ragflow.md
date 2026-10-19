---
translation_status: translated
---

# 与 RAGFlow 集成 { #integrate-with-ragflow }

RAGFlow 可接入 GPUStack，使用本地部署的 LLM、向量嵌入、重排序、语音转文本和文本转语音能力。

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

## 在 RAGFlow 中接入 GPUStack { #integrating-gpustack-into-ragflow }

1. 打开 RAGFlow 界面，点击右上角头像，选择 `Model Providers > GPUStack`，点击 `Add the model`，填写：

- Model type：按模型选择对应类型。
- Model name：必须与 GPUStack 中部署的模型名称一致。
- Base URL：`http://your-gpustack-url/v1`，不要附加其他路径，也不要使用 `localhost`，因为它指向容器内部网络。如使用自定义端口，请包含端口号。还需确认 RAGFlow 容器内可以访问该 URL，可使用 `curl` 测试。
- API-Key：填写前面复制的 API 密钥。
- Max Tokens：填写当前模型配置支持的最大 token 数。

点击 `OK` 添加模型：

![在 RAGFlow 中添加模型](../assets/integrations/integration-ragflow-add-model.png)

2. 按需添加其他模型，在 `Set default models` 中选择已添加的模型并保存：

![在 RAGFlow 中设置默认模型](../assets/integrations/integration-ragflow-set-default-models.png)

现在可以在 `Chat` 和 `Knowledge Base` 中使用模型。以下是一个简单示例：

1. 进入 `Knowledge base` 创建知识库，并添加文件：

![在 RAGFlow 中创建知识库](../assets/integrations/integration-ragflow-create-knowledge-base.png)

2. 打开 `Retrieval testing`，将重排序模型设为 `bge-reranker-v2-m3`：

![在 RAGFlow 中设置重排序模型](../assets/integrations/integration-ragflow-set-rerank-model.png)

3. 在 `Chat` 中创建助手，关联前面创建的知识库，并选择对话模型：

![在 RAGFlow 中创建助手](../assets/integrations/integration-ragflow-create-assistant.png)

4. 创建对话，即可与模型交互并查询知识库：

![在 RAGFlow 中与模型对话](../assets/integrations/integration-ragflow-chat-with-model.png)

5. 编辑助手，将模型切换为 `qwen2.5-vl-3b-instruct`。保存后创建新对话，并上传图片以启用多模态输入：

![在 RAGFlow 中使用视觉语言模型对话](../assets/integrations/integration-ragflow-chat-with-vlm.png)
