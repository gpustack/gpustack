---
translation_status: translated
---

# 与 Cherry Studio 集成 { #integrate-with-cherry-studio }

Cherry Studio 可接入 GPUStack，使用本地部署的 LLM、向量嵌入和重排序能力。

## 部署模型 { #deploying-models }

1. 在 GPUStack 界面中打开 `Deployments` 页面，点击 `Deploy Model` 部署所需模型，例如：

    - qwen3-instruct-2507
    - qwen2.5-vl-7b
    - bge-m3
    - bge-reranker-v2-m3

   ![GPUStack 模型列表](../assets/integrations/integration-gpustack-models2.png)

2. 在模型的操作菜单中打开 `API Access Info`，查看接入信息：

   ![API 访问信息](../assets/integrations/integration-gpustack-api-access-info2.png)

## 创建 API 密钥 { #create-an-api-key }

1. 打开 `Access Control` > `API Keys` 页面，点击 `New API Key`。
2. 填写名称，点击 `Save`。
3. 复制 API 密钥并保存，供后续使用。

## 在 Cherry Studio 中接入 GPUStack { #integrating-gpustack-into-cherry-studio }

1. 打开 Cherry Studio，进入 `Settings` → `Model Provider`，找到并启用 GPUStack，按下图配置：

    - `API Key`：填写前面复制的 API 密钥。
    - `API Host`：填写 `API Access Info` 面板中的 `Access URL`。

   ![Cherry Studio 中的 GPUStack 配置](../assets/integrations/integration-cherrystudio-01.png)

2. 在 GPUStack 提供商配置中点击“Manage”，启用所需模型：

   ![管理模型](../assets/integrations/integration-cherrystudio-02.png)

   ![启用模型](../assets/integrations/integration-cherrystudio-03.png)

3. 可选：测试 API：

   ![测试 API](../assets/integrations/integration-cherrystudio-04.png)

配置完成后，返回 Cherry Studio 首页，即可开始使用模型。

## 使用 LLM { #using-llms }

![使用 LLM](../assets/integrations/integration-cherrystudio-05.png)

## 使用多模态模型 { #using-multimodal-models }

1. 选择多模态模型：

   ![选择多模态模型](../assets/integrations/integration-cherrystudio-13.png)

2. 提出多模态问题：

   ![多模态问答](../assets/integrations/integration-cherrystudio-14.png)

   ![多模态问答](../assets/integrations/integration-cherrystudio-15.png)

## 使用向量嵌入和重排序改进知识库问答 { #use-embeddings-and-reranking-to-improve-knowledge-base-qa }

1. 打开知识库配置页面：

   ![知识库设置](../assets/integrations/integration-cherrystudio-06.png)

2. 添加知识库：

   ![添加知识库](../assets/integrations/integration-cherrystudio-08.png)

3. 向知识库添加内容，下面以“Notes”为例：

   ![添加知识库内容](../assets/integrations/integration-cherrystudio-09.png)

   ![知识库内容详情](../assets/integrations/integration-cherrystudio-10.png)

4. 返回首页，使用知识库问答：

   ![知识库问答](../assets/integrations/integration-cherrystudio-11.png)

   ![知识库问答示例](../assets/integrations/integration-cherrystudio-12.png)
