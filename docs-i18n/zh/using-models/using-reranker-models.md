---
translation_status: translated
---

# 使用重排序模型 { #using-reranker-models }

**重排序模型（Reranker Models）**根据查询相关性对条目列表重新排序。它们通常用于信息检索和搜索系统，优化初步检索结果，优先展示更符合用户意图的条目。重排序模型接收初步检索的文档列表并调整顺序，从而提高搜索引擎、推荐系统和问答任务等应用的准确性。

本指南介绍如何在 GPUStack 中部署和使用重排序模型。

## 前提条件 { #prerequisites }

开始前，请确认：

- GPUStack 已安装并运行；如未完成，请参考[快速入门](../quickstart.md)。
- 可以访问 Hugging Face，以下载模型文件。

## 步骤 1：部署模型 { #step-1-deploy-the-model }

按以下步骤从 Catalog 部署模型：

1. 在 GPUStack 界面中打开 `Catalog` 页面。
2. 在模型列表中通过下拉菜单选择 `Reranker` 筛选模型。
3. 查看模型说明、最大上下文长度和支持的规格。

![部署模型](../assets/using-models/using-reranker-models/deploy-model.png)

部署后，可在 `Deployments` 页面查看模型部署状态。

![模型列表](../assets/using-models/using-reranker-models/model-list.png)

## 步骤 2：生成 API 密钥 { #step-2-generate-an-api-key }

通过 GPUStack API 调用模型前，需要生成 API 密钥：

1. 打开 `Access Control` > `API Keys` 页面。
2. 点击 `New API Key`。
3. 输入密钥名称并点击 `Save`。
4. 复制生成的 API 密钥。密钥只能查看一次，请妥善保存。

## 步骤 3：重排序 { #step-3-reranking }

部署模型并取得 API 密钥后，即可通过 GPUStack API 对文档列表重新排序。以下是使用 `curl` 的示例：

```bash
export SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $SERVER_URL/v1/rerank \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -d '{
        "model": "qwen3-reranker-4b",
        "query": "What is a panda?",
        "top_n": 3,
        "documents": [
            "hi",
            "it is a bear",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
        ]
    }' | jq
```

将 `<your-server-url>` 替换为 GPUStack Server 的 URL，将 `<your-api-key>` 替换为上一步生成的 API 密钥。

响应示例：

```json
{
  "model": "qwen3-reranker-4b",
  "object": "list",
  "results": [
    {
      "index": 0,
      "document": {
        "text": "hi",
        "multi_modal": null
      },
      "relevance_score": 0.9996911287307739
    },
    {
      "index": 2,
      "document": {
        "text": "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        "multi_modal": null
      },
      "relevance_score": 0.8206241726875305
    },
    {
      "index": 1,
      "document": {
        "text": "it is a bear",
        "multi_modal": null
      },
      "relevance_score": 0.7244728803634644
    }
  ],
  "usage": {
    "total_tokens": 51
  }
}
```
