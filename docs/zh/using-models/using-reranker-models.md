# 使用重排序模型

**重排序模型** 是一类用于根据给定查询的相关性改进项目列表排序的专业模型。它们常用于信息检索和搜索系统，以优化初始搜索结果，优先呈现更可能满足用户意图的项目。重排序模型会接收初始文档列表并对条目重新排序，从而提升搜索引擎、推荐系统和问答等应用中的精度。

在本指南中，我们将演示如何在 GPUStack 中部署并使用重排序模型。

## 前提条件

开始之前，请确保你具备以下条件：

- 已安装并运行 GPUStack。如未安装，请参阅[快速开始指南](../quickstart.md)。
- 能访问 Hugging Face 以下载模型文件。

## 步骤 1：部署模型

按照以下步骤从 Hugging Face 部署模型：

1. 在 GPUStack 界面中进入 `Deployments` 页面。
2. 点击 `Deploy Model` 按钮。
3. 在下拉菜单中选择 `Hugging Face` 作为模型来源。
4. 启用 `GGUF` 复选框，以按 GGUF 格式筛选模型。
5. 使用左上角的搜索栏，搜索模型名称 `gpustack/bge-reranker-v2-m3-GGUF`。
6. 其余保持默认设置，点击 `Save` 按钮以部署模型。

![部署模型](../../assets/using-models/using-reranker-models/deploy-model.png)

部署完成后，你可以在 `Deployments` 页面查看模型部署状态。

![模型列表](../../assets/using-models/using-reranker-models/model-list.png)

## 步骤 2：生成 API 密钥

我们将使用 GPUStack API 与模型交互。为此，你需要生成一个 API 密钥：

1. 将鼠标悬停在用户头像上，进入 `API Keys` 页面。
2. 点击 `New API Key` 按钮。
3. 为 API 密钥输入名称并点击 `Save` 按钮。
4. 复制生成的 API 密钥。API 密钥仅可查看一次，请务必妥善保存。

## 步骤 3：重排序

在模型已部署且已有 API 密钥的情况下，你可以通过 GPUStack API 对文档列表进行重排序。以下是使用 `curl` 的示例脚本：

```bash
export SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $SERVER_URL/v1/rerank \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -d '{
        "model": "bge-reranker-v2-m3",
        "query": "What is a panda?",
        "top_n": 3,
        "documents": [
            "hi",
            "it is a bear",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
        ]
    }' | jq
```

将 `<your-server-url>` 替换为你的 GPUStack 服务器 URL，将 `<your-api-key>` 替换为你在上一步生成的 API 密钥。

示例响应：

```json
{
  "model": "bge-reranker-v2-m3",
  "object": "list",
  "results": [
    {
      "document": {
        "text": "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
      },
      "index": 2,
      "relevance_score": 1.951932668685913
    },
    {
      "document": {
        "text": "it is a bear"
      },
      "index": 1,
      "relevance_score": -3.7347371578216553
    },
    {
      "document": {
        "text": "hi"
      },
      "index": 0,
      "relevance_score": -6.157620906829834
    }
  ],
  "usage": {
    "prompt_tokens": 69,
    "total_tokens": 69
  }
}
```