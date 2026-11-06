---
translation_status: translated
---

# 推理 API { #inference-apis }

## OpenAI 兼容 API { #openai-compatible-apis }

GPUStack 在 `/v1` 端点提供 [OpenAI 兼容 API](https://platform.openai.com/docs/api-reference)。

支持 OpenAI 兼容 API 的应用或框架，只需将端点指向 GPUStack，即可接入并使用 GPUStack 上部署的模型。

### 支持的端点 { #supported-endpoints }

支持以下 API 端点：

- [x] [列出模型](https://platform.openai.com/docs/api-reference/models/list)
- [x] [创建补全](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [创建对话补全](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [创建响应](https://platform.openai.com/docs/api-reference/responses/create)
- [x] [创建向量嵌入](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [创建图像](https://platform.openai.com/docs/api-reference/images/create)
- [x] [编辑图像](https://platform.openai.com/docs/api-reference/images/createEdit)
- [x] [生成语音](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [x] [创建语音转写](https://platform.openai.com/docs/api-reference/audio/createTranscription)

### 用法 { #usage }

#### cURL 示例 { #curl-example }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'
```

## Anthropic 兼容 API { #anthropic-compatible-apis }

GPUStack 提供 Anthropic 兼容的 [`/v1/messages` API](https://platform.claude.com/docs/en/api/messages/create)。

### 用法 { #usage_1 }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "max_tokens": 1024
  }'
```

## Jina 兼容的 Rerank API { #jina-compatible-rerank-api }

在检索增强生成（RAG）中，重排序是指从检索出的文档或知识源中挑选最相关的信息，再将其展示给用户或用于生成答案。

OpenAI 兼容 API **不提供** `rerank` 端点。为补充这一能力，GPUStack 在 `/v1/rerank` 路径提供 [Jina 兼容的 Rerank API](https://jina.ai/reranker/)。

### 用法 { #usage_2 }

#### cURL 示例 { #curl-example_1 }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/rerank \
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

输出示例：

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

## System One 决策 API { #system-one-decision-api }

决策模型——`decision` 类别的部署——遵循 System One（Jev）决策契约，而非生成文本：给定一个 state 和一组 questions，它们返回类型化的答案而非补全。GPUStack 在 `/v1/systemone` 路径上提供该契约。

`model` 字段必填，取模型路由名称，与 `/v1/chat/completions` 完全一致。GPUStack 解析模型后，将请求体其余部分和整个响应原样透传给决策引擎。

### 使用方法 { #usage_3 }

#### cURL 示例 { #curl-example_2 }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/systemone \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
      "model": "qwen3-decision-8b",
      "state": {
          "messages": [
              { "role": "user", "content": "Write a 200-word poem in Hungarian." }
          ]
      },
      "questions": {
          "model_selection": {
              "type": "choice",
              "instructions": "Pick the model best suited.",
              "criteria": {
                  "qwen3-8b": "Strong multilingual generation.",
                  "llama3-8b": "Faster, weaker multilingual."
              }
          }
      }
  }' | jq
```

输出示例：

```json
{
  "model": "qwen3-decision-8b",
  "usage": {
    "input_tokens": 189,
    "output_tokens": 12
  },
  "answers": {
    "model_selection": {
      "type": "choice",
      "choice": "qwen3-8b",
      "confidence": 0.93,
      "probabilities": {
        "qwen3-8b": 0.93,
        "llama3-8b": 0.07
      }
    }
  }
}
```

`state` 是做出决策所依据的情境——纯文本，或 chat messages 这类结构化 JSON。`questions` 下的每个条目将问题 id 映射到问题定义，三种类型是常规用法：

| 类型 | 含义 | `criteria` | 答案字段 |
| --- | --- | --- | --- |
| `choice` | 选择一个选项 | 选项名 → 描述 | `choice`——选中的选项名 |
| `score` | 按有序级别对 state 评分 | 有序的级别描述列表 | `score`——级别索引 |
| `noul` | 二元（是/否）问题 | `true` / `false` 两侧的可选描述 | `noul`——布尔值 |

问题的答案以相同 id 返回在 `answers` 中，并附带 `confidence` 和 `probabilities`。契约中的问题集是开放的——引擎可以接受更多字段——GPUStack 只校验 `model`，其余内容原样转发。决策模型的部署参见[使用决策模型](../using-models/using-decision-models.md)。

## 其他 API { #other-apis }

对于其他 API 类型，可以在部署模型时启用 **Generic Proxy** 功能。

启用后，可通过以下两种方式指定目标模型：

**路径方式（推荐）**：将模型路由 ID 编码到 URL 中，格式为 `/model/proxy/<model_route_id>/<upstream-path>`，无需额外请求头：

```bash
export GPUSTACK_API_KEY=your_api_key
# Assume the model route id is 42.
curl http://your_gpustack_server_url/model/proxy/42/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{"inputs": ["What is Deep Learning?", "Deep Learning is not..."]}'
```

**请求头方式（已弃用）**：请求 `/model/proxy/<upstream-path>`，通过 `X-GPUStack-Model` 请求头或 JSON 请求体中的 `model` 字段指定目标模型：

```bash
curl http://your_gpustack_server_url/model/proxy/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -H "X-GPUStack-Model: bge-m3" \
  -d '{"inputs": ["What is Deep Learning?", "Deep Learning is not..."]}'
```

请求头方式仅为向后兼容而保留，将在未来版本中移除；新的集成应使用路径方式。

更多说明见[启用 Generic Proxy](../user-guide/model-deployment-management.md#enable-generic-proxy)。
