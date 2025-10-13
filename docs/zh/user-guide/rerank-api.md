# 重排序 API

在检索增强生成（RAG）的语境中，重排序（reranking）指的是在向用户呈现或用于答案生成之前，从已检索的文档或知识源中筛选出最相关信息的过程。

GPUStack 通过 `/v1/rerank` 路径提供与 [Jina 兼容的重排序 API](https://jina.ai/reranker/)。

## 支持的模型

以下模型可用于重排序：

- [bce-reranker-base_v1](https://huggingface.co/gpustack/bce-reranker-base_v1-GGUF)
- [jina-reranker-v1-turbo-en](https://huggingface.co/gpustack/jina-reranker-v1-turbo-en-GGUF)
- [jina-reranker-v1-tiny-en](https://huggingface.co/gpustack/jina-reranker-v1-tiny-en-GGUF)
- [bge-reranker-v2-m3](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF)
- [gte-multilingual-reranker-base](https://huggingface.co/gpustack/gte-multilingual-reranker-base-GGUF) <span title="experimental">🧪</span>
- [jina-reranker-v2-base-multilingual](https://huggingface.co/gpustack/jina-reranker-v2-base-multilingual-GGUF) <span title="experimental">🧪</span>

## 用法

下面是使用重排序 API 的示例：

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/rerank \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -d '{
        "model": "bge-reranker-v2-m3",
        "query": "什么是熊猫？",
        "top_n": 3,
        "documents": [
            "嗨",
            "它是一种熊",
            "大熊猫（Ailuropoda melanoleuca），有时被称为熊猫或简称熊猫，是中国特有的一种熊科动物。"
        ]
    }' | jq
```

示例输出：

```json
{
  "model": "bge-reranker-v2-m3",
  "object": "list",
  "results": [
    {
      "document": {
        "text": "大熊猫（Ailuropoda melanoleuca），有时被称为熊猫或简称熊猫，是中国特有的一种熊科动物。"
      },
      "index": 2,
      "relevance_score": 1.951932668685913
    },
    {
      "document": {
        "text": "它是一种熊"
      },
      "index": 1,
      "relevance_score": -3.7347371578216553
    },
    {
      "document": {
        "text": "嗨"
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