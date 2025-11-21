# Rerank API

In the context of Retrieval-Augmented Generation (RAG), reranking refers to the process of selecting the most relevant information from retrieved documents or knowledge sources before presenting them to the user or utilizing them for answer generation.

GPUStack serves [Jina compatible Rerank API](https://jina.ai/reranker/) using the `/v1/rerank` path.

!!! note

    For other APIs, GPUStack allows you to enable the Generic Proxy when deploying a model.

    With the Generic Proxy enabled, you can send API requests to a unified gateway and add the `X-GPUStack-Model` header.

    This header tells GPUStack which model should handle the request.

    For more details, see [Enable Generic Proxy](model-deployment-management.md/#enable-generic-proxy).

## Usage

The following is an example using the Rerank API:

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

Example output:

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
