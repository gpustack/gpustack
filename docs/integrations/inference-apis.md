# Inference APIs

## OpenAI-Compatible APIs

GPUStack provides [OpenAI-compatible APIs](https://platform.openai.com/docs/api-reference) at the `/v1` endpoint.

You can integrate and use models deployed on GPUStack with any application or framework that supports the OpenAI-compatible API, simply by pointing it to GPUStack's OpenAI-compatible endpoint.

### Supported Endpoints

The following API endpoints are supported:

- [x] [List Models](https://platform.openai.com/docs/api-reference/models/list)
- [x] [Create Completion](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [Create Chat Completion](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [Create Response](https://platform.openai.com/docs/api-reference/responses/create)
- [x] [Create Embeddings](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [Create Image](https://platform.openai.com/docs/api-reference/images/create)
- [x] [Create Image Edit](https://platform.openai.com/docs/api-reference/images/createEdit)
- [x] [Create Speech](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [x] [Create Transcription](https://platform.openai.com/docs/api-reference/audio/createTranscription)

### Usage

#### cURL Example

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

## Anthropic-Compatible APIs

GPUStack provides the Anthropic-compatible [`/v1/messages` API](https://platform.claude.com/docs/en/api/messages/create).

### Usage

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

## Jina-Compatible Rerank API

In the context of Retrieval-Augmented Generation (RAG), reranking refers to the process of selecting the most relevant information from retrieved documents or knowledge sources before presenting them to the user or utilizing them for answer generation.

Note that the OpenAI-compatible APIs **do not** provide a `rerank` endpoint. To fill this gap, GPUStack provides a [Jina-compatible Rerank API](https://jina.ai/reranker/) at the `/v1/rerank` path.

### Usage

#### cURL Example

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

## System One Decision API

Decision models — deployments in the `decision` category — answer the System One (Jev) decision contract instead of generating text: given a state and a set of questions, they return typed answers rather than completions. GPUStack exposes the contract at the `/v1/systemone` path.

The `model` field is required and takes the model route name, exactly like `/v1/chat/completions`. GPUStack resolves the model and passes the rest of the request body, and the whole response, through to the decision engine untouched.

### Usage

#### cURL Example

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

Example output:

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

A `state` is the situation the decision is made against — plain text or structured JSON such as chat messages. Each entry under `questions` maps a question id to a question definition, and three types are conventional:

| Type | Meaning | `criteria` | Answer field |
| --- | --- | --- | --- |
| `choice` | Pick one option | Option name → description | `choice` — the picked option name |
| `score` | Score the state against ordered levels | Ordered list of level descriptions | `score` — the level index |
| `noul` | Binary (yes/no) question | Optional descriptions for the `true` / `false` sides | `noul` — a boolean |

The answer for a question comes back under the same id in `answers`, carrying `confidence` and `probabilities` alongside the verdict. The question set is open in the contract — engines may accept further fields — and GPUStack validates only `model`, forwarding the rest verbatim. For deploying decision models, see [Using Decision Models](../using-models/using-decision-models.md).

## Other APIs

For other API types, GPUStack allows you to enable the **Generic Proxy** feature when deploying a model.

Once enabled, there are two ways to address the target model:

**Path-based form (recommended)** — encode the model route id directly in the URL as `/model/proxy/<model_route_id>/<upstream-path>`. No extra header is required:

```bash
export GPUSTACK_API_KEY=your_api_key
# Assume the model route id is 42.
curl http://your_gpustack_server_url/model/proxy/42/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{"inputs": ["What is Deep Learning?", "Deep Learning is not..."]}'
```

**Header-based form (deprecated)** — send to `/model/proxy/<upstream-path>` and specify the target model via the `X-GPUStack-Model` header or the `model` field in the JSON body:

```bash
curl http://your_gpustack_server_url/model/proxy/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -H "X-GPUStack-Model: bge-m3" \
  -d '{"inputs": ["What is Deep Learning?", "Deep Learning is not..."]}'
```

The header-based form is retained for backward compatibility and will be removed in a future release; new integrations should use the path-based form.

For more details, see [Enable Generic Proxy](../user-guide/model-deployment-management.md#enable-generic-proxy).
