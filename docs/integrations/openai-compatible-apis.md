# OpenAI-Compatible APIs

GPUStack serves [OpenAI-compatible APIs](https://platform.openai.com/docs/api-reference) using the `/v1` path.

For all applications and frameworks that support the OpenAI-compatible API, you can integrate and use the models deployed on GPUStack through the OpenAI-compatible API provided by GPUStack.

## Supported Endpoints

The following API endpoints are supported:

- [x] [List Models](https://platform.openai.com/docs/api-reference/models/list)
- [x] [Create Completion](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [Create Chat Completion](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [Create Embeddings](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [Create Image](https://platform.openai.com/docs/api-reference/images/create)
- [x] [Create Image Edit](https://platform.openai.com/docs/api-reference/images/createEdit)
- [x] [Create Speech](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [x] [Create Transcription](https://platform.openai.com/docs/api-reference/audio/createTranscription)

## Rerank API

In the context of Retrieval-Augmented Generation (RAG), reranking refers to the process of selecting the most relevant information from retrieved documents or knowledge sources before presenting them to the user or utilizing them for answer generation.

It is important to note that the OpenAI-compatible APIs does not provide a `rerank` API, so GPUStack serves [Jina compatible Rerank API](https://jina.ai/reranker/) using the `/v1/rerank` path.

## Non-OpenAI-Compatible APIs

For other non-OpenAI-compatible APIs, GPUStack allows you to enable the Generic Proxy when deploying a model.

With the Generic Proxy enabled, GPUStack determines which model to forward the request to by checking either of the following:

- the "model" field in the JSON body

- the `X-GPUStack-Model` header

After enabling the Generic Proxy, GPUStack can forward API requests to the target model via the `/model/proxy` endpoint. For example:

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/model/proxy/embed \
-X POST \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $GPUSTACK_API_KEY" \
-H "X-GPUStack-Model: bge-m3" \
-d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}'
```

For more details, see [Enable Generic Proxy](../user-guide/model-deployment-management.md#enable-generic-proxy).
