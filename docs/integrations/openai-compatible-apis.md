# OpenAI Compatible APIs

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
