# OpenAI 兼容 API

GPUStack 通过 `/v1-openai` 路径提供[与 OpenAI 兼容的 API](https://platform.openai.com/docs/api-reference)。除 `models` 端点（保留给 GPUStack 管理 API）外，大多数 API 也可通过 `/v1` 路径的别名使用。

对于所有支持 OpenAI 兼容 API 的应用与框架，您可以通过 GPUStack 提供的 OpenAI 兼容 API 集成并使用部署在 GPUStack 上的模型。

## 支持的端点

支持以下 API 端点：

- [x] [列出模型](https://platform.openai.com/docs/api-reference/models/list)
- [x] [创建补全](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [创建聊天补全](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [创建向量嵌入](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [创建图像](https://platform.openai.com/docs/api-reference/images/create)
- [x] [创建图像编辑](https://platform.openai.com/docs/api-reference/images/createEdit)
- [x] [创建语音](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [x] [创建转录](https://platform.openai.com/docs/api-reference/audio/createTranscription)

## 重排序（Rerank）API

在检索增强生成（RAG）的场景中，重排序是指在向用户展示或用于答案生成之前，从已检索的文档或知识源中选出最相关信息的过程。

需要注意的是，OpenAI 兼容 API 并不提供 `rerank` API，因此 GPUStack 通过 `/v1/rerank` 路径提供[与 Jina 兼容的重排序 API](https://jina.ai/reranker/)。