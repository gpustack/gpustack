# GPUStack

GPUStack aims to get you started with running LLMs and performing inference in a simple yet scalable manner.

- Supports a wide variety of hardware.
- Scales with your GPU inventory.
- A lightweight Python package with minimal dependencies and operational overhead.
- OpenAI-compatible APIs.
- User and API key management.
- GPU metrics monitoring.
- Token usage and rate metrics.

## Supported Platforms

- [x] MacOS
- [x] Linux
- [x] Windows

## Supported Accelerators

We haven't tested everything out in the wild, but we plan to support all of the following in the near future.

- [x] Apple Metal
- [x] NVIDIA CUDA
- [ ] AMD ROCm
- [ ] Intel oneAPI

## Supported Models

GPUStack uses [llama.cpp](https://github.com/ggerganov/llama.cpp) as the backend and supports large language models in [GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md). Models from the following sources are supported:

1. [Hugging Face](https://huggingface.co/)

2. [Ollama Library](https://ollama.com/library)

Here are some example models:

- [x] [LLaMA](https://huggingface.co/meta-llama)
- [x] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [x] [Baichuan](https://huggingface.co/models?search=baichuan-inc/Baichuan)
- [x] [Aquila](https://huggingface.co/models?search=BAAI/Aquila)
- [x] [Yi](https://huggingface.co/models?search=01-ai/Yi)
- [x] [StableLM](https://huggingface.co/stabilityai)
- [x] [Deepseek](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [Phi](https://huggingface.co/models?search=microsoft/phi)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)

## Architecture

// TODO
