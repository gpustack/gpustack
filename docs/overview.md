# GPUStack

![demo](assets/gpustack-demo.gif)

GPUStack is an open-source GPU cluster manager for running large language models(LLMs).

### Key Features

- **Supports a Wide Variety of Hardware:** Run with different brands of GPUs in Apple MacBooks, Windows PCs, and Linux servers.
- **Scales with Your GPU Inventory:** Easily add more GPUs or nodes to scale up your operations.
- **Distributed Inference**: Supports both single-node multi-GPU and multi-node inference and serving.
- **Lightweight Python Package:** Minimal dependencies and operational overhead.
- **OpenAI-compatible APIs:** Serve APIs that are compatible with OpenAI standards.
- **User and API key management:** Simplified management of users and API keys.
- **GPU metrics monitoring:** Monitor GPU performance and utilization in real-time.
- **Token usage and rate metrics:** Track token usage and manage rate limits effectively.

## Supported Platforms

- [x] MacOS
- [x] Windows
- [x] Linux

| Distributions | Versions        | 
|---------------|-----------------| 
| Ubuntu        | \>= 20.04       |
| Debian        | \>= 11          |
| RHEL          | \>= 9           |
| Rocky         | \>= 9           |
| Fedora        | \>= 36          |
| OpenSUSE      | \>= 15.3 (leap) |
| OpenEuler     | \>= 22.03       |

!!! note

    The installation of GPUStack worker on a Linux system requires that the GLIBC version be 2.29 or higher.


## Supported Accelerators

- [x] Apple Metal
- [x] NVIDIA CUDA([Compute Capability](https://developer.nvidia.com/cuda-gpus) 6.0 and above)

We plan to support the following accelerators in future releases.

- [ ] AMD ROCm
- [ ] Intel oneAPI
- [ ] MTHREADS MUSA
- [ ] Qualcomm AI Engine

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
- [x] [Gemma](https://huggingface.co/models?search=google/gemma)
- [x] [Mamba](https://huggingface.co/models?search=state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/xai-org/grok-1)

## OpenAI-Compatible APIs

GPUStack serves OpenAI compatible APIs. For details, please refer to [OpenAI Compatible APIs](./user-guide/openai-compatible-apis.md)
