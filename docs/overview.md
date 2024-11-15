# GPUStack

![demo](assets/gpustack-demo.gif)

GPUStack is an open-source GPU cluster manager for running large language models(LLMs).

### Key Features

- **Supports a Wide Variety of Hardware:** Run with different brands of GPUs in Apple MacBooks, Windows PCs, and Linux servers.
- **Scales with Your GPU Inventory:** Easily add more GPUs or nodes to scale up your operations.
- **Distributed Inference**: Supports both single-node multi-GPU and multi-node inference and serving.
- **Multiple Inference Backends**: Supports llama-box (llama.cpp) and vLLM as the inference backend.
- **Lightweight Python Package:** Minimal dependencies and operational overhead.
- **OpenAI-compatible APIs:** Serve APIs that are compatible with OpenAI standards.
- **User and API key management:** Simplified management of users and API keys.
- **GPU metrics monitoring:** Monitor GPU performance and utilization in real-time.
- **Token usage and rate metrics:** Track token usage and manage rate limits effectively.

## Supported Platforms

- [x] MacOS
- [x] Windows
- [x] Linux

The following Linux distributions are verified to work with GPUStack:

| Distributions | Versions        |
| ------------- | --------------- |
| Ubuntu        | \>= 20.04       |
| Debian        | \>= 11          |
| RHEL          | \>= 8           |
| Rocky         | \>= 8           |
| Fedora        | \>= 36          |
| OpenSUSE      | \>= 15.3 (leap) |
| OpenEuler     | \>= 22.03       |

!!! note

    The installation of GPUStack worker on a Linux system requires that the GLIBC version be 2.29 or higher.

### Supported Architectures

GPUStack supports both **AMD64** and **ARM64** architectures, with the following notes:

- On MacOS and Linux, if using Python versions below 3.12, ensure you install the Python distribution matching your architecture.
- On Windows, please use the AMD64 distribution of Python, as wheel packages for certain dependencies are unavailable for ARM64. If you use tools like `conda`, this will be handled automatically, as conda installs the AMD64 distribution by default.

## Supported Accelerators

- [x] Apple Metal
- [x] NVIDIA CUDA([Compute Capability](https://developer.nvidia.com/cuda-gpus) 6.0 and above)
- [x] Ascend CANN
- [x] Moore Threads MUSA

We plan to support the following accelerators in future releases.

- [ ] AMD ROCm
- [ ] Intel oneAPI

- [ ] Qualcomm AI Engine

## Supported Models

GPUStack uses [llama.cpp](https://github.com/ggerganov/llama.cpp) and [vLLM](https://github.com/vllm-project/vllm) as the backends and supports a wide range of models. Models from the following sources are supported:

1. [Hugging Face](https://huggingface.co/)

2. [ModelScope](https://modelscope.cn/)

3. [Ollama Library](https://ollama.com/library)

Example language models:

- [x] [LLaMA](https://huggingface.co/meta-llama)
- [x] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [x] [Baichuan](https://huggingface.co/models?search=baichuan-inc/Baichuan)
- [x] [Yi](https://huggingface.co/models?search=01-ai/Yi)
- [x] [Deepseek](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [Phi](https://huggingface.co/models?search=microsoft/phi)
- [x] [Grok-1](https://huggingface.co/xai-org/grok-1)

Example multimodal models:

- [x] [Llama3.2-Vision](https://huggingface.co/models?pipeline_tag=image-text-to-text&search=llama3.2)
- [x] [Pixtral](https://huggingface.co/models?search=pixtral)
- [x] [Qwen2-VL](https://huggingface.co/models?search=Qwen/Qwen2-VL)
- [x] [LLaVA](https://huggingface.co/models?search=llava)
- [x] [InternVL2](https://huggingface.co/models?search=internvl2)

For full list of supported models, please refer to the supported models section in the [inference backends](./user-guide/inference-backends.md) documentation.

## OpenAI-Compatible APIs

GPUStack serves OpenAI compatible APIs. For details, please refer to [OpenAI Compatible APIs](./user-guide/openai-compatible-apis.md)
