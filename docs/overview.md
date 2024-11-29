# GPUStack

![demo](assets/gpustack-demo.gif)

GPUStack is an open-source GPU cluster manager for running AI models.

### Key Features

- **Broad Hardware Compatibility:** Run with different brands of GPUs in Apple MacBooks, Windows PCs, and Linux servers.
- **Broad Model Support:** From LLMs to diffusion models, audio, embedding, and reranker models.
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

- [x] Apple Metal (M-series chips)
- [x] NVIDIA CUDA ([Compute Capability](https://developer.nvidia.com/cuda-gpus) 6.0 and above)
- [x] Ascend CANN
- [x] Moore Threads MUSA

We plan to support the following accelerators in future releases.

- [ ] AMD ROCm
- [ ] Intel oneAPI

- [ ] Qualcomm AI Engine

## Supported Models

GPUStack uses [llama-box](https://github.com/gpustack/llama-box) (bundled [llama.cpp](https://github.com/ggerganov/llama.cpp) and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) server), [vLLM](https://github.com/vllm-project/vllm) and [vox-box](https://github.com/gpustack/vox-box) as the backends and supports a wide range of models. Models from the following sources are supported:

1. [Hugging Face](https://huggingface.co/)

2. [ModelScope](https://modelscope.cn/)

3. [Ollama Library](https://ollama.com/library)

4. Local File Path

### Example Models:

| **Category**                     | **Models**                                                                                                                                                                                                                                                                                                                                   |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Large Language Models(LLMs)**  | [Qwen](https://huggingface.co/models?search=Qwen/Qwen), [LLaMA](https://huggingface.co/meta-llama), [Mistral](https://huggingface.co/mistralai), [Deepseek](https://huggingface.co/models?search=deepseek-ai/deepseek), [Phi](https://huggingface.co/models?search=microsoft/phi), [Yi](https://huggingface.co/models?search=01-ai/Yi)       |
| **Vision Language Models(VLMs)** | [Llama3.2-Vision](https://huggingface.co/models?pipeline_tag=image-text-to-text&search=llama3.2), [Pixtral](https://huggingface.co/models?search=pixtral) , [Qwen2-VL](https://huggingface.co/models?search=Qwen/Qwen2-VL), [LLaVA](https://huggingface.co/models?search=llava), [InternVL2](https://huggingface.co/models?search=internvl2) |
| **Diffusion Models**             | [Stable Diffusion](https://huggingface.co/models?search=gpustack/stable-diffusion), [FLUX](https://huggingface.co/models?search=gpustack/flux)                                                                                                                                                                                               |
| **Rerankers**                    | [GTE](https://huggingface.co/gpustack/gte-multilingual-reranker-base-GGUF), [BCE](https://huggingface.co/gpustack/bce-reranker-base_v1-GGUF), [BGE](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF), [Jina](https://huggingface.co/models?search=gpustack/jina)                                                                     |
| **Audio Models**                 | [Whisper](https://huggingface.co/models?search=Systran/faster) (speech-to-text), [CosyVoice](https://huggingface.co/models?search=FunAudioLLM/CosyVoice) (text-to-speech)                                                                                                                                                                    |

For full list of supported models, please refer to the supported models section in the [inference backends](./user-guide/inference-backends.md) documentation.

## OpenAI-Compatible APIs

GPUStack serves OpenAI compatible APIs. For details, please refer to [OpenAI Compatible APIs](./user-guide/openai-compatible-apis.md)
