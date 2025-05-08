# GPUStack

![demo](assets/gpustack-demo.gif)

GPUStack is an open-source GPU cluster manager for running AI models.

### Key Features

- **Broad Hardware Compatibility:** Run with different brands of GPUs in Apple Macs, Windows PCs, and Linux servers.
- **Broad Model Support:** From LLMs to diffusion models, audio, embedding, and reranker models.
- **Scales with Your GPU Inventory:** Easily add more GPUs or nodes to scale up your operations.
- **Distributed Inference**: Supports both single-node multi-GPU and multi-node inference and serving.
- **Multiple Inference Backends**: Supports llama-box (llama.cpp & stable-diffusion.cpp), vox-box and vLLM as the inference backends.
- **Lightweight Python Package:** Minimal dependencies and operational overhead.
- **OpenAI-compatible APIs:** Serve APIs that are compatible with OpenAI standards.
- **User and API key management:** Simplified management of users and API keys.
- **GPU metrics monitoring:** Monitor GPU performance and utilization in real-time.
- **Token usage and rate metrics:** Track token usage and manage rate limits effectively.

## Supported Platforms

- [x] macOS
- [x] Windows
- [x] Linux

## Supported Accelerators

- [x] NVIDIA CUDA ([Compute Capability](https://developer.nvidia.com/cuda-gpus) 6.0 and above)
- [x] Apple Metal (M-series chips)
- [x] AMD ROCm
- [x] Ascend CANN
- [x] Hygon DTK
- [x] Moore Threads MUSA

We plan to support the following accelerators in future releases.

- [ ] Intel oneAPI
- [ ] Qualcomm AI Engine

## Supported Models

GPUStack uses [llama-box](https://github.com/gpustack/llama-box) (bundled [llama.cpp](https://github.com/ggml-org/llama.cpp) and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) server), [vLLM](https://github.com/vllm-project/vllm) and [vox-box](https://github.com/gpustack/vox-box) as the backends and supports a wide range of models. Models from the following sources are supported:

1. [Hugging Face](https://huggingface.co/)

2. [ModelScope](https://modelscope.cn/)

3. Local File Path

### Example Models:

| **Category**                     | **Models**                                                                                                                                                                                                                                                                                                                                           |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Large Language Models(LLMs)**  | [Qwen](https://huggingface.co/models?search=Qwen/Qwen), [LLaMA](https://huggingface.co/meta-llama), [Mistral](https://huggingface.co/mistralai), [DeepSeek](https://huggingface.co/models?search=deepseek-ai/deepseek), [Phi](https://huggingface.co/models?search=microsoft/phi), [Gemma](https://huggingface.co/models?search=Google/gemma)        |
| **Vision Language Models(VLMs)** | [Llama3.2-Vision](https://huggingface.co/models?pipeline_tag=image-text-to-text&search=llama3.2), [Pixtral](https://huggingface.co/models?search=pixtral) , [Qwen2.5-VL](https://huggingface.co/models?search=Qwen/Qwen2.5-VL), [LLaVA](https://huggingface.co/models?search=llava), [InternVL2.5](https://huggingface.co/models?search=internvl2_5) |
| **Diffusion Models**             | [Stable Diffusion](https://huggingface.co/models?search=gpustack/stable-diffusion), [FLUX](https://huggingface.co/models?search=gpustack/flux)                                                                                                                                                                                                       |
| **Embedding Models**             | [BGE](https://huggingface.co/gpustack/bge-m3-GGUF), [BCE](https://huggingface.co/gpustack/bce-embedding-base_v1-GGUF), [Jina](https://huggingface.co/models?search=gpustack/jina-embeddings)                                                                                                                                                         |
| **Reranker Models**              | [BGE](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF), [BCE](https://huggingface.co/gpustack/bce-reranker-base_v1-GGUF), [Jina](https://huggingface.co/models?search=gpustack/jina-reranker)                                                                                                                                                |
| **Audio Models**                 | [Whisper](https://huggingface.co/models?search=Systran/faster) (Speech-to-Text), [CosyVoice](https://huggingface.co/models?search=FunAudioLLM/CosyVoice) (Text-to-Speech)                                                                                                                                                                            |

For full list of supported models, please refer to the supported models section in the [inference backends](./user-guide/inference-backends.md) documentation.

## OpenAI-Compatible APIs

GPUStack serves OpenAI compatible APIs. For details, please refer to [OpenAI Compatible APIs](./user-guide/openai-compatible-apis.md)
