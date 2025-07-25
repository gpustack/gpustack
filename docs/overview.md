# GPUStack

![demo](assets/gpustack-demo.gif)

GPUStack is an open-source GPU cluster manager for running AI models.

### Key Features

- **Broad GPU Compatibility:** Seamlessly supports GPUs from various vendors across Apple Macs, Windows PCs, and Linux servers.
- **Extensive Model Support:** Supports a wide range of models including LLMs, VLMs, image models, audio models, embedding models, and rerank models.
- **Flexible Inference Backends:** Flexibly integrates with multiple inference backends including vLLM, Ascend MindIE, llama-box (llama.cpp & stable-diffusion.cpp) and vox-box.
- **Multi-Version Backend Support:** Run multiple versions of inference backends concurrently to meet the diverse runtime requirements of different models.
- **Distributed Inference:** Supports single-node and multi-node multi-GPU inference, including heterogeneous GPUs across vendors and runtime environments.
- **Scalable GPU Architecture:** Easily scale up by adding more GPUs or nodes to your infrastructure.
- **Robust Model Stability:** Ensures high availability with automatic failure recovery, multi-instance redundancy, and load balancing for inference requests.
- **Intelligent Deployment Evaluation:** Automatically assess model resource requirements, backend and architecture compatibility, OS compatibility, and other deployment-related factors.
- **Automated Scheduling:** Dynamically allocate models based on available resources.
- **Lightweight Python Package:** Minimal dependencies and low operational overhead.
- **OpenAI-Compatible APIs:** Fully compatible with OpenAI’s API specifications for seamless integration.
- **User & API Key Management:** Simplified management of users and API keys.
- **Real-Time GPU Monitoring:** Track GPU performance and utilization in real time.
- **Token and Rate Metrics:** Monitor token usage and API request rates.

## Supported Platforms

- [x] Linux
- [x] macOS
- [x] Windows

## Supported Accelerators

- [x] NVIDIA CUDA ([Compute Capability](https://developer.nvidia.com/cuda-gpus) 6.0 and above)
- [x] Apple Metal (M-series chips)
- [x] AMD ROCm
- [x] Ascend CANN
- [x] Hygon DTK
- [x] Moore Threads MUSA
- [x] Iluvatar Corex
- [x] Cambricon MLU

## Supported Models

GPUStack uses [vLLM](https://github.com/vllm-project/vllm), [Ascend MindIE](https://www.hiascend.com/en/software/mindie), [llama-box](https://github.com/gpustack/llama-box) (bundled [llama.cpp](https://github.com/ggml-org/llama.cpp) and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) server) and [vox-box](https://github.com/gpustack/vox-box) as the backends and supports a wide range of models. Models from the following sources are supported:

1. [Hugging Face](https://huggingface.co/)

2. [ModelScope](https://modelscope.cn/)

3. Local File Path

### Example Models

| **Category**                     | **Models**                                                                                                                                                                                                                                                                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Large Language Models(LLMs)**  | [Qwen](https://huggingface.co/models?search=Qwen/Qwen), [LLaMA](https://huggingface.co/meta-llama), [Mistral](https://huggingface.co/mistralai), [DeepSeek](https://huggingface.co/models?search=deepseek-ai/deepseek), [Phi](https://huggingface.co/models?search=microsoft/phi), [Gemma](https://huggingface.co/models?search=Google/gemma)    |
| **Vision Language Models(VLMs)** | [Llama3.2-Vision](https://huggingface.co/models?pipeline_tag=image-text-to-text&search=llama3.2), [Pixtral](https://huggingface.co/models?search=pixtral) , [Qwen2.5-VL](https://huggingface.co/models?search=Qwen/Qwen2.5-VL), [LLaVA](https://huggingface.co/models?search=llava), [InternVL3](https://huggingface.co/models?search=internvl3) |
| **Diffusion Models**             | [Stable Diffusion](https://huggingface.co/models?search=gpustack/stable-diffusion), [FLUX](https://huggingface.co/models?search=gpustack/flux)                                                                                                                                                                                                   |
| **Embedding Models**             | [BGE](https://huggingface.co/gpustack/bge-m3-GGUF), [BCE](https://huggingface.co/gpustack/bce-embedding-base_v1-GGUF), [Jina](https://huggingface.co/models?search=gpustack/jina-embeddings), [Qwen3-Embedding](https://huggingface.co/models?search=qwen/qwen3-embedding)                                                                       |
| **Reranker Models**              | [BGE](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF), [BCE](https://huggingface.co/gpustack/bce-reranker-base_v1-GGUF), [Jina](https://huggingface.co/models?search=gpustack/jina-reranker), [Qwen3-Reranker](https://huggingface.co/models?search=qwen/qwen3-reranker)                                                                |
| **Audio Models**                 | [Whisper](https://huggingface.co/models?search=Systran/faster) (Speech-to-Text), [CosyVoice](https://huggingface.co/models?search=FunAudioLLM/CosyVoice) (Text-to-Speech)                                                                                                                                                                        |

For full list of supported models, please refer to the supported models section in the [inference backends](./user-guide/inference-backends.md) documentation.

## OpenAI-Compatible APIs

GPUStack serves OpenAI compatible APIs. For details, please refer to [OpenAI Compatible APIs](./user-guide/openai-compatible-apis.md)
