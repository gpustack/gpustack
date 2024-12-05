# Inference Backends

GPUStack supports the following inference backends:

- llama-box
- vLLM
- vox-box

When users deploy a model, the backend is selected automatically based on the following criteria:

- If the model is a [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) model, `llama-box` is used.
- If the model is a known `text-to-speech` or `speech-to-text` model, `vox-box` is used.
- Otherwise, `vLLM` is used.

## llama-box

[llama-box](https://github.com/gpustack/llama-box) is a LM inference server based on [llama.cpp](https://github.com/ggerganov/llama.cpp) and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

### Supported Platforms

The llama-box backend supports Linux, macOS and Windows (with CPU offloading only on Windows ARM architecture) platforms.

### Supported Models

- LLMs: For supported LLMs, refer to the llama.cpp [README](https://github.com/ggerganov/llama.cpp#description).
- Difussion Models: Supported models are listed in this [Hugging Face collection](https://huggingface.co/collections/gpustack/image-672dafeb2fa0d02dbe2539a9).
- Reranker Models: Supported models can be found in this [Hugging Face collection](https://huggingface.co/collections/gpustack/reranker-6721a234527f6fcd90deedc4).

### Supported Features

#### Allow CPU Offloading

After enabling CPU offloading, GPUStack prioritizes loading as many layers as possible onto the GPU to optimize performance. If GPU resources are limited, some layers will be offloaded to the CPU, with full CPU inference used only when no GPU is available.

#### Allow Distributed Inference Across Workers

Enable distributed inference across multiple workers. The primary Model Instance will communicate with backend instances on one or more others workers, offloading computation tasks to them.

### Parameters Reference

See the full list of supported parameters for llama-box [here](https://github.com/gpustack/llama-box#usage).

## vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput and memory-efficient LLMs inference engine. It is a popular choice for running LLMs in production. vLLM seamlessly supports most state-of-the-art open-source models, including: Transformer-like LLMs (e.g., Llama), Mixture-of-Expert LLMs (e.g., Mixtral), Embedding Models (e.g. E5-Mistral), Multi-modal LLMs (e.g., LLaVA)

By default, GPUStack estimates the VRAM requirement for the model instance based on the model's metadata. You can customize the parameters to fit your needs. The following vLLM parameters might be useful:

- `--gpu-memory-utilization` (default: 0.9): The fraction of GPU memory to use for the model instance.
- `--max-model-len`: Model context length. For large-context models, GPUStack automatically sets this parameter to `8192` to simplify model deployment, especially in resource constrained environments. You can customize this parameter to fit your needs.
- `--tensor-parallel-size`: Number of tensor parallel replicas. By default, GPUStack sets this parameter given the GPU resources available and the estimation of the model's memory requirement. You can customize this parameter to fit your needs.

For more details, please refer to [vLLM documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#command-line-arguments-for-the-server).

### Supported Platforms

The vLLM backend works on AMD Linux.

!!! note

    1. When users install GPUStack on amd64 Linux using the installation script, vLLM is automatically installed.
    2. When users deploy a model using the vLLM backend, GPUStack sets worker label selectors to `{"os": "linux", "arch": "amd64"}` by default to ensure the model instance is scheduled to proper workers. You can customize the worker label selectors in the model configuration.

### Supported Models

Please refer to the vLLM [documentation](https://docs.vllm.ai/en/stable/models/supported_models.html) for supported models.

### Supported Features

#### Multimodal Language Models

vLLM supports multimodal language models listed [here](https://docs.vllm.ai/en/stable/models/supported_models.html#multimodal-language-models). When users deploy a vision language model using the vLLM backend, image inputs are supported in the chat completion API.

### Parameters Reference

See the full list of supported parameters for vLLM [here](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#command-line-arguments-for-the-server).

## vox-box

[vox-box](https://github.com/gpustack/vox-box) is an inference engine designed for deploying text-to-speech and speech-to-text models. It also provides an API that is fully compatible with the OpenAI audio API.

### Supported Platforms

The vox-box backend supports Linux, macOS and Windows platforms.

!!! note

    1. To use Nvidia GPUs, ensure the following NVIDIA libraries are installed on workers:
        - [cuBLAS for CUDA 12](https://developer.nvidia.com/cublas)
        - [cuDNN 9 for CUDA 12](https://developer.nvidia.com/cudnn)
    2. When users install GPUStack on Linux, macOS and Windows using the installation script, vox-box is automatically installed.
    3. CosyVoice models are natively supported on Linux AMD architecture and macOS. However, these models are not supported on Linux ARM or Windows architectures.

### Supported Models

| Model                           | Type           | Link                                                                                                                                               | Supported Platforms                                     |
| ------------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Faster-whisper-large-v3         | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3)                                                                             | Linux, macOS, Windows                                   |
| Faster-whisper-large-v2         | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v2)                                                                             | Linux, macOS, Windows                                   |
| Faster-whisper-large-v1         | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v1)                                                                             | Linux, macOS, Windows                                   |
| Faster-whisper-medium           | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium)                                                                               | Linux, macOS, Windows                                   |
| Faster-whisper-medium.en        | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium.en)                                                                            | Linux, macOS, Windows                                   |
| Faster-whisper-small            | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small)                                                                                | Linux, macOS, Windows                                   |
| Faster-whisper-small.en         | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small.en)                                                                             | Linux, macOS, Windows                                   |
| Faster-distil-whisper-large-v3  | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v3)                                                                      | Linux, macOS, Windows                                   |
| Faster-distil-whisper-large-v2  | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v2)                                                                      | Linux, macOS, Windows                                   |
| Faster-distil-whisper-medium.en | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-medium.en)                                                                     | Linux, macOS, Windows                                   |
| Faster-whisper-tiny             | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny)                                                                                 | Linux, macOS, Windows                                   |
| Faster-whisper-tiny.en          | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny.en)                                                                              | Linux, macOS, Windows                                   |
| CosyVoice-300M-Instruct         | text-to-speech | [Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice-300M-Instruct), [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-Instruct) | Linux(ARM not supported), macOS, Windows(Not supported) |
| CosyVoice-300M-SFT              | text-to-speech | [Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT), [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-SFT)           | Linux(ARM not supported), macOS, Windows(Not supported) |
| CosyVoice-300M                  | text-to-speech | [Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice-300M), [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M)                   | Linux(ARM not supported), macOS, Windows(Not supported) |
| CosyVoice-300M-25Hz             | text-to-speech | [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-25Hz)                                                                                 | Linux(ARM not supported), macOS, Windows(Not supported) |

### Supported Features

#### Allow GPU/CPU Offloading

vox-box supports deploying models to NVIDIA GPUs. If GPU resources are insufficient, it will automatically deploy the models to the CPU.
