# Inference Backends

GPUStack supports the following inference backends:

- [llama-box](#llama-box)
- [vLLM](#vllm)
- [vox-box](#vox-box)
- [Ascend MindIE](#ascend-mindie-experimental)

When users deploy a model, the backend is selected automatically based on the following criteria:

- If the model is a [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) model, `llama-box` is used.
- If the model is a known `Text-to-Speech` or `Speech-to-Text` model, `vox-box` is used.
- Otherwise, `vLLM` is used.

## llama-box

[llama-box](https://github.com/gpustack/llama-box) is a LM inference server based on [llama.cpp](https://github.com/ggml-org/llama.cpp) and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

### Supported Platforms

The llama-box backend supports Linux, macOS and Windows (with CPU offloading only on Windows ARM architecture) platforms.

### Supported Models

- LLMs: For supported LLMs, refer to the llama.cpp [README](https://github.com/ggml-org/llama.cpp#description).
- Diffussion Models: Supported models are listed in this [Hugging Face collection](https://huggingface.co/collections/gpustack/image-672dafeb2fa0d02dbe2539a9) or this [ModelScope collection](https://modelscope.cn/collections/Image-fab3d241f8a641).
- Reranker Models: Supported models can be found in this [Hugging Face collection](https://huggingface.co/collections/gpustack/reranker-6721a234527f6fcd90deedc4) or this [ModelScope collection](https://modelscope.cn/collections/Reranker-7576210e79de4a).

### Supported Features

#### Allow CPU Offloading

After enabling CPU offloading, GPUStack prioritizes loading as many layers as possible onto the GPU to optimize performance. If GPU resources are limited, some layers will be offloaded to the CPU, with full CPU inference used only when no GPU is available.

#### Allow Distributed Inference Across Workers

Enable distributed inference across multiple workers. The primary Model Instance will communicate with backend instances on one or more others workers, offloading computation tasks to them.

#### Multimodal Language Models

Llama-box supports the following vision language models. When using a vision language model, image inputs are supported in the chat completion API.

- LLaVA Series
- MiniCPM VL Series
- Qwen2 VL Series
- GLM-Edge-V Series
- Granite VL Series
- Gemma3 VL Series
- SmolVLM Series
- Pixtral Series
- MobileVLM Series
- Mistral Small 3.1
- Qwen2.5 VL Series

!!! Note

    When deploying a vision language model, GPUStack downloads and uses the multimodal projector file with the pattern `*mmproj*.gguf` by default. If multiple files match the pattern, GPUStack selects the file with higher precision (e.g., `f32` over `f16`). If the default pattern does not match the projector file or you want to use a specific one, you can customize the multimodal projector file by setting the `--mmproj` parameter in the model configuration. You can specify the relative path to the projector file in the model source. This syntax acts as shorthand, and GPUStack will download the file from the source and normalize the path when using it.

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

The vLLM backend works on AMD64 Linux.

!!! Note

    1. When users install GPUStack on amd64 Linux using the installation script, vLLM is automatically installed.
    2. When users deploy a model using the vLLM backend, GPUStack sets worker label selectors to `{"os": "linux", "arch": "amd64"}` by default to ensure the model instance is scheduled to proper workers. You can customize the worker label selectors in the model configuration.

### Supported Models

Please refer to the vLLM [documentation](https://docs.vllm.ai/en/stable/models/supported_models.html) for supported models.

### Supported Features

#### Multimodal Language Models

vLLM supports multimodal language models listed [here](https://docs.vllm.ai/en/stable/models/supported_models.html#multimodal-language-models). When users deploy a vision language model using the vLLM backend, image inputs are supported in the chat completion API.

#### Distributed Inference Across Workers (Experimental)

vLLM supports distributed inference across multiple workers using [Ray](https://ray.io). You can enable a Ray cluster in GPUStack by using the `--enable-ray` start parameter, allowing vLLM to run distributed inference across multiple workers.

!!! warning "Known Limitations"

    1. The GPUStack server and all participating workers must run on Linux and use the same version of Python, which is a requirement of Ray.
    2. Model files must be accessible at the same path on all participating workers. You must either use a shared file system or download the model files to the same path on all participating workers.
    3. Each worker can only be assigned to one distributed vLLM model instance at a time.
    4. A custom vLLM version may not work if its Ray distributed executor implementation is incompatible with the built-in vLLM version.
    5. If you install GPUStack with Docker, you must use the host network mode to leverage RDMA/InfiniBand and ensure connectivity between nodes.

Auto-scheduling is supported with the following conditions:

- Participating workers have the same number of GPUs.
- All GPUs in the worker satisfy the gpu_memory_utilization(defaults to 0.9) requirement.
- The total number of GPUs can be divided by the number of attention heads.
- The total VRAM claim is greater than the estimated VRAM claim.

If the above conditions are not met, the model instance will not be scheduled automatically. However, you can manually schedule it by selecting the desired workers/GPUs in the model configuration.

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

| Model                           | Type           | Link                                                                                                                                                                | Supported Platforms                                     |
| ------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Faster-whisper-large-v3         | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v3)                 | Linux, macOS, Windows                                   |
| Faster-whisper-large-v2         | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v2), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v2)                 | Linux, macOS, Windows                                   |
| Faster-whisper-large-v1         | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v1), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v1)                 | Linux, macOS, Windows                                   |
| Faster-whisper-medium           | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-medium)                     | Linux, macOS, Windows                                   |
| Faster-whisper-medium.en        | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-medium.en)               | Linux, macOS, Windows                                   |
| Faster-whisper-small            | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-small)                       | Linux, macOS, Windows                                   |
| Faster-whisper-small.en         | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-small.en)                 | Linux, macOS, Windows                                   |
| Faster-distil-whisper-large-v3  | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v3), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-large-v3)   | Linux, macOS, Windows                                   |
| Faster-distil-whisper-large-v2  | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v2), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-large-v2)   | Linux, macOS, Windows                                   |
| Faster-distil-whisper-medium.en | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-medium.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-medium.en) | Linux, macOS, Windows                                   |
| Faster-whisper-tiny             | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-tiny)                         | Linux, macOS, Windows                                   |
| Faster-whisper-tiny.en          | speech-to-text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-tiny.en)                   | Linux, macOS, Windows                                   |
| CosyVoice-300M-Instruct         | text-to-speech | [Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice-300M-Instruct), [ModelScope](https://modelscope.cn/models/gpustack/CosyVoice-300M-Instruct)             | Linux(ARM not supported), macOS, Windows(Not supported) |
| CosyVoice-300M-SFT              | text-to-speech | [Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT), [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-SFT)                            | Linux(ARM not supported), macOS, Windows(Not supported) |
| CosyVoice-300M                  | text-to-speech | [Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice-300M), [ModelScope](https://modelscope.cn/models/gpustack/CosyVoice-300M)                               | Linux(ARM not supported), macOS, Windows(Not supported) |
| CosyVoice-300M-25Hz             | text-to-speech | [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-25Hz)                                                                                                  | Linux(ARM not supported), macOS, Windows(Not supported) |
| CosyVoice2-0.5B                 | text-to-speech | [Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B), [ModelScope](https://modelscope.cn/models/iic/CosyVoice2-0.5B)                                  | Linux(ARM not supported), macOS, Windows(Not supported) |
| Dia-1.6B                        | text-to-speech | [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B), [ModelScope](https://modelscope.cn/models/nari-labs/Dia-1.6B)                                            | Linux(ARM not supported), macOS, Windows(Not supported) |

### Supported Features

#### Allow GPU/CPU Offloading

vox-box supports deploying models to NVIDIA GPUs. If GPU resources are insufficient, it will automatically deploy the models to the CPU.

## Ascend MindIE (Experimental)

[Ascend MindIE](https://www.hiascend.com/en/software/mindie) is a high-performance inference service
on [Ascend hardware](https://www.hiascend.com/en/hardware/product).

### Supported Platforms

The Ascend MindIE backend works on Linux platforms only, including ARM64 and x86_64 architectures.

### Supported Models

Ascend MindIE supports various models
listed [here](https://www.hiascend.com/document/detail/zh/mindie/20RC1/modellist/mindie_modellist_0001.html).

Within GPUStack, support
[large language models (LLMs)](https://www.hiascend.com/document/detail/zh/mindie/20RC1/modellist/mindie_modellist_0001.html)
and
[multimodal language models (VLMs)](https://www.hiascend.com/document/detail/zh/mindie/20RC1/modellist/mindie_modellist_0002.html)
. However, _embedding models_ and _multimodal generation models_ are not supported yet.

### Supported Features

Ascend MindIE owns a variety of features
outlined [here](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0001.html).

At present, GPUStack supports a subset of these capabilities, including
[Quantization](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0288.html),
[Extending Context Size](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0295.html),
[Mixture of Experts(MoE)](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0297.html),
[Prefix Caching](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0302.html),
[Function Calling](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0303.html),
[Multimodal Understanding](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0304.html),
[Multi-head Latent Attention(MLA)](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0305.html).

!!! Note

    1. Quantization needs specific weight, and must adjust the model's `config.json`,
       please follow the [reference(guide)](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0288.html) to prepare the correct weight.
    2. For Multimodal Understanding feature, some versions of Ascend MindIE's API are incompatible with OpenAI,
       please track this [issue](https://github.com/gpustack/gpustack/issues/1803) for more support.
    3. Some features are mutually exclusive, so be careful when using them.

### Parameters Reference

Ascend MindIE has
configurable [parameters](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0004.html)
and [environment variables](https://www.hiascend.com/document/detail/zh/mindie/20RC1/mindiellm/llmdev/mindie_llm0416.html).

To avoid directly configuring JSON, GPUStack provides a set of command line parameters as below.

| Parameter                        | Default | Description                                                                                                                                                                                            |
| -------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--trust-remote-code`            |         | Trust remote code (for model loading).                                                                                                                                                                 |
| `--npu-memory-fraction`          | 0.9     | Fraction of NPU memory to be used for the model executor (0 to 1). For example: `0.5` means 50% memory utilization.                                                                                    |
| `--max-link-num`                 | 1000    | Maximum number of parallel requests.                                                                                                                                                                   |
| `--token-timeout`                | 60      | Timeout for a token generation in seconds.                                                                                                                                                             |
| `--e2e-timeout`                  | 60      | E2E (from request accepted to inference stopped) timeout in seconds.                                                                                                                                   |
| `--max-seq-len`                  | 8192    | Model context length. If unspecified, it will be derived from the model config.                                                                                                                        |
| `--max-input-token-len`          |         | Maximum input token length. If unspecified, it will be derived from `--max-seq-len`.                                                                                                                   |
| `--truncation`                   |         | Truncate the input token length when it exceeds the minimum of `--max-input-token-len` and `--max-seq-len` - 1.                                                                                        |
| `--cpu-mem-size`                 | 5       | CPU swap space size in GiB. If unspecified, the default value will be used.                                                                                                                            |
| `--cache-block-size`             | 128     | KV cache block size. Must be a power of 2.                                                                                                                                                             |
| `--max-batch-size`               | 200     | Maximum number of requests batched during decode stage.                                                                                                                                                |
| `--max-prefill-batch-size`       | 50      | Maximum number of requests batched during prefill stage. Must be less than `--max-batch-size`.                                                                                                         |
| `--max-preempt-count`            | 0       | Maximum number of preempted requests allowed during decoding. Must be less than `--max-batch-size`.                                                                                                    |
| `--max-queue-delay-microseconds` | 5000    | Maximum queue wait time in microseconds.                                                                                                                                                               |
| `--prefill-time-ms-per-req`      | 150     | Estimated prefill time per request (ms). Used to decide between prefill and decode stage.                                                                                                              |
| `--prefill-policy-type`          | 0       | Prefill stage strategy: <br> `0`: FCFS (First Come First Serve). <br> `1`: STATE (same as FCFS). <br> `2`: PRIORITY (priority queue). <br> `3`: MLFQ (Multi-Level Feedback Queue).                     |
| `--decode-time-ms-per-req`       | 50      | Estimated decode time per request (ms). Used with `--prefill-time-ms-per-req` for batch selection.                                                                                                     |
| `--decode-policy-type`           | 0       | Decode stage strategy: <br> `0`: FCFS <br> `1`: STATE (prioritize preempted or swapped requests) <br> `2`: PRIORITY <br> `3`: MLFQ                                                                     |
| `--support-select-batch`         |         | Enable batch selection. Determines execution priority based on `--prefill-time-ms-per-req` and `--decode-time-ms-per-req`.                                                                             |
| `--enable-prefix-caching`        |         | Enable prefix caching. Use `--no-enable-prefix-caching` to disable explicitly.                                                                                                                         |
| `--enforce-eager`                |         | Emit operators in eager mode.                                                                                                                                                                          |
| `--dtype`                        | auto    | Data type for model weights and activations. <br> `auto`: use the default data type of the model config. <br> `half`/`float16`: for FP16. <br> `bfloat16`: for BF16. <br> `float`/`float32`: for FP32. |
| `--rope-scaling`                 |         | RoPE scaling configuration in JSON format. For example: `{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}`. This will merge into the `config.json` of the model structure.        |
| `--rope-theta`                   |         | RoPE theta configuration. This will merge into the `config.json` of the model structure.                                                                                                               |
| `--override-generation-config`   |         | Overrides or sets generation config in JSON format. For example: `{"temperature": 0.5}`. This will merge into the `generation_config.json` of the model structure.                                     |
| `--metrics`                      |         | Expose metrics at `/metrics` endpoint.                                                                                                                                                                 |
| `--log-level`                    | Info    | Log level for MindIE. Options: `Verbose`, `Info`, `Warning`, `Warn`, `Error`, `Debug`.                                                                                                                 |

!!! Note

    GPUStack allows users to inject custom environment variables during model deployment,
    however, some variables may be conflicted with GPUStack managment.
    Hence, GPUStack will override/prevent those variables.
    Please compare the model instance logs' output with your expectations.
