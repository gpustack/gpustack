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

The vLLM backend works on Linux.

!!! Note

    1. When users install GPUStack on amd64 Linux using the installation script, vLLM is automatically installed.
    2. When users deploy a model using the vLLM backend, GPUStack sets worker label selectors to `{"os": "linux"}` by default to ensure the model instance is scheduled to proper workers. You can customize the worker label selectors in the model configuration.

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

The Ascend MindIE backend defaults to Ascend MindIE 2.0.RC2, 
which is compatible with Linux platforms only, including both ARM64 and x86_64 architectures.

### Supported Models

Ascend MindIE supports various models
listed [here](https://www.hiascend.com/software/mindie/modellist).

Within GPUStack, support
[large language models (LLMs)](https://www.hiascend.com/software/mindie/modellist)
and
[multimodal language models (VLMs)](https://www.hiascend.com/software/mindie/modellist)
. However, _embedding models_ and _multimodal generation models_ are not supported yet.

### Supported Features

Ascend MindIE owns a variety of features
outlined [here](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0001.html).

At present, GPUStack supports a subset of these capabilities, including
[Quantization](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0288.html),
[Extending Context Size](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0295.html),
[Mixture of Experts(MoE)](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0297.html),
[Split Fuse](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0300.html),
[Speculative Decoding](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0301.html),
[Prefix Caching](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0302.html),
[Function Calling](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0303.html),
[Multimodal Understanding](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0304.html),
[Multi-head Latent Attention(MLA)](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0305.html),
[Data Parallelism](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0424.html),
[Buffer Response(Since Ascend MindIE 2.0.RC1)](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0425.html).

!!! Note

    1. Quantization needs specific weight, and must adjust the model's `config.json`,
       please follow the [reference(guide)](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0288.html) to prepare the correct weight.
    2. Some features are mutually exclusive, so be careful when using them. 
       For example, with Prefix Caching enabled, the Extending Context Size feature cannot be used.

### Parameters Reference

Ascend MindIE has
configurable [parameters](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0004.html)
and [environment variables](https://www.hiascend.com/document/detail/zh/mindie/20RC2/mindiellm/llmdev/mindie_llm0416.html).

To avoid directly configuring JSON, GPUStack provides a set of command line parameters as below.

| Parameter                        | Default | Range                          | Scope                                  | Description                                                                                                                                                                                                                                                                     |
|----------------------------------|---------|--------------------------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--log-level`                    | Info    |                                | Log Config                             | Log level for MindIE. Options: `Verbose`, `Info`, `Warning`, `Warn`, `Error`, `Debug`.                                                                                                                                                                                          |
| `--max-link-num`                 | 1000    | [1, 1000]                      | Server Config                          | Maximum number of parallel requests.                                                                                                                                                                                                                                            |
| `--token-timeout`                | 60      | [1, 3600]                      | Server Config                          | Timeout for a token generation in seconds.                                                                                                                                                                                                                                      |
| `--e2e-timeout`                  | 60      | [1, 3600]                      | Server Config                          | E2E (from request accepted to inference stopped) timeout in seconds.                                                                                                                                                                                                            |
| `--max-seq-len`                  | 8192    | (0, ]                          | Model Deploy Config                    | Model context length. If unspecified, it will be derived from the model config.                                                                                                                                                                                                 |
| `--max-input-token-len`          |         | (0, `--max-seq-len`]           | Model Deploy Config                    | Maximum input token length. If unspecified, it will be derived from `--max-seq-len`.                                                                                                                                                                                            |
| `--truncation`                   |         |                                | Model Deploy Config                    | Truncate the input token length when it exceeds the minimum of `--max-input-token-len` and `--max-seq-len` - 1.                                                                                                                                                                 |
| `--cpu-mem-size`                 | 5       | (0, ]                          | Model Config                           | CPU swap space size in GiB. If unspecified, the default value will be used.                                                                                                                                                                                                     |
| `--npu-memory-fraction`          | 0.9     | (0, 1]                         | Model Config                           | Fraction of NPU memory to be used for the model executor (0 to 1). For example: `0.5` means 50% memory utilization.                                                                                                                                                             |
| `--trust-remote-code`            |         |                                | Model Config                           | Trust remote code (for model loading).                                                                                                                                                                                                                                          |
| `--cache-block-size`             | 128     |                                | Schedule Config                        | KV cache block size. Must be a power of 2.                                                                                                                                                                                                                                      |
| `--max-prefill-batch-size`       | 50      | [1, `--max-batch-size`]        | Schedule Config                        | Maximum number of requests batched during prefill stage. Must be less than `--max-batch-size`.                                                                                                                                                                                  |
| `--prefill-time-ms-per-req`      | 150     | [0, 1000]                      | Schedule Config                        | Estimated prefill time per request (ms). Used to decide between prefill and decode stage.                                                                                                                                                                                       |
| `--prefill-policy-type`          | 0       |                                | Schedule Config                        | Prefill stage strategy: <br> `0`: FCFS (First Come First Serve). <br> `1`: STATE (same as FCFS). <br> `2`: PRIORITY (priority queue). <br> `3`: MLFQ (Multi-Level Feedback Queue).                                                                                              |
| `--max-batch-size`               | 200     | [1, 5000]                      | Schedule Config                        | Maximum number of requests batched during decode stage.                                                                                                                                                                                                                         |
| `--decode-time-ms-per-req`       | 50      | [0, 1000]                      | Schedule Config                        | Estimated decode time per request (ms). Used with `--prefill-time-ms-per-req` for batch selection.                                                                                                                                                                              |
| `--decode-policy-type`           | 0       |                                | Schedule Config                        | Decode stage strategy: <br> `0`: FCFS <br> `1`: STATE (prioritize preempted or swapped requests) <br> `2`: PRIORITY <br> `3`: MLFQ                                                                                                                                              |
| `--max-preempt-count`            | 0       | [0, `--max-batch-size`]        | Schedule Config                        | Maximum number of preempted requests allowed during decoding. Must be less than `--max-batch-size`.                                                                                                                                                                             |
| `--support-select-batch`         |         |                                | Schedule Config                        | Enable batch selection. Determines execution priority based on `--prefill-time-ms-per-req` and `--decode-time-ms-per-req`. Use `--no-support-select-batch` to disable explicitly.                                                                                               |
| `--max-queue-delay-microseconds` | 5000    | [500, 1000000]                 | Schedule Config                        | Maximum queue wait time in microseconds.                                                                                                                                                                                                                                        |
| `--override-generation-config`   |         |                                |                                        | Overrides or sets generation config in JSON format. For example: `{"temperature": 0.5}`. This will merge into the `generation_config.json` of the model structure.                                                                                                              |
| `--enforce-eager`                |         |                                |                                        | Emit operators in eager mode.                                                                                                                                                                                                                                                   |
| `--metrics`                      |         |                                |                                        | Expose metrics at `/metrics` endpoint.                                                                                                                                                                                                                                          |
| `--dtype`                        | auto    |                                |                                        | Data type for model weights and activations. <br> `auto`: use the default data type of the model config. <br> `half`/`float16`: for FP16. <br> `bfloat16`: for BF16. <br> `float`/`float32`: for FP32.                                                                          |
| `--rope-scaling`                 |         |                                | Extending Context Size                 | RoPE scaling configuration in JSON format. For example: `{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}`. This will merge into the `config.json` of the model structure.                                                                                 |
| `--rope-theta`                   |         |                                | Extending Context Size                 | RoPE theta configuration. This will merge into the `config.json` of the model structure.                                                                                                                                                                                        |
| `--enable-split`                 |         |                                | Split Fuse                             | Enable split fuse, something like chunked prefill. Use `--no-enable-split` to disable explicitly.                                                                                                                                                                               |
| `--policy-type`                  | 0       |                                | Split Fuse                             | Strategy of split fuse. <br> `0`: FCFS, first come first serving. <br> `4`: SJF, shortest job first. <br> `5`: LJF, longest job first. <br> `6`: Skip-Join MLFQ, skip-Join multi-levels feedback queue. <br> `7`: SJF-MLFQ, shortest job first and multi-levels feedback queue. |
| `--split-chunk-tokens`           | 512     | [512, `--max-input-token-len`] | Split Fuse                             | Tokens size to batch for split fuse.                                                                                                                                                                                                                                            |
| `--split-start-batch-size`       | 16      | [0, `--max-batch-size`]        | Split Fuse                             | Batch size to start splitting for split fuse.                                                                                                                                                                                                                                   |
| `--enable-memory-decoding`       |         |                                | Speculative Decoding / Memory Decoding | Enable memory decoding speculation. Use `--no-enable-memory-decoding` to disable explicitly.                                                                                                                                                                                    |
| `--memory-decoding-length`       | 16      | [1, 16]                        | Speculative Decoding / Memory Decoding | Length for memory decoding speculation.                                                                                                                                                                                                                                         |
| `--memory-decoding-dynamic-algo` |         |                                | Speculative Decoding / Memory Decoding | Enable dynamic algorithm for memory decoding speculation.                                                                                                                                                                                                                       |
| `--enable-lookahead`             |         |                                | Speculative Decoding / Lookahead       | Enable lookahead speculation. Use `--no-enable-lookahead` to disable explicitly.                                                                                                                                                                                                |
| `--lookahead-level`              | 4       | [3, 16]                        | Speculative Decoding / Lookahead       | Level for lookahead speculation.                                                                                                                                                                                                                                                |
| `--lookahead-window`             | 5       | [1, 16]                        | Speculative Decoding / Lookahead       | Window size for lookahead speculation.                                                                                                                                                                                                                                          |
| `--lookahead-guess-set-size`     | 5       | [1, 16]                        | Speculative Decoding / Lookahead       | Guess set size for lookahead speculation.                                                                                                                                                                                                                                       |
| `--enable-prefix-caching`        |         |                                | Prefix Caching                         | Enable prefix caching. Use `--no-enable-prefix-caching` to disable explicitly.                                                                                                                                                                                                  |
| `--tensor-parallel-size`, `-tp`  | -1      |                                | Data Parallelism                       | Number of tensor parallel groups. `-1` means using world size as tensor parallel size, otherwise, must be a power of 2.                                                                                                                                                         |
| `--enable-expert-parallel`       |         |                                | Data Parallelism                       | Use expert parallelism instead of tensor parallelism for MoE layers. Use `--no-enable-expert-parallel` to disable explicitly.                                                                                                                                                   |
| `--data-parallel-size`, `-dp`    | -1      |                                | Data Parallelism                       | Number of data parallel groups. `-1` means disabling data parallelism, otherwise, must be a power of 2. MoE layers will be sharded according to the product of the tensor parallel size and data parallel size.                                                                 |
| `--enable-buffer-response`       |         |                                | Buffer Response                        | Enable buffer response. Use `--no-enable-buffer-response` to disable explicitly.                                                                                                                                                                                                |
| `--prefill-expected-time-ms`     |         |                                | Buffer Response                        | Expected latency (SLO) for Time to First Token (TTFT) in milliseconds.                                                                                                                                                                                                          |
| `--decode-expected-time-ms`      |         |                                | Buffer Response                        | Expected latency (SLO) for Time Per Output Token (TPOT) in milliseconds.                                                                                                                                                                                                        |

!!! Note

    GPUStack allows users to inject custom environment variables during model deployment,
    however, some variables may be conflicted with GPUStack managment.
    Hence, GPUStack will override/prevent those variables.
    Please compare the model instance logs' output with your expectations.
