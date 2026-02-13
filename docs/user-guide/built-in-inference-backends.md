# Built-in Inference Backends

GPUStack supports the following inference backends:

- [vLLM](#vllm)
- [SGLang](#sglang)
- [MindIE](#mindie)
- [VoxBox](#voxbox)

When initiating a model deployment, the UI will automatically pre-select a backend based on the following criteria:

- If the model is a known `Text-to-Speech` or `Speech-to-Text` model, `VoxBox` is used.
- If the model is a known `Image` model, `SGLang` is used.
- Otherwise, `vLLM` is used.

This pre-selection only populates the default value in the deployment form. The actual backend used for deployment will be the one explicitly selected by the user when submitting the deployment request. Users can select from the list of supported backends provided above.

!!! Note

    For all supported inference backends, we have pre-built Docker images available at [DockerHub](https://hub.docker.com/r/gpustack/runner). When users deploy models, the system will automatically pull and run the corresponding images.

## vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput and memory-efficient LLMs inference backend. It is a popular choice for running LLMs in production.

vLLM seamlessly supports most state-of-the-art open-source models, including:

- Transformer-like LLMs (e.g., `Qwen3`)
- Mixture-of-Expert LLMs (e.g., `DeepSeek V3/R1`)
- Multi-modal LLMs (e.g., `Qwen3-VL`)
- Embedding Models (e.g. `Qwen3-Embedding`)
- Reranker Models (e.g. `Qwen3-Reranker`)

By default, GPUStack estimates the VRAM requirement for the model instance based on the model's metadata.

You can customize the parameters to fit your needs. The following vLLM parameters might be useful:

- `--gpu-memory-utilization` (default: 0.9): The fraction of GPU memory to use for the model instance.
- `--max-model-len`: Model context length. For large-context models, GPUStack automatically sets this parameter to `8192` to simplify model deployment, especially in resource constrained environments. You can customize this parameter to fit your needs.
- `--tensor-parallel-size`: Number of tensor parallel replicas. By default, GPUStack sets this parameter given the GPU resources available and the estimation of the model's memory requirement. You can customize this parameter to fit your needs.

For more details, please refer to [vLLM CLI Reference](https://docs.vllm.ai/en/stable/cli/serve/).

### Supported Models

Please refer to the vLLM [documentation](https://docs.vllm.ai/en/stable/models/supported_models.html) for supported models.

### Supported Features

#### Omni-Modal Support

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) is an extension of vLLM designed for omni-modal inference, supporting:

- **Omni Models**: Simultaneous processing of text, audio, images, and video (e.g., `Qwen3-Omni`)
- **Image Tasks**: Image generation and editing (e.g., `Z-Image-Turbo`)
- **Video Tasks**: Video generation and editing (e.g., `Wan2.2`)
- **Audio Tasks**: Speech synthesis, voice cloning, and more (e.g., `Qwen3-TTS`)

GPUStack integrates with vLLM-Omni to deliver a seamless experience for deploying and managing omni-modal models. When a model is deployed via the vLLM backend, GPUStack automatically detects whether it is omni-modal based on its metadata and sets the required parameters for vLLM-Omni.

#### Distributed Inference Across Workers (Experimental)

vLLM supports distributed inference across multiple workers using [Ray](https://ray.io). You can enable a Ray cluster in GPUStack by checking the `Allow Distributed Inference Across Workers` option when deploying a model. This allowing vLLM to run distributed inference across multiple workers.

!!! warning "Known Limitations"

    1. Model files must be accessible at the same path on all participating workers. You must either use a shared file system or download the model files to the same path on all participating workers.
    2. Each worker can only be assigned to one distributed vLLM model instance at a time.

Auto-scheduling is supported with the following conditions:

- Participating workers have the same number of GPUs.
- All GPUs in the worker satisfy the gpu_memory_utilization(defaults to 0.9) requirement.
- With tensor parallelism, GPUs are divided based on the number of attention heads.
- The total VRAM claim is greater than the estimated VRAM claim.

If the above conditions are not met, the model instance will not be scheduled automatically. However, you can manually schedule it by selecting the desired workers/GPUs in the model configuration.

### Parameters Reference

See the full list of supported parameters for vLLM [here](https://docs.vllm.ai/en/stable/cli/serve/).

## SGLang

[SGLang](https://github.com/sgl-project/sglang) is a high-performance serving framework for large language models and vision-language models.

It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters.

By default, GPUStack estimates the VRAM requirement for the model instance based on model metadata.

When needed, GPUStack also sets several parameters automatically for large-context models. Common SGLang parameters include:

- `--mem-fraction-static` (default: `0.9`): The per-GPU allocatable VRAM fraction. The scheduler uses this value for resource matching and candidate selection. You can override it via the model's `backend_parameters`.
- `--context-length`: Model context length. For large-context models, if the automatically estimated context length exceeds device capability, GPUStack sets this parameter to `8192` to simplify deployment in resource-constrained environments. You can customize this parameter as needed.
- `--tp-size`: Tensor parallel size. When not explicitly provided, GPUStack infers and sets this parameter based on the selected GPUs.
- `--pp-size`: Pipeline parallel size. In multi-node deployments, GPUStack determines a combination of `--tp-size` and `--pp-size` according to the model and cluster configuration.
- Multi-node arguments: `--nnodes`, `--node-rank`, `--dist-init-addr`. When distributed inference is enabled, GPUStack injects these arguments to initialize multi-node communication.

For more details, please refer to [SGLang documentation](https://docs.sglang.ai/index.html).

### Supported Models

Please refer to the SGLang [documentation](https://docs.sglang.ai/supported_models/generative_models.html) for supported models.

SGLang also supports image models. The ones we have verified include: Qwen-Image, Flux 1 Dev. For more information, please refer to the [SGLang Diffusion documentation](https://lmsys.org/blog/2025-11-07-sglang-diffusion/).

### Supported Features

#### Distributed Inference Across Workers (Experimental)

You can enable distributed SGLang inference across multiple workers in GPUStack.

!!! warning "Known Limitations"

    1. All participating nodes must run Linux, and images/environments should be compatible in Python version and communication libraries (e.g., NCCL).
    2. Model files must be accessible at the same path on all nodes. Use a shared file system or download the model to the same path on each node.

Auto-scheduling candidate selection considers the following conditions:

- Participating GPUs satisfy the `--mem-fraction-static` requirement (default 0.9).
- In single-node multi-GPU or multi-node multi-GPU scenarios, the total effective allocatable VRAM must meet the model's requirement.
- Model parallelism requirements must be met (e.g., total number of attention heads divisible by the tensor parallel size), otherwise the candidate is rejected.

If the above conditions are not met, you can still manually schedule the model instance by selecting workers/GPUs in the configuration, though overcommit risk may be indicated.

#### Diffusion Models (Experimental)

SGLang Diffusion is a high-performance inference engine designed specifically for diffusion models, aiming to accelerate the generation process of images and videos.
Through SGLang's parallel processing techniques and optimized kernels, it achieves up to 1.2x higher generation speed compared to mainstream baselines (e.g., Hugging Face Diffusers).

For more details, please refer to [SGLang Diffusion](https://lmsys.org/blog/2025-11-07-sglang-diffusion/).

#### Other Advanced Features

Additional advanced features are available, such as:

- [Speculative Decoding](https://docs.sglang.ai/advanced_features/speculative_decoding.html)
- [Hierarchical KV Caching](https://docs.sglang.ai/advanced_features/hicache.html)

Please refer to the official documentation for usage instructions.

### Parameters Reference

See the full list of supported parameters for SGLang [here](https://docs.sglang.ai/advanced_features/server_arguments.html).

## MindIE

[MindIE](https://www.hiascend.com/en/software/mindie) is a high-performance inference service on [Ascend hardware](https://www.hiascend.com/en/hardware/product).

### Supported Models

MindIE supports various models listed [here](https://www.hiascend.com/software/mindie/modellist).

Within GPUStack, support [large language models (LLMs)](https://www.hiascend.com/software/mindie/modellist) and [multimodal language models (VLMs)](https://www.hiascend.com/software/mindie/modellist).

However, _embedding models_ and _multimodal generation models_ are not supported yet.

### Supported Features

MindIE owns a variety of features outlined [here](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0001.html).

At present, GPUStack supports a subset of these capabilities, including
[Quantization](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0279.html),
[Extending Context Size](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0295.html),
[Distributed Inference](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0296.html),
[Mixture of Experts(MoE)](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0297.html),
[Split Fuse](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0300.html),
[Speculative Decoding](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0301.html),
[Multi-Token Prediction](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0500.html),
[Prefix Caching](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0302.html),
[Function Calling](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0303.html),
[Multimodal Understanding](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0304.html),
[Multi-head Latent Attention(MLA)](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0305.html),
[Tensor Parallelism](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0511.html),
[Context Parallelism](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0508.html),
[Sequence Parallelism](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0517.html),
[Expert Parallelism](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0509.html),
[Data Parallelism](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0424.html),
[Buffer Response(Since Ascend MindIE 2.0.RC1)](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0425.html).
[KV Pool(Since Ascend MindIE 2.2.RC1)](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0538.html).

!!! note

    1. Quantization needs specific weight, and must adjust the model's `config.json`, please follow the [reference(guide)](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0279.html) to prepare the correct weight.

    2. Some features are mutually exclusive, so be careful when using them. For example, with Prefix Caching enabled, the Extending Context Size feature cannot be used.

### Parameters Reference

MindIE has configurable [parameters](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_service0285.html) and [environment variables](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0416.html).

To avoid directly configuring JSON, GPUStack provides a set of command line parameters as below.

| Parameter                                            | Default | Range                    | Scope                                  | Description                                                                                                                                                                                                                                                                     |
|------------------------------------------------------|---------|--------------------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--log-level`                                        | Info    |                          | Log Config                             | Log level for MindIE. Options: `Verbose`, `Info`, `Warning`, `Warn`, `Error`, `Debug`.                                                                                                                                                                                          |
| `--max-link-num`                                     | 1000    | [1, 1000]                | Server Config                          | Maximum number of parallel requests.                                                                                                                                                                                                                                            |
| `--token-timeout`                                    | 60      | [1, 3600]                | Server Config                          | Timeout for a token generation in seconds.                                                                                                                                                                                                                                      |
| `--e2e-timeout`                                      | 60      | [1, 3600]                | Server Config                          | E2E (from request accepted to inference stopped) timeout in seconds.                                                                                                                                                                                                            |
| `--kv-pool-config`                                   |         |                          | Backend Config                         | KV pool configuration in JSON format. For example: `{"backend":"<KV pool backend name>", "configPath":"/path/to/your/config/file"}`.                                                                                                                                            |
| `--max-seq-len`                                      | 8192    | (0, \*)                  | Model Deploy Config                    | Model context length. If unspecified, it will be derived from the model config.                                                                                                                                                                                                 |
| `--max-input-token-len`                              |         | (0, `--max-seq-len`]     | Model Deploy Config                    | Maximum input token length. If unspecified, it will be derived from `min(--max-seq-len, --max-prefill-tokens)`.                                                                                                                                                                 |
| `--truncation`                                       |         |                          | Model Deploy Config                    | Truncate the input token length when it exceeds the minimum of `--max-input-token-len` and `--max-seq-len` - 1.                                                                                                                                                                 |
| `--cpu-mem-size`                                     | 0       | [0, \*)                  | Model Config                           | CPU swap space size in GiB. Works when specified `--max-preempt-count`.                                                                                                                                                                                                         |
| `--npu-memory-fraction` (env: `NPU_MEMORY_FRACTION`) | 0.9     | (0, 1]                   | Model Config                           | Fraction of NPU memory to be used for the model executor (0 to 1). For example: `0.5` means 50% memory utilization.                                                                                                                                                             |
| `--trust-remote-code`                                |         |                          | Model Config                           | Trust remote code (for model loading).                                                                                                                                                                                                                                          |
| `--models`                                           |         |                          | Model Config                           | Models configuration in JSON format, for certain specific configurations, like Expert Parallelism Implementation Method, Tensor Parallelism LM Header/Output Attention Split.                                                                                                   |
| `--async-scheduler-wait-time`                        | 120     | [1, 3600]                | Model Config                           | The wait time (in seconds) for the asynchronous scheduler to start.                                                                                                                                                                                                             |
| `--cache-block-size`                                 | 128     |                          | Schedule Config                        | KV cache block size. Must be a power of 2.                                                                                                                                                                                                                                      |
| `--max-prefill-batch-size`                           | 50      | [1, `--max-batch-size`]  | Schedule Config                        | Maximum number of requests batched during prefill stage. Must be less than `--max-batch-size`.                                                                                                                                                                                  |
| `--max-prefill-tokens  `                             |         | [1, `--max-seq-len`]     | Schedule Config                        | During each prefill, the total number of all input tokens in the current batch cannot exceed `--max-prefill-tokens`. Default is same as `--max-seq-len`.                                                                                                                        |
| `--prefill-time-ms-per-req`                          | 150     | [0, 1000]                | Schedule Config                        | Estimated prefill time per request (ms). Used to decide between prefill and decode stage.                                                                                                                                                                                       |
| `--prefill-policy-type`                              | 0       |                          | Schedule Config                        | Prefill stage strategy: <br> `0`: FCFS (First Come First Serve). <br> `1`: STATE (same as FCFS). <br> `2`: PRIORITY (priority queue). <br> `3`: MLFQ (Multi-Level Feedback Queue).                                                                                              |
| `--max-batch-size`                                   | 200     | [1, 5000]                | Schedule Config                        | Maximum number of requests batched during decode stage.                                                                                                                                                                                                                         |
| `--max-iter-times`                                   |         | [1, `--max-seq-len`]     | Schedule Config                        | Maximum iterations for decoding stage. Default is same as `--max-seq-len`.                                                                                                                                                                                                      |
| `--decode-time-ms-per-req`                           | 50      | [0, 1000]                | Schedule Config                        | Estimated decode time per request (ms). Used with `--prefill-time-ms-per-req` for batch selection.                                                                                                                                                                              |
| `--decode-policy-type`                               | 0       |                          | Schedule Config                        | Decode stage strategy: <br> `0`: FCFS <br> `1`: STATE (prioritize preempted or swapped requests) <br> `2`: PRIORITY <br> `3`: MLFQ                                                                                                                                              |
| `--max-preempt-count`                                | 0       | [0, `--max-batch-size`]  | Schedule Config                        | Maximum number of preempted requests allowed during decoding. Must be less than `--max-batch-size`.                                                                                                                                                                             |
| `--support-select-batch`                             |         |                          | Schedule Config                        | Enable batch selection. Determines execution priority based on `--prefill-time-ms-per-req` and `--decode-time-ms-per-req`. Use `--no-support-select-batch` to disable explicitly.                                                                                               |
| `--max-queue-delay-microseconds`                     | 5000    | [500, 1000000]           | Schedule Config                        | Maximum queue wait time in microseconds.                                                                                                                                                                                                                                        |
| `--max-first-token-wait-time`                        | 2500    | [0, 3600000]             | Schedule Config                        | Maximum milliseconds to wait for the first token generation.                                                                                                                                                                                                                    |
| `--override-generation-config`                       |         |                          |                                        | Overrides or sets generation config in JSON format. For example: `{"temperature": 0.5}`. This will merge into the `generation_config.json` of the model structure.                                                                                                              |
| `--enforce-eager`                                    |         |                          |                                        | Emit operators in eager mode.                                                                                                                                                                                                                                                   |
| `--no-metrics`                                       |         |                          |                                        | Disable exposing metrics at `/metrics` endpoint.                                                                                                                                                                                                                                |
| `--dtype`                                            | auto    |                          |                                        | Data type for model weights and activations. <br> `auto`: use the default data type of the model config. <br> `half`/`float16`: for FP16. <br> `bfloat16`: for BF16. <br> `float`/`float32`: for FP32.                                                                          |
| `--rope-scaling`                                     |         |                          | Extending Context Size                 | RoPE scaling configuration in JSON format. For example: `{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}`. This will merge into the `config.json` of the model structure.                                                                                 |
| `--rope-theta`                                       |         |                          | Extending Context Size                 | RoPE theta configuration. This will merge into the `config.json` of the model structure.                                                                                                                                                                                        |
| `--enable-split`                                     |         |                          | Split Fuse                             | Enable split fuse, something like chunked prefill. Use `--no-enable-split` to disable explicitly.                                                                                                                                                                               |
| `--policy-type`                                      | 0       |                          | Split Fuse                             | Strategy of split fuse. <br> `0`: FCFS, first come first serving. <br> `4`: SJF, shortest job first. <br> `5`: LJF, longest job first. <br> `6`: Skip-Join MLFQ, skip-Join multi-levels feedback queue. <br> `7`: SJF-MLFQ, shortest job first and multi-levels feedback queue. |
| `--split-chunk-tokens`                               | 512     | [1, 8192]                | Split Fuse                             | Tokens size to batch for split fuse. Multiples of 16.                                                                                                                                                                                                                           |
| `--split-start-batch-size`                           | 16      | [0, `--max-batch-size`]  | Split Fuse                             | Batch size to start splitting for split fuse.                                                                                                                                                                                                                                   |
| `--enable-memory-decoding`                           |         |                          | Speculative Decoding / Memory Decoding | Enable memory decoding speculation. Use `--no-enable-memory-decoding` to disable explicitly.                                                                                                                                                                                    |
| `--memory-decoding-length`                           | 16      | [1, 16]                  | Speculative Decoding / Memory Decoding | Length for memory decoding speculation.                                                                                                                                                                                                                                         |
| `--memory-decoding-dynamic-algo`                     |         |                          | Speculative Decoding / Memory Decoding | Enable dynamic algorithm for memory decoding speculation.                                                                                                                                                                                                                       |
| `--enable-lookahead`                                 |         |                          | Speculative Decoding / Lookahead       | Enable lookahead speculation. Use `--no-enable-lookahead` to disable explicitly.                                                                                                                                                                                                |
| `--lookahead-level`                                  | 4       | [3, 16]                  | Speculative Decoding / Lookahead       | Level for lookahead speculation.                                                                                                                                                                                                                                                |
| `--lookahead-window`                                 | 5       | [1, 16]                  | Speculative Decoding / Lookahead       | Window size for lookahead speculation.                                                                                                                                                                                                                                          |
| `--lookahead-guess-set-size`                         | 5       | [1, 16]                  | Speculative Decoding / Lookahead       | Guess set size for lookahead speculation.                                                                                                                                                                                                                                       |
| `--enable-multi-token-prediction`                    |         |                          | Multi-Token Prediction                 | Enable multi-token prediction. Use `--no-enable-multi-token-prediction` to disable explicitly.                                                                                                                                                                                  |
| `--multi-token-prediction-tokens`                    | 1       | (0, \*)                  | Multi-Token Prediction                 | Number of multi-token prediction tokens. This is only effective when `--enable-multi-token-prediction` is enabled.                                                                                                                                                              |
| `--enable-prefix-caching`                            |         |                          | Prefix Caching                         | Enable prefix caching. Use `--no-enable-prefix-caching` to disable explicitly.                                                                                                                                                                                                  |
| `--pipeline-parallel-size`, `-pp`                    | 1       | (0, \*)                  | Parallelism                            | Number of pipeline parallel groups.                                                                                                                                                                                                                                             |
| `--data-parallel-size`, `-dp`                        | -1      |                          | Parallelism                            | Number of data parallel groups for Attention layers. `-1` means disabling data parallelism, otherwise, must be a power of 2.                                                                                                                                                    |
| `--context-parallel-size`, `-cp`                     | -1      |                          | Parallelism                            | Number of context parallel groups for Attention layers. `-1` means disabling context parallelism, otherwise, must be a power of 2.                                                                                                                                              |
| `--tensor-parallel-size`, `-tp`                      | -1      |                          | Parallelism                            | Number of tensor parallel groups for Attention layers. `-1` means using world size as tensor parallel size, otherwise, must be a power of 2.                                                                                                                                    |
| `--sequence-parallel-size`, `-sp`                    | -1      | `--tensor-parallel-size` | Parallelism                            | Number of sequence parallel groups for MLP layers. `-1` means disabling sequence parallelism, otherwise, must be power of 2.                                                                                                                                                    |
| `--moe-expert-parallel-size`, `-moe-ep`              | -1      |                          | Parallelism                            | Number of expert parallel groups. `-1` means disabling MoE expert parallelism, otherwise, must be power of 2.                                                                                                                                                                   |
| `--moe-tensor-parallel-size`, `-moe-tp`              | -1      |                          | Parallelism                            | Number of tensor parallel groups for MoE MLP layers. `-1` means using world size as MoE tensor parallel size, otherwise, must be power of 2.                                                                                                                                    |
| `--enable-buffer-response`                           |         |                          | Buffer Response                        | Enable buffer response. Use `--no-enable-buffer-response` to disable explicitly.                                                                                                                                                                                                |
| `--prefill-expected-time-ms`                         |         |                          | Buffer Response                        | Expected latency (SLO) for Time to First Token (TTFT) in milliseconds.                                                                                                                                                                                                          |
| `--decode-expected-time-ms`                          |         |                          | Buffer Response                        | Expected latency (SLO) for Time Per Output Token (TPOT) in milliseconds.                                                                                                                                                                                                        |

!!! note

    GPUStack allows users to inject custom environment variables during model deployment, however, some variables may be conflicted with GPUStack managment.

    Hence, GPUStack will override/prevent those variables. Please compare the model instance logs' output with your expectations.

## Voxbox

[VoxBox](https://github.com/gpustack/vox-box) is an inference engine designed for deploying Text-to-Speech and Speech-to-Text models. It also provides an API that is fully compatible with the OpenAI audio API.

### Supported Models

| Model                           | Type           | Link                                                                                                                                                                | Supported Platforms |
| ------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| Faster-whisper-large-v3         | Speech-to-Text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v3)                 | AMD64,ARM64         |
| Faster-whisper-large-v2         | Speech-to-Text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v2), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v2)                 | AMD64,ARM64         |
| Faster-whisper-large-v1         | Speech-to-Text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v1), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v1)                 | AMD64,ARM64         |
| Faster-whisper-medium           | Speech-to-Text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-medium)                     | AMD64,ARM64         |
| Faster-whisper-medium.en        | Speech-to-Text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-medium.en)               | AMD64,ARM64         |
| Faster-whisper-small            | Speech-to-Text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-small)                       | AMD64,ARM64         |
| Faster-whisper-small.en         | Speech-to-Text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-small.en)                 | AMD64,ARM64         |
| Faster-distil-whisper-large-v3  | Speech-to-Text | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v3), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-large-v3)   | AMD64,ARM64         |
| Faster-distil-whisper-large-v2  | Speech-to-Text | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v2), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-large-v2)   | AMD64,ARM64         |
| Faster-distil-whisper-medium.en | Speech-to-Text | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-medium.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-medium.en) | AMD64,ARM64         |
| Faster-whisper-tiny             | Speech-to-Text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-tiny)                         | AMD64,ARM64         |
| Faster-whisper-tiny.en          | Speech-to-Text | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-tiny.en)                   | AMD64,ARM64         |
| CosyVoice-300M-Instruct         | Text-to-Speech | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M-Instruct), [ModelScope](https://modelscope.cn/models/gpustack/CosyVoice-300M-Instruct)                | AMD64               |
| CosyVoice-300M-SFT              | Text-to-Speech | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M-SFT), [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-SFT)                               | AMD64               |
| CosyVoice-300M                  | Text-to-Speech | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M), [ModelScope](https://modelscope.cn/models/gpustack/CosyVoice-300M)                                  | AMD64               |
| CosyVoice-300M-25Hz             | Text-to-Speech | [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-25Hz)                                                                                                  | AMD64               |
| CosyVoice2-0.5B                 | Text-to-Speech | [Hugging Face](https://huggingface.co/gpustack/CosyVoice2-0.5B), [ModelScope](https://modelscope.cn/models/iic/CosyVoice2-0.5B)                                     | AMD64               |
| Dia-1.6B                        | Text-to-Speech | [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B), [ModelScope](https://modelscope.cn/models/nari-labs/Dia-1.6B)                                            | AMD64               |

### Supported Features

#### Allow GPU/CPU Offloading

VoxBox supports deploying models to NVIDIA GPUs. If GPU resources are insufficient, it will automatically deploy the models to the CPU.
