<br>

<p align="center">
    <img alt="GPUStack" src="https://raw.githubusercontent.com/gpustack/gpustack/main/docs/assets/gpustack-logo.png" width="300px"/>
</p>

<br>

<p align="center">
  <a href="https://github.com/gpustack/gpustack/blob/main/LICENSE" target="_blank">
    <img alt="License" src="https://img.shields.io/github/license/gpustack/gpustack?logo=github&logoColor=white&label=License&color=blue">
  </a>
  <a href="https://discord.gg/VXYJzuaqwD" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-GPUStack-blue?logo=discord&logoColor=white">
  </a>
  <a href="../assets/wechat-group-qrcode.jpg" target="_blank">
    <img alt="WeChat" src="https://img.shields.io/badge/微信群-GPUStack-blue?logo=wechat&logoColor=white">
  </a>
</p>

<p align="center">
  <script async defer src="https://buttons.github.io/buttons.js"></script>
  <a class="github-button" href="https://github.com/gpustack/gpustack" data-show-count="true" data-size="large" aria-label="Star">Star</a>
  <a class="github-button" href="https://github.com/gpustack/gpustack/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
  <a class="github-button" href="https://github.com/gpustack/gpustack/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>

GPUStack is an open-source GPU cluster manager for running AI models.

### Key Features

- **Broad GPU Compatibility:** Seamlessly supports GPUs from various vendors.
- **Extensive Model Support:** Supports a wide range of models including LLMs, VLMs, image models, audio models, embedding models, and rerank models.
- **Flexible Inference Backends:** Flexibly integrates with multiple inference backends including vLLM, Ascend MindIE, and vox-box.
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

## Supported Accelerators

GPUStack supports a variety of General-Purpose Accelerators, including:

- [x] NVIDIA GPU
- [x] AMD GPU
- [x] Ascend NPU
- [x] Hygon DCU (Experimental)
- [x] MThreads GPU (Experimental)
- [x] Iluvatar GPU (Experimental)
- [x] MetaX GPU (Experimental)
- [x] Cambricon MLU (Experimental)

## Supported Models

GPUStack uses [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [MindIE](https://www.hiascend.com/en/software/mindie) and [vox-box](https://github.com/gpustack/vox-box) as built-in inference backends, and it also supports any custom backend that can run in a container and expose a serving API. This allows GPUStack to work with a wide range of models.

Models can come from the following sources:

1. [Hugging Face](https://huggingface.co/)

2. [ModelScope](https://modelscope.cn/)

3. Local File Path

For information on which models are supported by each built-in inference backend, please refer to the supported models section in the [Built-in Inference Backends](user-guide/built-in-inference-backends.md) documentation.

## OpenAI-Compatible APIs

GPUStack serves OpenAI compatible APIs. For details, please refer to [OpenAI Compatible APIs](./user-guide/openai-compatible-apis.md)
