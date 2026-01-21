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
  <a href="../assets/docs-wechat-group-qrcode.jpg" target="_blank">
    <img alt="WeChat" src="https://img.shields.io/badge/微信群-GPUStack-blue?logo=wechat&logoColor=white">
  </a>
</p>

<p align="center">
  <script async defer src="https://buttons.github.io/buttons.js"></script>
  <a class="github-button" href="https://github.com/gpustack/gpustack" data-show-count="true" data-size="large" aria-label="Star">Star</a>
  <a class="github-button" href="https://github.com/gpustack/gpustack/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
  <a class="github-button" href="https://github.com/gpustack/gpustack/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>

GPUStack is an open-source GPU cluster manager designed for efficient AI model deployment. It lets you run models efficiently on your own GPU hardware by choosing the best inference engines, scheduling GPU resources, analyzing model architectures, and automatically configuring deployment parameters.

The following figure shows how GPUStack delivers improved inference throughput over the unoptimized vLLM baseline:

![a100-throughput-comparison](assets/a100-throughput-comparison.png)

For detailed benchmarking methods and results, visit our [Inference Performance Lab](https://docs.gpustack.ai/latest/performance-lab/overview/).

## Tested Inference Engines, GPUs, and Models

GPUStack uses a plug-in architecture that makes it easy to add new AI models, inference engines, and GPU hardware. We work closely with partners and the open-source community to test and optimize emerging models across different inference engines and GPUs. Below is the current list of supported inference engines, GPUs, and models, which will continue to expand over time.

**Tested Inference Engines:**

- vLLM
- SGLang
- TensorRT-LLM
- MindIE

**Tested GPUs:**

- NVIDIA A100
- NVIDIA H100/H200
- Ascend 910B

**Tuned Models:**

- Qwen3
- gpt-oss
- GLM-4.5-Air
- GLM-4.x
- DeepSeek-R1
- DeepSeek-V3.2

## Architecture

GPUStack enables development teams, IT organizations, and service providers to deliver Model-as-a-Service at scale. It supports industry-standard APIs for LLM, voice, image, and video models. The platform includes built-in user authentication and access control, real-time monitoring of GPU performance and utilization, and detailed metering of token usage and API request rates.

The figure below illustrates how a single GPUStack server can manage multiple GPU clusters across both on-premises and cloud environments. The GPUStack scheduler allocates GPUs to maximize resource utilization and selects the appropriate inference engines for optimal performance. Administrators also gain full visibility into system health and metrics through integrated Grafana and Prometheus dashboards.

![gpustack-v2-architecture](assets/gpustack-v2-architecture.png)

GPUStack provides a powerful framework for deploying AI models. Its core features include:

- **Multi-Cluster GPU Management.** Manages GPU clusters across multiple environments. This includes on-premises servers, Kubernetes clusters, and cloud providers.
- **Pluggable Inference Engines.** Automatically configures high-performance inference engines such as vLLM, SGLang, and TensorRT-LLM. You can also add custom inference engines as needed.
- **Performance-Optimized Configurations.** Offers pre-tuned modes for low latency or high throughput. GPUStack supports extended KV cache systems like LMCache and HiCache to reduce TTFT. It also includes built-in support for speculative decoding methods such as EAGLE3, MTP, and N-grams.
- **Enterprise-Grade Operations.** Offers support for automated failure recovery, load balancing, monitoring, authentication, and access control.
