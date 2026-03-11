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
  <a href="https://github.com/gpustack/gpustack/blob/main/docs/assets/wechat-group-qrcode.jpg" target="_blank">
    <img alt="WeChat" src="https://img.shields.io/badge/微信群-GPUStack-blue?logo=wechat&logoColor=white">
  </a>
</p>

<p align="center">
  <script async defer src="https://buttons.github.io/buttons.js"></script>
  <a class="github-button" href="https://github.com/gpustack/gpustack" data-show-count="true" data-size="large" aria-label="Star">Star</a>
  <a class="github-button" href="https://github.com/gpustack/gpustack/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
  <a class="github-button" href="https://github.com/gpustack/gpustack/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>

## Overview

GPUStack is an open-source GPU cluster manager designed for efficient AI model deployment. It configures and orchestrates inference engines — vLLM, SGLang, TensorRT-LLM, or your own — to optimize performance across GPU clusters. Its core features include:

- **Multi-Cluster GPU Management.** Manages GPU clusters across multiple environments. This includes on-premises servers, Kubernetes clusters, and cloud providers.
- **Pluggable Inference Engines.** Automatically configures high-performance inference engines such as vLLM, SGLang, and TensorRT-LLM. You can also add custom inference engines as needed.
- **Day 0 Model Support.** GPUStack's pluggable engine architecture enables you to deploy new models on the day they are released.
- **Performance-Optimized Configurations.** Offers pre-tuned modes for low latency or high throughput. GPUStack supports extended KV cache systems like LMCache and HiCache to reduce TTFT. It also includes built-in support for speculative decoding methods such as EAGLE3, MTP, and N-grams.
- **Enterprise-Grade Operations.** Offers support for automated failure recovery, load balancing, monitoring, authentication, and access control.

## Architecture

GPUStack enables development teams, IT organizations, and service providers to deliver Model-as-a-Service at scale. It supports industry-standard APIs for LLM, voice, image, and video models. The platform includes built-in user authentication and access control, real-time monitoring of GPU performance and utilization, and detailed metering of token usage and API request rates.

The figure below illustrates how a single GPUStack server can manage multiple GPU clusters across both on-premises and cloud environments. The GPUStack scheduler allocates GPUs to maximize resource utilization and selects the appropriate inference engines for optimal performance. Administrators also gain full visibility into system health and metrics through integrated Grafana and Prometheus dashboards.

![gpustack-v2-architecture](assets/gpustack-v2-architecture.png)

## Optimized Inference Performance

GPUStack's automated engine selection and parameter optimization deliver strong inference performance out of the box. The following figure shows throughput improvements over default vLLM configurations:

![a100-throughput-comparison](assets/a100-throughput-comparison.png)

For detailed benchmarking methods and results, visit our [Inference Performance Lab](https://docs.gpustack.ai/latest/performance-lab/overview/).

## Supported Accelerators

GPUStack supports a wide range of accelerators for AI inference:

- **NVIDIA GPU**
- **AMD GPU**
- **Ascend NPU**
- **Hygon DCU**
- **MThreads GPU**
- **Iluvatar GPU**
- **MetaX GPU**
- **Cambricon MLU**
- **T-Head PPU**

For detailed requirements and setup instructions, see the [Installation Requirements](installation/requirements.md) documentation.
