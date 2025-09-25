# 固定推理后端版本

生成式 AI 领域的推理引擎正在快速演进，以提升性能并解锁新能力。这种持续演进带来了令人振奋的机遇，但也在维护模型兼容性和部署稳定性方面提出了挑战。

GPUStack 允许你将[推理后端](inference-backends.md)版本固定到特定发布版，在保持及时获取最新进展与确保可靠运行环境之间取得平衡。该特性在以下场景尤为有用：

- 利用最新后端特性，而无需等待 GPUStack 更新。
- 锁定特定后端版本以维持现有模型的兼容性。
- 为具有不同需求的模型分配不同的后端版本。

通过固定后端版本，你可以完全掌控推理环境，使部署既灵活又可预测。

## 固定后端版本的自动安装

为简化部署，GPUStack 在可行时支持自动安装固定的后端版本。具体流程取决于后端类型：

1. **预编译二进制**  
   对于 `llama-box` 等后端，GPUStack 会使用与 GPUStack 引导相同的机制下载指定版本。

!!! tip

    你可以使用 `--tools-download-base-url` [配置项](../cli-reference/start.md)自定义下载源。

2. **基于 Python 的后端**  
   对于 `vLLM` 和 `vox-box` 等后端，GPUStack 使用 `pipx` 在隔离的 Python 环境中安装指定版本。

!!! tip

    - 请确保工作节点已安装 `pipx`。
    - 如果 `pipx` 不在系统 PATH 中，可通过 `--pipx-path` [配置项](../cli-reference/start.md)指定其位置。

此自动化流程可减少人工干预，让你专注于模型的部署与使用。

- **指定后端依赖项**

    对于基于 Python 的后端，可通过环境变量 `GPUSTACK_BACKEND_DEPS` 配置自定义依赖，位置在：
    `Deployment > Advanced > Environment Variables`。当你的模型需要特定版本时，这将覆盖默认包。
    例如：
    `GPUSTACK_BACKEND_DEPS:transformers==4.53.3,torch>=2.0.0`

## 固定后端版本的手动安装

当自动安装不可行或你更倾向于手动管理时，GPUStack 提供了简便的方法来手动安装指定版本的推理后端。步骤如下：

1. **准备可执行文件**  
   将后端可执行文件安装或链接到 GPUStack 的 bin 目录下。默认位置为：

   - **Linux/macOS：** `/var/lib/gpustack/bin`
   - **Windows：** `$env:AppData\gpustack\bin`

!!! tip

    你可以通过 `--bin-dir` [配置项](../cli-reference/start.md)自定义 bin 目录。

2. **命名可执行文件**  
   确保可执行文件按以下格式命名：

   - **Linux/macOS：** `<backend>_<version>`
   - **Windows：** `<backend>_<version>.exe`

例如，vLLM 的 v0.7.3 版本可执行文件在 Linux 上应命名为 `vllm_v0.7.3`。

通过以上步骤，你可以完全掌控后端的安装流程，确保部署时使用正确的版本。