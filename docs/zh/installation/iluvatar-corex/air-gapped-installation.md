# 隔离环境安装

您可以在隔离环境中安装 GPUStack。所谓隔离环境，是指在离线状态下安装 GPUStack 的环境。

在隔离环境中安装 GPUStack 可使用以下方法：

| 操作系统 | 架构  | 支持的安装方法                           |
| ------- | ----- | ---------------------------------------- |
| Linux   | AMD64 | [Docker 安装](#docker-installation)      |

## 支持的后端

- [x] vLLM

## Docker 安装 {#docker-installation}

### 先决条件

- [适用于 MR-V100、MR-V50、BI-V100、BI-V150 的驱动程序](https://support.iluvatar.com/#/ProductLine?id=2)

检查是否已安装驱动：

```bash
ixsmi
```

- [Docker](https://support.iluvatar.com/#/ProductLine?id=2)
- [Corex 容器工具包](https://support.iluvatar.com/#/ProductLine?id=2)

### 运行 GPUStack

使用 Docker 运行 GPUStack 时，只要镜像可用，即可在隔离环境中开箱即用。请按以下步骤操作：

1. 在联网环境中拉取 GPUStack 的 Docker 镜像：

```bash
docker pull gpustack/gpustack:latest-corex
```

如果联网环境与隔离环境在操作系统或架构上不同，请在拉取镜像时指定隔离环境的操作系统与架构：

```bash
docker pull --platform linux/amd64 gpustack/gpustack:latest-corex
```

2. 将 Docker 镜像发布到私有镜像仓库，或在隔离环境中直接加载该镜像。
3. 参考[Docker 安装](online-installation.md#docker-installation)指南，使用 Docker 运行 GPUStack。