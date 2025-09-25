# 隔离环境安装

你可以在隔离（离线）环境中安装 GPUStack。隔离环境指的是在离线状态下安装 GPUStack 的场景。

以下是在隔离环境中安装 GPUStack 的可用方法：

| 操作系统 | 架构  | 支持的安装方法                         |
| -------- | ----- | -------------------------------------- |
| Linux    | AMD64 | [Docker 安装](#docker-installation) |

## Docker 安装 {#docker-installation}

### 先决条件

- [端口要求](../installation-requirements.md#port-requirements)
- 对 llama-box 后端的 CPU 支持：AMD64 且支持 AVX

检查 CPU 是否支持：

```bash
lscpu | grep avx
```

- [Docker](https://docs.docker.com/engine/install/)
- [DCU 驱动 rock-6.3](https://developer.sourcefind.cn/tool/)

检查驱动是否已安装：

```bash
lsmod | grep dcu
```

### 运行 GPUStack

在 Docker 中运行 GPUStack 时，只要所需的 Docker 镜像可用，即可在隔离环境中开箱即用。请按照以下步骤操作：

1. 在联网环境中拉取 GPUStack 的 Docker 镜像：

```bash
docker pull gpustack/gpustack:latest-dcu
```

如果你的联网环境与隔离环境在操作系统或架构上不同，请在拉取镜像时指定隔离环境的 OS 和架构：

```bash
docker pull --platform linux/amd64 gpustack/gpustack:latest-dcu
```

2. 将该 Docker 镜像发布到私有镜像仓库，或直接在隔离环境中加载该镜像。
3. 参考 [Docker 安装](online-installation.md#docker-installation) 指南，通过 Docker 运行 GPUStack。