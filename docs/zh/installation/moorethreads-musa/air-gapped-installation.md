# 离线（Air-Gapped）安装

你可以在离线（物理隔离）环境中安装 GPUStack。离线环境指在无网络连接的情况下安装 GPUStack 的场景。

在离线环境中安装 GPUStack 可用的方法如下：

| 操作系统 | 架构  | 支持的方法                             |
| -------- | ----- | -------------------------------------- |
| Linux    | AMD64 | [Docker 安装](#docker-installation)    |

## 使用 Docker 安装 {#docker-installation}

### 先决条件

- [端口要求](../installation-requirements.md#port-requirements)
- 对 llama-box 后端的 CPU 要求：AMD64，支持 AVX

检查 CPU 是否受支持：

```bash
lscpu | grep avx
```

- [MTT S80/S3000/S4000 驱动](https://developer.mthreads.com/sdk/download/musa)

检查是否已安装驱动：

```bash
mthreads-gmi
```

- [Docker](https://docs.docker.com/engine/install/)
- [MT 容器工具包（MT Container Toolkits）](https://developer.mthreads.com/sdk/download/CloudNative)

检查是否已安装 MT 容器工具包并将其设置为默认运行时：

```bash
# cd /usr/bin/musa && sudo ./docker setup $PWD
docker info | grep Runtimes | grep mthreads
```

### 运行 GPUStack

在 Docker 中运行 GPUStack 时，只要所需的 Docker 镜像已准备好，即可在离线环境中直接使用。请按以下步骤操作：

1. 在可联网的环境中拉取 GPUStack 的 Docker 镜像：

```bash
docker pull gpustack/gpustack:latest-musa
```

如果联网环境与离线环境的操作系统或架构不同，请在拉取镜像时指定离线环境的 OS 和架构：

```bash
docker pull --platform linux/amd64 gpustack/gpustack:latest-musa
```

2. 将该 Docker 镜像发布到私有镜像仓库，或在离线环境中直接加载该镜像。
3. 参考[Docker 安装](online-installation.md#docker-installation)指南使用 Docker 运行 GPUStack。