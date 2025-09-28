# 物理隔离环境安装

您可以在物理隔离环境中安装 GPUStack。物理隔离环境指的是离线安装 GPUStack 的场景。

在物理隔离环境中安装 GPUStack 可用的方法如下：

| 操作系统 | 架构  | 支持的方法                         |
| ------- | ----- | ---------------------------------- |
| Linux   | ARM64 | [Docker 安装](#docker-installation) |

## Docker 安装 {#docker-installation}

### 前置条件

- [端口要求](../installation-requirements.md#port-requirements)
- 对 llama-box 后端的 CPU 支持：ARM64 且支持 NEON

检查 CPU 是否受支持：

```bash
grep -E -i "neon|asimd" /proc/cpuinfo
```

- [NPU 驱动与固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.2.RC1&driver=Ascend+HDK+25.2.0)（必须支持 CANN 8.2.RC1）

检查是否已安装 NPU 驱动：

```bash
npu-smi info
```

- [Docker](https://docs.docker.com/engine/install/)

### 运行 GPUStack

在 Docker 中运行 GPUStack 时，只要镜像可用，即可在物理隔离环境中直接工作。为此，请按以下步骤操作：

1. 在联网环境中拉取 GPUStack Docker 镜像：

=== "Ascend 910B"

    ```bash
    docker pull gpustack/gpustack:latest-npu
    ```

=== "Ascend 310P"

    ```bash
    docker pull gpustack/gpustack:latest-npu-310p
    ```

如果您的联网环境与物理隔离环境的操作系统或架构不同，请在拉取镜像时指定物理隔离环境的 OS 和架构：

=== "Ascend 910B"

    ```bash
    docker pull --platform linux/arm64 gpustack/gpustack:latest-npu
    ```

=== "Ascend 310P"

    ```bash
    docker pull --platform linux/arm64 gpustack/gpustack:latest-npu-310p
    ```

2. 将 Docker 镜像推送到私有镜像仓库，或在物理隔离环境中直接加载该镜像。
3. 参考[Docker 安装](online-installation.md#docker-installation)指南，使用 Docker 运行 GPUStack。