# 隔离环境安装

你可以在隔离网络环境中安装 GPUStack。隔离环境指的是离线安装 GPUStack 的场景。

在隔离环境中安装 GPUStack 可用以下方法：

| 操作系统 | 架构            | 支持的安装方式                                                                                   |
| ------- | --------------- | ------------------------------------------------------------------------------------------------ |
| Linux   | AMD64<br>ARM64 | [Docker 安装](#docker-installation)（推荐）<br>[pip 安装](#pip-installation)                     |
| Windows | AMD64<br>ARM64 | [pip 安装](#pip-installation)                                                                     |

## 先决条件

- [端口要求](../installation-requirements.md#port-requirements)
- CPU（AMD64 支持 AVX 或 ARM64 支持 NEON）

=== "Linux"

    检查 CPU 是否受支持：

    === "AMD64"

        ```bash
        lscpu | grep avx
        ```

    === "ARM64"

        ```bash
        grep -E -i "neon|asimd" /proc/cpuinfo
        ```

=== "Windows"

    Windows 用户需要手动验证上述指令的支持情况。

<a id="docker-installation"></a>

## Docker 安装

### 先决条件

- [Docker](https://docs.docker.com/engine/install/)

### 运行 GPUStack

在 Docker 中运行 GPUStack 时，只要镜像可用，即可在隔离环境中开箱即用。请按以下步骤操作：

1. 在联网环境中拉取 GPUStack 的 Docker 镜像：

```bash
docker pull gpustack/gpustack:latest-cpu
```

如果联网环境与隔离环境在操作系统或架构上不同，请在拉取镜像时指定隔离环境的 OS 与架构：

```bash
docker pull --platform linux/amd64 gpustack/gpustack:latest-cpu
```

2. 将镜像推送到私有仓库，或直接在隔离环境中加载该镜像。
3. 参考[Docker 安装](online-installation.md#docker-installation)指南使用 Docker 运行 GPUStack。

<a id="pip-installation"></a>

## pip 安装

### 先决条件

- Python 3.10 ~ 3.12

检查 Python 版本：

```bash
python -V
```

### 安装 GPUStack

进行手动 pip 安装时，你需要在联网环境中准备所需的包与工具，然后将其传输到隔离环境。

请搭建与隔离环境完全一致的联网环境，包括操作系统（OS）、架构（architecture）和 Python 版本。

#### 步骤 1：下载所需包

在联网环境中执行以下命令：

=== "Linux"

    ```bash
    PACKAGE_SPEC="gpustack[audio]"
    # 如果需要安装特定版本
    # PACKAGE_SPEC="gpustack[audio]==0.6.0"
    ```

    如果不需要音频模型支持，请设置：

    ```bash
    PACKAGE_SPEC="gpustack"
    ```

=== "Windows"

    ```powershell
    $PACKAGE_SPEC = "gpustack[audio]"
    # 如果需要安装特定版本
    # $PACKAGE_SPEC = "gpustack[audio]==0.6.0"
    ```

    如果不需要音频模型支持，请设置：

    ```powershell
    $PACKAGE_SPEC = "gpustack"
    ```

下载所有所需包：

```bash
pip wheel $PACKAGE_SPEC -w gpustack_offline_packages
```

安装 GPUStack 以使用其 CLI：

```bash
pip install gpustack
```

下载依赖工具并保存为归档文件：

```bash
gpustack download-tools --save-archive gpustack_offline_tools.tar.gz
```

如果你的联网环境与隔离环境不同，请显式指定操作系统、架构和设备：

```bash
gpustack download-tools --save-archive gpustack_offline_tools.tar.gz --system linux --arch amd64 --device cpu
```

#### 步骤 2：传输包文件

将以下文件从联网环境传输到隔离环境：

- `gpustack_offline_packages` 目录
- `gpustack_offline_tools.tar.gz` 文件

#### 步骤 3：安装 GPUStack

在隔离环境中运行以下命令：

```bash
# 从已下载的包安装 GPUStack
pip install --no-index --find-links=gpustack_offline_packages gpustack

# 加载并应用预先下载的工具归档
gpustack download-tools --load-archive gpustack_offline_tools.tar.gz
```

现在你可以按照[pip 安装](online-installation.md#pip-installation)指南中的说明运行 GPUStack。