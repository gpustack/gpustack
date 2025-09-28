# 隔离网络（Air-Gapped）安装

你可以在隔离网络环境中安装 GPUStack。隔离网络（Air-Gapped）环境指离线安装的场景。

在隔离网络环境中安装 GPUStack 可选择以下方法：

| OS      | 架构           | 支持的安装方式                                                        |
| ------- | -------------- |----------------------------------------------------------------|
| Linux   | AMD64<br>ARM64 | [Docker 安装](#docker-installation)（推荐）<br>[pip 安装](#pip-installation) |
| Windows | AMD64          | [桌面安装程序](../desktop-installer.md)（推荐）<br>[pip 安装](#pip-installation)           |

## 支持的后端

- [x] vLLM（计算能力 7.0 及以上，仅支持 Linux AMD64）
- [x] llama-box
- [x] vox-box

## 先决条件

- [端口要求](../installation-requirements.md#port-requirements)
- llama-box 后端的 CPU 要求：AMD64 支持 AVX，或 ARM64 支持 NEON

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

    Windows 用户需要根据上述说明手动确认是否支持。

- [NVIDIA 驱动](https://www.nvidia.com/en-us/drivers/)

检查是否已安装 NVIDIA 驱动：

```bash
nvidia-smi --format=csv,noheader --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu
```

并确保驱动支持 CUDA 12.4 或更高版本：

=== "Linux"

    ```bash
    nvidia-smi | grep "CUDA Version"
    ```

=== "Windows"

    ```powershell
    nvidia-smi | findstr "CUDA Version"
    ```

<a id="docker-installation"></a>

## Docker 安装

### 先决条件

- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

检查是否已安装 Docker 与 NVIDIA Container Toolkit：

```bash
docker info | grep Runtimes | grep nvidia
```

- [在 Docker 中禁用 systemd cgroup 管理](https://github.com/NVIDIA/nvidia-container-toolkit/issues/48)

!!! note

    当 systemd 用于管理容器的 cgroup，且触发了对任何引用 NVIDIA GPU 的 Unit 文件的重载（例如执行 systemctl daemon-reload）时，容器化的 GPU 负载可能会突然失去对 GPU 的访问。

    在 GPUStack 中，可能会在资源菜单中看不到 GPU，且在 GPUStack 容器内运行 nvidia-smi 可能会出现错误：Failed to initialize NVML: Unknown Error

    为防止[此问题](https://github.com/NVIDIA/nvidia-container-toolkit/issues/48)，需要在 Docker 中禁用 systemd 的 cgroup 管理。

在 /etc/docker/daemon.json 中设置参数 "exec-opts": ["native.cgroupdriver=cgroupfs"] 并重启 docker，例如：

```bash
vim /etc/docker/daemon.json
```

```json
{
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  },
  "exec-opts": ["native.cgroupdriver=cgroupfs"]
}
```

```bash
systemctl daemon-reload && systemctl restart docker
```

### 运行 GPUStack

使用 Docker 运行 GPUStack 时，只要镜像可用，即可在隔离网络环境下直接运行。请按以下步骤操作：

1. 在联网环境拉取 GPUStack 镜像：

```bash
docker pull gpustack/gpustack
```

如果你使用 Blackwell 系列或 GeForce RTX 50 系列，或者你的 NVIDIA 驱动支持 CUDA 12.8（可通过 nvidia-smi | grep "CUDA Version" 验证），强烈建议使用 latest-cuda12.8 镜像：

```bash
docker pull gpustack/gpustack:latest-cuda12.8
```

如果在线环境与隔离环境的 OS 或架构不同，拉取镜像时请指定隔离环境的 OS 与架构：

```bash
docker pull --platform linux/amd64 gpustack/gpustack
```

2. 将镜像发布到私有镜像仓库，或在隔离环境中直接加载该镜像。
3. 参考[Docker 安装](online-installation.md#docker-installation)指南，使用 Docker 运行 GPUStack。

<a id="pip-installation"></a>

## pip 安装

### 先决条件

- Python 3.10 ~ 3.12

检查 Python 版本：

```bash
python -V
```

- [NVIDIA CUDA 工具包 12](https://developer.nvidia.com/cuda-toolkit)

检查是否已安装 CUDA 且版本至少为 12.4：

```bash
nvcc -V
```

- [NVIDIA cuDNN 9](https://developer.nvidia.com/cudnn)（可选，音频模型需要）

检查是否已安装 cuDNN 9：

=== "Linux"

    ```bash
    ldconfig -p | grep libcudnn
    ```

=== "Windows"

    ```powershell
    Get-ChildItem -Path C:\ -Recurse -Filter "cudnn*.dll" -ErrorAction SilentlyContinue
    ```

### 安装 GPUStack

对于手动 pip 安装，你需要在联网环境准备所需的软件包和工具，然后传输到隔离网络环境。

搭建一个与隔离环境在操作系统、架构与 Python 版本完全一致的联网环境。

#### 第一步：下载所需包

=== "Linux"

    在联网环境执行以下命令：

    === "AMD64"

        ```bash
        # 可选的额外依赖：vllm、audio、all
        # vllm 仅适用于 Linux AMD64
        PACKAGE_SPEC="gpustack[all]"
        # 安装指定版本
        # PACKAGE_SPEC="gpustack[all]==0.6.0"
        ```

    === "ARM64"

        ```bash
        PACKAGE_SPEC="gpustack[audio]"
        # 安装指定版本
        # PACKAGE_SPEC="gpustack[audio]==0.6.0"
        ```

    如果不需要 vLLM 后端和音频模型支持，可设置：

    ```bash
    PACKAGE_SPEC="gpustack"
    ```

=== "Windows"

    在联网环境执行以下命令：

    ```powershell
    $PACKAGE_SPEC = "gpustack[audio]"
    # 安装指定版本
    # $PACKAGE_SPEC = "gpustack[audio]==0.6.0"
    ```

    如果不需要音频模型支持，可设置：

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

下载依赖工具并保存为归档：

```bash
gpustack download-tools --save-archive gpustack_offline_tools.tar.gz
```

如果你的联网环境与隔离环境不一致，请显式指定操作系统、架构与设备类型：

```bash
gpustack download-tools --save-archive gpustack_offline_tools.tar.gz --system linux --arch amd64 --device cuda
```

!!!note

    上述指令假设联网环境与隔离环境使用相同类型的 GPU。若 GPU 类型不同，请使用 --device 参数指定隔离环境的设备类型。更多信息参考 [download-tools](../../cli-reference/download-tools.md) 命令。

#### 第二步：传输包

将以下文件从联网环境传输到隔离环境。

- gpustack_offline_packages 目录
- gpustack_offline_tools.tar.gz 文件

#### 第三步：安装 GPUStack

在隔离网络环境中执行以下命令。

从已下载的包安装 GPUStack：

```bash
pip install --no-index --find-links=gpustack_offline_packages gpustack
```

加载并应用预下载的工具归档：

```bash
gpustack download-tools --load-archive gpustack_offline_tools.tar.gz
```

现在你可以按照 [pip 安装](online-installation.md#pip-installation) 指南运行 GPUStack。