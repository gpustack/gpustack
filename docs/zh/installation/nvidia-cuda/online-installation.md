# 在线安装

## 支持的设备

- [x] NVIDIA GPU（计算能力 6.0 及以上，查看[您的 GPU 计算能力](https://developer.nvidia.com/cuda-gpus)）

## 支持的平台

| 操作系统 | 架构            | 支持的安装方式                                                                                                                                        |
| ------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux   | AMD64<br>ARM64  | [Docker 安装](#docker-installation)（推荐）<br>[pip 安装](#pip-installation)<br>[安装脚本](#install-scripts)（已弃用）                                                               |
| Windows | AMD64           | [桌面安装程序](../desktop-installer.md)（推荐）<br>[pip 安装](#pip-installation)<br>[安装脚本](#install-scripts)（已弃用）                                                  |

## 支持的后端

- [x] vLLM（计算能力 7.0 及以上，仅支持 AMD64 Linux）
- [x] llama-box
- [x] vox-box

## 前置条件

- [端口要求](../installation-requirements.md#port-requirements)
- llama-box 后端的 CPU 要求：支持 AVX 的 AMD64，或支持 NEON 的 ARM64

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

    Windows 用户需要按照以上说明手动验证是否受支持。

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

### 前置条件

- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

检查是否已安装 Docker 和 NVIDIA Container Toolkit：

```bash
docker info | grep Runtimes | grep nvidia
```

- [在 Docker 中禁用 systemd 的 cgroup 管理](https://github.com/NVIDIA/nvidia-container-toolkit/issues/48)

!!! note

    当使用 systemd 来管理容器的 cgroup，并且触发了对任何引用 NVIDIA GPU 的 Unit 文件的重新加载（例如执行 systemctl daemon-reload）时，容器化的 GPU 工作负载可能会突然失去对 GPU 的访问。

    在 GPUStack 中，可能会在 Resources 菜单中看不到 GPU，并且在 GPUStack 容器内运行 `nvidia-smi` 可能会出现错误：`Failed to initialize NVML: Unknown Error`

    为防止[此问题](https://github.com/NVIDIA/nvidia-container-toolkit/issues/48)，需要在 Docker 中禁用 systemd 的 cgroup 管理。

在 `/etc/docker/daemon.json` 文件中设置参数 `"exec-opts": ["native.cgroupdriver=cgroupfs"]` 并重启 docker，例如：

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

运行以下命令启动 GPUStack 服务器和内置的 worker（推荐使用 host 网络模式）：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

如果你使用的是 Blackwell 系列或 GeForce RTX 50 系列，或者你的 NVIDIA 驱动支持 CUDA 12.8（可通过 `nvidia-smi | grep "CUDA Version"` 验证），强烈建议使用 `latest-cuda12.8` 镜像：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-cuda12.8
```

如果需要更改默认的服务器端口 80，请使用 `--port` 参数：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack \
    --port 9090
```

如其他端口存在冲突，或需要自定义启动选项，请参考 [CLI 参考](../../cli-reference/start.md) 获取可用的标志与配置说明。

检查启动日志是否正常：

```bash
docker logs -f gpustack
```

如日志正常，在浏览器中打开 `http://your_host_ip` 访问 GPUStack UI。使用用户名 `admin` 和默认密码登录。可通过以下命令获取默认部署的密码：

```bash
docker exec -it gpustack cat /var/lib/gpustack/initial_admin_password
```

### （可选）添加 Worker

你可以向 GPUStack 添加更多 GPU 节点，构建 GPU 集群。需要在其他 GPU 节点上添加 worker，并通过 `--server-url` 与 `--token` 参数加入 GPUStack。

在 GPUStack 的服务器节点上运行以下命令获取用于添加 worker 的 token：

```bash
docker exec -it gpustack cat /var/lib/gpustack/token
```

在 worker 节点上运行以下命令以 worker 模式启动 GPUStack，并将其注册到 GPUStack 服务器。请将 URL 和 token 替换为你的实际值：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack \
    --server-url http://your_gpustack_url --token your_gpustack_token
```

!!! note

    1. 支持异构集群。无论是什么类型的设备，都可以通过指定 `--server-url` 与 `--token` 参数将其作为 worker 加入当前 GPUStack。

    2. 你可以在 docker run 命令末尾追加参数，为 `gpustack start` 命令设置额外的标志。配置细节请参考 [CLI 参考](../../cli-reference/start.md)。

    3. 可以使用 `--ipc=host` 标志或 `--shm-size` 标志，让容器访问主机的共享内存。vLLM 和 PyTorch 会在底层使用它在进程间共享数据，尤其用于张量并行推理。

### 构建自定义 Docker 镜像

例如，官方的 GPUStack NVIDIA CUDA 镜像基于 CUDA 12.4。如果你想使用其他 CUDA 版本，可以自行构建 Docker 镜像。

```dockerfile
# Example Dockerfile
ARG CUDA_VERSION=12.4.1

FROM nvidia/cuda:$CUDA_VERSION-cudnn-runtime-ubuntu22.04

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    tzdata \
    iproute2 \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/gpustack
RUN cd /workspace/gpustack && \
    make build

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
    # Install vllm dependencies for x86_64
    WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[all]"; \
    else  \
    WHEEL_PACKAGE="$(ls /workspace/gpustack/dist/*.whl)[audio]"; \
    fi && \
    pip install pipx && \
    pip install $WHEEL_PACKAGE && \
    pip cache purge && \
    rm -rf /workspace/gpustack

RUN gpustack download-tools

ENTRYPOINT [ "gpustack", "start" ]
```

运行以下命令构建 Docker 镜像：

```bash
docker build -t gpustack:cuda-12.8 --build-arg CUDA_VERSION=12.8.1 --file pack/Dockerfile .
```

<a id="pip-installation"></a>

## pip 安装

### 前置条件

- Python 3.10 ~ 3.12

检查 Python 版本：

```bash
python -V
```

- [NVIDIA CUDA Toolkit 12](https://developer.nvidia.com/cuda-toolkit)

检查是否已安装 CUDA 并确认版本至少为 12.4：

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

执行以下命令安装 GPUStack。

=== "Linux AMD64"

    ```bash
    # Extra dependencies options are "vllm", "audio" and "all"
    # "vllm" is only available for Linux AMD64
    pip install "gpustack[all]"
    ```

=== "Linux ARM64 或 Windows"

    ```bash
    pip install "gpustack[audio]"
    ```

如果不需要 vLLM 后端和音频模型支持，只需运行：

```bash
pip install gpustack
```

验证安装：

```bash
gpustack version
```

### 运行 GPUStack

运行以下命令启动 GPUStack 服务器和内置的 worker：

```bash
gpustack start
```

如启动日志正常，在浏览器中打开 `http://your_host_ip` 访问 GPUStack UI。使用用户名 `admin` 和默认密码登录。可通过以下命令获取默认部署的密码：

=== "Linux"

    ```bash
    cat /var/lib/gpustack/initial_admin_password
    ```

=== "Windows"

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\initial_admin_password" -Raw
    ```

默认情况下，GPUStack 使用 `/var/lib/gpustack` 作为数据目录，因此你需要具备 `sudo` 或相应权限。你也可以通过以下命令设置自定义数据目录：

```bash
gpustack start --data-dir mypath
```

更多可用的 CLI 标志参见 [CLI 参考](../../cli-reference/start.md)。

### （可选）添加 Worker

要向 GPUStack 集群添加 worker，需要指定服务器 URL 和认证 token。

在 GPUStack 的服务器节点上运行以下命令获取用于添加 worker 的 token：

=== "Linux"

    ```bash
    cat /var/lib/gpustack/token
    ```

=== "Windows"

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\token" -Raw
    ```

在 worker 节点上运行以下命令以 worker 模式启动 GPUStack，并将其注册到 GPUStack 服务器。请将 URL、token 和节点 IP 替换为你的实际值：

```bash
gpustack start --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_worker_host_ip
```

### 将 GPUStack 作为系统服务运行

推荐的方式是将 GPUStack 设置为开机自启服务。例如，使用 systemd：

在 `/etc/systemd/system/gpustack.service` 中创建服务文件：

```bash
tee /etc/systemd/system/gpustack.service > /dev/null <<EOF
[Unit]
Description=GPUStack Service
Wants=network-online.target
After=network-online.target

[Service]
EnvironmentFile=-/etc/default/%N
ExecStart=$(command -v gpustack) start
Restart=always
StandardOutput=append:/var/log/gpustack.log
StandardError=append:/var/log/gpustack.log

[Install]
WantedBy=multi-user.target
EOF
```

然后启动 GPUStack：

```bash
systemctl daemon-reload && systemctl enable gpustack --now
```

检查服务状态：

```bash
systemctl status gpustack
```

并确认 GPUStack 的启动日志正常：

```bash
tail -200f /var/log/gpustack.log
```

<a id="install-scripts"></a>

## 安装脚本（已弃用）

!!! warning

    自 0.7 版本起，安装脚本方式已弃用。我们建议在 Linux 上使用 Docker，在 macOS 或 Windows 上使用[桌面安装程序](https://gpustack.ai/)。

### 前置条件

- [NVIDIA CUDA Toolkit 12](https://developer.nvidia.com/cuda-toolkit)

检查是否已安装 CUDA 并确认版本至少为 12.4：

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

### 运行 GPUStack

GPUStack 提供脚本将其安装为端口默认为 80 的服务。

=== "Linux"

    ```bash
    curl -sfL https://get.gpustack.ai | sh -s -
    ```

    如需在运行脚本时配置额外的环境变量和启动标志，请参考[安装脚本](../installation-script.md)。

    安装完成后，确认 GPUStack 的启动日志正常：

    ```bash
    tail -200f /var/log/gpustack.log
    ```

    如果启动日志正常，可在浏览器中打开 `http://your_host_ip` 访问 GPUStack UI。使用用户名 `admin` 和默认密码登录。可通过以下命令获取默认部署的密码：

    ```bash
    cat /var/lib/gpustack/initial_admin_password
    ```

    如果指定了 `--data-dir` 参数设置数据目录，则 `initial_admin_password` 文件位于所指定的目录中。

=== "Windows"

    ```powershell
    Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content
    ```

    如需在运行脚本时配置额外的环境变量和启动标志，请参考[安装脚本](../installation-script.md)。

    安装完成后，确认 GPUStack 的启动日志正常：

    ```powershell
    Get-Content "$env:APPDATA\gpustack\log\gpustack.log" -Tail 200 -Wait
    ```

    如果启动日志正常，可在浏览器中打开 `http://your_host_ip` 访问 GPUStack UI。使用用户名 `admin` 和默认密码登录。可通过以下命令获取默认部署的密码：

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\initial_admin_password" -Raw
    ```
    如果指定了 `--data-dir` 参数设置数据目录，则 `initial_admin_password` 文件位于所指定的目录中。

### （可选）添加 Worker

=== "Linux"

    要向 GPUStack 集群添加 worker，需要在 worker 上安装 GPUStack 时指定服务器 URL 和认证 token。

    在 GPUStack 的服务器节点上运行以下命令获取用于添加 worker 的 token：

    ```bash
    cat /var/lib/gpustack/token
    ```

    如果指定了 `--data-dir` 参数设置数据目录，则 `token` 文件位于所指定的目录中。

    在 **worker 节点** 上运行以下命令安装 GPUStack 并以 worker 模式启动，**注册到 GPUStack 服务器**。请将 URL 和 token 替换为你的实际值：

    ```bash
    curl -sfL https://get.gpustack.ai | sh -s - --server-url http://your_gpustack_url --token your_gpustack_token
    ```

    安装完成后，确认 GPUStack 的启动日志正常：

    ```bash
    tail -200f /var/log/gpustack.log
    ```

=== "Windows"

    要向 GPUStack 集群添加 worker，需要在 worker 上安装 GPUStack 时指定服务器 URL 和认证 token。

    在 GPUStack 的服务器节点上运行以下命令获取用于添加 worker 的 token：

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\token" -Raw
    ```

    如果指定了 `--data-dir` 参数设置数据目录，则 `token` 文件位于所指定的目录中。

    在 **worker 节点** 上运行以下命令安装 GPUStack 并以 worker 模式启动，**注册到 GPUStack 服务器**。请将 URL 和 token 替换为你的实际值：

    ```powershell
    Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --server-url http://your_gpustack_url --token your_gpustack_token"
    ```

    安装完成后，确认 GPUStack 的启动日志正常：

    ```powershell
    Get-Content "$env:APPDATA\gpustack\log\gpustack.log" -Tail 200 -Wait
    ```