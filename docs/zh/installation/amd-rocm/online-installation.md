# 在线安装

## 支持的设备

- [x] AMD GPUs

## 支持的平台

| 操作系统 | 版本                         | 架构  | 支持的安装方式                                                                                                              |
| ------- | ---------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------- |
| Linux   | Ubuntu 22.04<br>Ubuntu 24.04 | AMD64 | [Docker 安装](#docker-installation) (Recommended)<br>[安装脚本](#installation-script-deprecated) (Deprecated)              |
| Windows | 10<br>11                     | AMD64 | [桌面安装程序](../desktop-installer.md) (Recommended)<br>[安装脚本](#installation-script-deprecated) (Deprecated)           |

## 先决条件

- [端口要求](../installation-requirements.md#port-requirements)
- llama-box 后端的 CPU 要求：AMD64 且支持 AVX

=== "Linux"

    检查 CPU 是否受支持：

    ```bash
    lscpu | grep avx
    ```

=== "Windows"

    Windows 用户需要手动验证是否符合上述要求。

<a id="docker-installation"></a>

## Docker 安装

### 支持的设备

| 设备                                                | 支持的后端      |
| --------------------------------------------------- | --------------- |
| gfx1100: AMD Radeon RX 7900 XTX/7900 XT/7900 GRE    | llama-box, vLLM |
| gfx1101: AMD Radeon RX 7800 XT/7700 XT              | llama-box, vLLM |
| gfx1102: AMD Radeon RX 7600 XT/7600                 | llama-box, vLLM |
| gfx942: AMD Instinct MI300X/MI300A                  | llama-box, vLLM |
| gfx90a: AMD Instinct MI250X/MI250/MI210             | llama-box, vLLM |
| gfx1030: AMD Radeon RX 6950 XT/6900 XT/6800 XT/6800 | llama-box       |
| gfx1031: AMD Radeon RX 6750 XT/6700 XT/6700         | llama-box       |
| gfx1032: AMD Radeon RX 6650 XT/6600 XT/6600         | llama-box       |
| gfx908: AMD Instinct MI100                          | llama-box       |
| gfx906: AMD Instinct MI50                           | llama-box       |

### 前置条件

- [Docker](https://docs.docker.com/engine/install/)
- [ROCm 6.2.4](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.4/install/install-overview.html)

请选择适合您系统的安装方式。此处提供 Linux（Ubuntu 22.04）的步骤。其他系统请参考 ROCm 文档。

1. 安装 ROCm：

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/jammy/amdgpu-install_6.2.60204-1_all.deb
sudo apt install ./amdgpu-install_6.2.60204-1_all.deb
amdgpu-install -y --usecase=graphics,rocm
sudo reboot
```

2. 设置用户组权限：

```bash
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

3. 验证安装：

验证当前用户是否已加入 render 和 video 组：

```bash
groups
```

检查是否安装了 amdgpu 内核驱动：

```bash
dkms status
```

检查 GPU 是否被列为 agent：

```bash
rocminfo
```

检查 `rocm-smi`：

```bash
rocm-smi -i --showmeminfo vram --showpower --showserial --showuse --showtemp --showproductname --showuniqueid --json
```

### 运行 GPUStack

运行以下命令启动 GPUStack 服务器并启动内置 worker（建议使用 host 网络模式）：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --device=/dev/kfd \
    --device=/dev/dri \
    --network=host \
    --ipc=host \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-rocm
```

如果需要更改默认服务器端口 80，请使用 `--port` 参数：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --device=/dev/kfd \
    --device=/dev/dri \
    --network=host \
    --ipc=host \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-rocm \
    --port 9090
```

如果其他端口发生冲突，或希望自定义启动选项，请参阅 [CLI 参考](../../cli-reference/start.md) 了解可用的标志及配置说明。

检查启动日志是否正常：

```bash
docker logs -f gpustack
```

如果日志正常，在浏览器打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。您可以运行以下命令获取默认安装生成的密码：

```bash
docker exec -it gpustack cat /var/lib/gpustack/initial_admin_password
```

### （可选）添加 Worker

您可以向 GPUStack 添加更多 GPU 节点以形成 GPU 集群。需要在其他 GPU 节点上添加 worker，并指定 `--server-url` 和 `--token` 参数以加入 GPUStack。

要获取用于添加 worker 的 token，请在 GPUStack“服务器节点”上运行以下命令：

```bash
docker exec -it gpustack cat /var/lib/gpustack/token
```

要以 worker 方式启动 GPUStack，并将其“注册到 GPUStack 服务器”，请在“worker 节点”上运行以下命令。请将 URL 和 token 替换为您的实际值：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --device=/dev/kfd \
    --device=/dev/dri \
    --network=host \
    --ipc=host \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-rocm \
    --server-url http://your_gpustack_url --token your_gpustack_token
```

!!! note

    1. 支持“异构集群”。无论是哪种类型的设备，只需指定 `--server-url` 和 `--token` 参数即可将其作为 worker 加入当前 GPUStack。

    2. 可以在 docker run 命令后追加 gpustack start 的其他标志。有关配置详情，请参阅 [CLI 参考](../../cli-reference/start.md)。

    3. 可以使用 `--ipc=host` 或 `--shm-size` 使容器访问宿主机的共享内存。共享内存在底层被 vLLM 和 pyTorch 用于进程间共享数据，特别是在张量并行推理时。

<a id="installation-script-deprecated"></a>

## 安装脚本（已弃用）

!!! warning

    自 0.7 版本起，安装脚本方式已弃用。我们建议在 Linux 上使用 Docker，在 macOS 或 Windows 上使用[桌面安装程序](../desktop-installer.md)。

#### 支持的设备

=== "Linux"

    | 设备                                                | 支持的后端 |
    | --------------------------------------------------- | ---------- |
    | gfx1100: AMD Radeon RX 7900 XTX/7900 XT/7900 GRE    | llama-box  |
    | gfx1101: AMD Radeon RX 7800 XT/7700 XT              | llama-box  |
    | gfx1102: AMD Radeon RX 7600 XT/7600                 | llama-box  |
    | gfx1030: AMD Radeon RX 6950 XT/6900 XT/6800 XT/6800 | llama-box  |
    | gfx1031: AMD Radeon RX 6750 XT/6700 XT/6700         | llama-box  |
    | gfx1032: AMD Radeon RX 6650 XT/6600 XT/6600         | llama-box  |
    | gfx942: AMD Instinct MI300X/MI300A                  | llama-box  |
    | gfx90a: AMD Instinct MI250X/MI250/MI210             | llama-box  |
    | gfx908: AMD Instinct MI100                          | llama-box  |
    | gfx906: AMD Instinct MI50                           | llama-box  |

    点击此处查看更多详细信息 [here](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.4/reference/system-requirements.html)。

=== "Windows"

    | 设备                                                | 支持的后端 |
    | --------------------------------------------------- | ---------- |
    | gfx1100: AMD Radeon RX 7900 XTX/7900 XT             | llama-box  |
    | gfx1101: AMD Radeon RX 7800 XT/7700 XT              | llama-box  |
    | gfx1102: AMD Radeon RX 7600 XT/7600                 | llama-box  |
    | gfx1030: AMD Radeon RX 6950 XT/6900 XT/6800 XT/6800 | llama-box  |
    | gfx1031: AMD Radeon RX 6750 XT/6700 XT/6700         | llama-box  |
    | gfx1032: AMD Radeon RX 6650 XT/6600 XT/6600         | llama-box  |

    点击此处查看更多详细信息 [here](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-6.2.4/reference/system-requirements.html)。

#### 前置条件

=== "Linux"

    - [ROCm 6.2.4](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.4/install/install-overview.html)

    1. 安装 ROCm 6.2.4：

    ```bash
    sudo apt update
    wget https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/jammy/amdgpu-install_6.2.60204-1_all.deb
    sudo apt install ./amdgpu-install_6.2.60204-1_all.deb
    amdgpu-install -y --usecase=graphics,rocm
    sudo reboot
    ```

    2. 设置用户组权限：

    ```bash
    sudo usermod -a -G render,video $LOGNAME
    sudo reboot
    ```

    3. 验证安装：

    验证当前用户是否已加入 render 和 video 组：

    ```bash
    groups
    ```

    检查是否安装了 amdgpu 内核驱动：

    ```bash
    dkms status
    ```

    检查 GPU 是否被列为 agent：

    ```bash
    rocminfo
    ```

    检查 `rocm-smi`：

    ```bash
    rocm-smi -i --showmeminfo vram --showpower --showserial --showuse --showtemp --showproductname --showuniqueid --json
    ```

=== "Windows"

    - [HIP SDK 6.2.4](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-6.2.4/how-to/install.html)

    1. 在 PowerShell 中输入以下命令，确认获取的信息与[支持的 SKU](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-6.2.4/reference/system-requirements.html#supported-skus-win)中列出的内容一致：

    ```powershell
    Get-ComputerInfo | Format-Table CsSystemType,OSName,OSDisplayVersion
    ```

    2. 从 [HIP-SDK 下载页面](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)下载安装程序。

    3. 运行安装程序。

### 运行 GPUStack

GPUStack 提供脚本将其安装为服务，默认端口为 80。

=== "Linux"

    ```bash
    curl -sfL https://get.gpustack.ai | sh -s -
    ```

    如需在运行脚本时配置额外的环境变量和启动标志，请参阅[安装脚本](../installation-script.md)。

    安装完成后，确认 GPUStack 启动日志正常：

    ```bash
    tail -200f /var/log/gpustack.log
    ```

    如果启动日志正常，在浏览器打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。您可以运行以下命令获取默认安装生成的密码：

    ```bash
    cat /var/lib/gpustack/initial_admin_password
    ```

    如果指定了 `--data-dir` 参数设置数据目录，`initial_admin_password` 文件将位于该目录中。

=== "Windows"

    ```powershell
    Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content
    ```

    如需在运行脚本时配置额外的环境变量和启动标志，请参阅[安装脚本](../installation-script.md)。

    安装完成后，确认 GPUStack 启动日志正常：

    ```powershell
    Get-Content "$env:APPDATA\gpustack\log\gpustack.log" -Tail 200 -Wait
    ```

    如果启动日志正常，在浏览器打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。您可以运行以下命令获取默认安装生成的密码：

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\initial_admin_password" -Raw
    ```
    如果指定了 `--data-dir` 参数设置数据目录，`initial_admin_password` 文件将位于该目录中。

### （可选）添加 Worker

=== "Linux"

    要向 GPUStack 集群添加 worker，需要在 worker 安装 GPUStack 时指定服务器 URL 和认证 token。

    要获取用于添加 worker 的 token，请在 GPUStack“服务器节点”上运行以下命令：

    ```bash
    cat /var/lib/gpustack/token
    ```

    如果指定了 `--data-dir` 参数设置数据目录，`token` 文件将位于该目录中。

    要安装 GPUStack 并以 worker 方式启动，同时“注册到 GPUStack 服务器”，请在“worker 节点”上运行以下命令。请将 URL 和 token 替换为您的实际值：

    ```bash
    curl -sfL https://get.gpustack.ai | sh -s - --server-url http://your_gpustack_url --token your_gpustack_token
    ```

    安装完成后，确认 GPUStack 启动日志正常：

    ```bash
    tail -200f /var/log/gpustack.log
    ```

=== "Windows"

    要向 GPUStack 集群添加 worker，需要在 worker 安装 GPUStack 时指定服务器 URL 和认证 token。

    要获取用于添加 worker 的 token，请在 GPUStack“服务器节点”上运行以下命令：

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\token" -Raw
    ```

    如果指定了 `--data-dir` 参数设置数据目录，`token` 文件将位于该目录中。

    要安装 GPUStack 并以 worker 方式启动，同时“注册到 GPUStack 服务器”，请在“worker 节点”上运行以下命令。请将 URL 和 token 替换为您的实际值：

    ```powershell
    Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --server-url http://your_gpustack_url --token your_gpustack_token"
    ```

    安装完成后，确认 GPUStack 启动日志正常：

    ```powershell
    Get-Content "$env:APPDATA\gpustack\log\gpustack.log" -Tail 200 -Wait
    ```