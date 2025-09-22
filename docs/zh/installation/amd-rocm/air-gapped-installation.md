# 物理隔离（离线）环境安装

你可以在物理隔离（离线）环境中安装 GPUStack。物理隔离环境指在离线状态下安装 GPUStack 的环境。

以下方法可用于在物理隔离环境中安装 GPUStack：

| 操作系统 | 架构 | 支持的安装方式                               |
| ------- | ---- | -------------------------------------------- |
| Linux   | AMD64| [Docker 安装](#docker-installation)           |
| Windows | AMD64| [桌面安装程序](../desktop-installer.md)       |

## 支持的设备

| 设备                                                 | 支持的后端        |
| ---------------------------------------------------- | ----------------- |
| gfx1100: AMD Radeon RX 7900 XTX/7900 XT/7900 GRE     | llama-box, vLLM   |
| gfx1101: AMD Radeon RX 7800 XT/7700 XT               | llama-box, vLLM   |
| gfx1102: AMD Radeon RX 7600 XT/7600                  | llama-box, vLLM   |
| gfx942: AMD Instinct MI300X/MI300A                   | llama-box, vLLM   |
| gfx90a: AMD Instinct MI250X/MI250/MI210              | llama-box, vLLM   |
| gfx1030: AMD Radeon RX 6950 XT/6900 XT/6800 XT/6800  | llama-box         |
| gfx1031: AMD Radeon RX 6750 XT/6700 XT/6700          | llama-box         |
| gfx1032: AMD Radeon RX 6650 XT/6600 XT/6600          | llama-box         |
| gfx908: AMD Instinct MI100                           | llama-box         |
| gfx906: AMD Instinct MI50                            | llama-box         |

## 先决条件

- [端口要求](../installation-requirements.md#port-requirements)
- llama-box 后端的 CPU 要求：AMD64 且支持 AVX

检查 CPU 是否受支持：

```bash
lscpu | grep avx
```

## Docker 安装
<a id="docker-installation"></a>

### 前置条件

- [Docker](https://docs.docker.com/engine/install/)
- [ROCm 6.2.4](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.4/install/install-overview.html)

请选择与你系统匹配的安装方式。这里提供 Linux（Ubuntu 22.04）的步骤。其他系统请参考 ROCm 文档。

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

确认当前用户已加入 render 与 video 组：

```bash
groups
```

检查是否已安装 amdgpu 内核驱动：

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

使用 Docker 运行 GPUStack 时，只要相应的 Docker 镜像可用，即可在物理隔离环境中开箱即用。请按以下步骤操作：

1. 在联网环境中拉取 GPUStack 的 Docker 镜像：

```bash
docker pull gpustack/gpustack:latest-rocm
```

如果联网环境与物理隔离环境在操作系统或架构上不一致，请在拉取镜像时指定物理隔离环境的 OS 和架构：

```bash
docker pull --platform linux/amd64 gpustack/gpustack:latest-rocm
```

2. 将 Docker 镜像发布到私有镜像仓库，或在物理隔离环境中直接加载该镜像。
3. 参考[Docker 安装](online-installation.md#docker-installation)指南，使用 Docker 运行 GPUStack。