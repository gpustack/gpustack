# 安装要求 { #installation-requirements }

本页介绍运行 GPUStack 的节点所需的软件和网络条件。

## 操作系统要求 { #operating-system-requirements }

GPUStack 支持 **AMD64** 和 **ARM64** 架构上的大多数现代 Linux 发行版。

!!! note

    - 不支持直接通过 PyPI 安装 GPUStack。为获得最佳兼容性，请使用提供的 Docker 镜像。
    - 必须安装 Network Time Protocol（NTP）软件包，确保节点之间的状态同步保持一致。

## 数据库要求 { #database-requirements }

GPUStack Server 将状态存储在 SQL 数据库中。GPUStack Server 容器镜像包含内嵌 PostgreSQL 数据库，无需任何配置即可使用。

也可以通过 `--database-url` 参数或 `GPUSTACK_DATABASE_URL` 环境变量使用外部数据库。兼容以下数据库：

| 数据库 | 已验证版本 | 数据库 URL 格式 |
| ---------- | --------------------- | ------------------------------------------------ |
| PostgreSQL | `13.0+` | `postgresql://username:password@host:port/dbname` |
| MySQL | `8.0.36+` | `mysql://username:password@host:port/dbname` |
| openGauss | `v6.0.5`, `v7.0.0-RC3` | `postgresql://username:password@host:port/dbname` |
| OceanBase | `CE-v4.3.5` | `mysql://username:password@host:port/dbname` |

!!! note

    openGauss 兼容 PostgreSQL，OceanBase 兼容 MySQL，因此分别使用 `postgresql://` 和 `mysql://` URL 格式。

配置示例见[使用外部数据库](installation.md#using-an-external-database)。

## 加速器运行时要求 { #accelerator-runtime-requirements }

GPUStack 支持多种通用加速器作为推理后端，包括：

- [x] NVIDIA GPU
- [x] AMD GPU
- [x] 昇腾 NPU
- [x] 海光 DCU
- [x] 沐曦 GPU
- [x] 摩尔线程 GPU（实验性）
- [x] 天数智芯 GPU（实验性）
- [x] 寒武纪 MLU（实验性）
- [x] 平头哥 PPU（实验性）

运行 GPUStack 前，请确保已安装所有所需驱动和工具包。

### NVIDIA GPU { #nvidia-gpu }

#### 要求 { #requirements }

- 支持 NVIDIA CUDA 12.8 或更高版本的 [NVIDIA GPU Driver](https://www.nvidia.com/en-us/drivers/)。
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit)

!!! note "Kubernetes GPU 节点"

    如果 GPU 由 Kubernetes 节点提供，请按上述要求在每个节点安装驱动和工具包，或由 [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/overview.html) 管理。如果已安装 NVIDIA GPU Operator，请禁用其 device plugin、GPU Feature Discovery、Node Feature Discovery、MIG Manager 和 CDI 组件，因为它们会与 GPUStack 的设备管理冲突。详情见[厂商前提条件](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#nvidia)。

运行以下命令验证：

```bash
sudo nvidia-smi

# If using Docker
sudo docker info 2>/dev/null | grep -q "nvidia" \
    && echo "NVIDIA Container Toolkit OK" \
    || (echo "NVIDIA Container Toolkit not configured"; exit 1)
```

#### 支持的推理后端 { #supported-inference-backends }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] [VoxBox](https://github.com/gpustack/vox-box)
- [x] 自定义

### AMD GPU { #amd-gpu }

#### 要求 { #requirements_1 }

- 支持 AMD ROCm 7.0 或更高版本的 [AMD GPU Driver](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)。
- [AMD Container Runtime](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/overview.html)

!!! note "Kubernetes GPU 节点"

    如果 GPU 由 Kubernetes 节点提供，请按上述要求在每个节点安装驱动，或由 [AMD GPU Operator](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/index.html) 管理。如果已安装 AMD GPU Operator，请禁用其 device plugin、node labeller、内置 Node Feature Discovery 和 Kernel Module Management 组件，因为它们会与 GPUStack 的设备管理冲突。详情见[厂商前提条件](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#amd)。

运行以下命令验证：

```bash
sudo amd-smi static

# If using Docker
sudo docker info 2>/dev/null | grep -q "amd" \
    && echo "AMD Container Toolkit OK" \
    || (echo "AMD Container Toolkit not configured"; exit 1)
```

#### 支持的推理后端 { #supported-inference-backends_1 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)（需要 gfx9 系列 GPU）
- [x] 自定义

### 昇腾 NPU { #ascend-npu }

#### 要求 { #requirements_2 }

- [昇腾 NPU Driver](https://www.hiascend.com/hardware/firmware-drivers/community)
- [昇腾 Docker Runtime](https://www.hiascend.com/document/detail/zh/mindcluster/730/clustersched/dlug/dlug_installation_017.html)

运行以下命令验证：

```bash
sudo npu-smi info

# If using Docker
sudo docker info 2>/dev/null | grep -q "ascend" \
    && echo "Ascend Container Toolkit OK" \
    || (echo "Ascend Container Toolkit not configured"; exit 1)
```

#### 支持的设备 { #supported-devices }

- [x] 昇腾 NPU 910C 系列
- [x] 昇腾 NPU 910B 系列（910B1 ~ 910B4）
- [x] 昇腾 NPU 310P3

#### 支持的推理后端 { #supported-inference-backends_2 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] [MindIE](https://www.hiascend.com/document/detail/zh/mindie/21RC2/index/index.html)
- [x] 自定义

### 海光 DCU { #hygon-dcu }

#### 要求 { #requirements_3 }

- [海光 DCU Driver](https://developer.sourcefind.cn/tool/)
- [海光 DTK Toolkit](https://developer.sourcefind.cn/tool/)

!!! note "Kubernetes GPU 节点"

    如果 DCU 由 Kubernetes 节点提供，请按上述要求在每个节点安装驱动和工具包，或由 [海光 DCU Operator](https://developer.sourcefind.cn/document/7541efc4-54b1-11f1-8265-0242ac150003) 管理。如果已安装海光 DCU Operator，请禁用其 device plugin 和 node labeller 组件，因为它们会与 GPUStack 的设备管理冲突。详情见[厂商前提条件](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#hygon)。

运行以下命令验证：

```bash
sudo hy-smi
```

#### 支持的设备 { #supported-devices_1 }

- [x] 海光 DCU

#### 支持的推理后端 { #supported-inference-backends_3 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] 自定义

### 沐曦 GPU { #metax-gpu }

#### 要求 { #requirements_4 }

- [沐曦 GPU Driver](https://developer.metax-tech.com/softnova/download?package_kind=Driver)
- [沐曦 MACA SDK](https://developer.metax-tech.com/softnova/download?package_kind=SDK)

!!! note "Kubernetes GPU 节点"

    如果 GPU 由 Kubernetes 节点提供，请按上述要求在每个节点安装驱动和 SDK，或由 [沐曦 GPU Operator](https://developer.metax-tech.com/api/client/document/preview/1411/k8s/00_overview.html) 管理。沐曦 GPU Operator 的 device plugin 无法禁用，因此如果已安装该 Operator，请在保留节点驱动和 SDK 的前提下将其卸载，再将节点添加到 GPUStack。详情见[厂商前提条件](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#metax)。

运行以下命令验证：

```bash
sudo mx-smi
```

#### 支持的推理后端 { #supported-inference-backends_4 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] 自定义

### 摩尔线程 GPU { #mthreads-gpu }

#### 要求 { #requirements_5 }

- [摩尔线程 GPU Driver](https://docs.mthreads.com/)
- [摩尔线程 Container Toolkit](https://developer.mthreads.com/sdk/download/CloudNative)

!!! note "Kubernetes GPU 节点"

    如果 GPU 由 Kubernetes 节点提供，请按上述要求在每个节点安装驱动和工具包，或由 [MT GPU Operator](https://docs.mthreads.com/cloud-native/cloud-native-doc-online/introduction/) 管理。MT GPU Operator 在默认 `full` 模式下无法禁用 device plugin，因此如果已安装该 Operator，请切换到不包含 device plugin 的 `core` 模式，或在保留节点驱动和工具包的前提下卸载该 Operator，再将节点添加到 GPUStack。详情见[厂商前提条件](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#moore-threads)。

运行以下命令验证：

```bash
sudo mthreads-gmi

# If using Docker
sudo docker info 2>/dev/null | grep -q "mthreads" \
    && echo "MThreads Container Toolkit OK" \
    || (echo "MThreads Container Toolkit not configured"; exit 1)
```

#### 支持的推理后端 { #supported-inference-backends_5 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] 自定义

### 天数智芯 GPU { #iluvatar-gpu }

#### 要求 { #requirements_6 }

- [天数智芯 GPU Driver](https://support.iluvatar.com/#/login)
- [天数智芯 Container Toolkit](https://github.com/Deep-Spark/ix-container-toolkit)

!!! note "Kubernetes GPU 节点"

    如果 GPU 由 Kubernetes 节点提供，请按上述要求在每个节点安装驱动和工具包，或由 [ix-GPU-Operator](https://developer.iluvatar.com/docs/generaldocs_ix_gpu_operator) 管理。如果已安装 ix-GPU-Operator，请禁用其 device plugin 和 feature discovery 组件，因为它们会与 GPUStack 的设备管理冲突。详情见[厂商前提条件](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#iluvatar)。

运行以下命令验证：

```bash
sudo ixsmi

# If using Docker
sudo docker info 2>/dev/null | grep -q "iluvatar" \
    && echo "Iluvatar Container Toolkit OK" \
    || (echo "Iluvatar Container Toolkit not configured"; exit 1)
```

#### 支持的推理后端 { #supported-inference-backends_6 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] 自定义

### 寒武纪 MLU { #cambricon-mlu }

#### 要求 { #requirements_7 }

- 寒武纪 MLU Driver
- 寒武纪 NeuWare Toolkit

运行以下命令验证：

```bash
sudo cnmon
```

#### 支持的推理后端 { #supported-inference-backends_7 }

- [x] 自定义

### 平头哥 PPU { #t-head-ppu }

#### 要求 { #requirements_8 }

- 平头哥 PPU Driver
- 平头哥 PPU SDK

运行以下命令验证：

```bash
sudo ppu-smi
```

#### 支持的推理后端 { #supported-inference-backends_8 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] 自定义

## 网络要求 { #networking-requirements }

### 连通性要求 { #connectivity-requirements }

GPUStack 正常运行需要以下网络连通性：

**GPUStack Server 到 Worker：** GPUStack Server 必须能够访问 Worker，以代理推理请求。

**Worker 到 GPUStack Server：** Worker 必须能够访问 GPUStack Server，以注册和发送更新。

**Worker 到 Worker：** 跨多个 Worker 进行分布式推理时需要。

### 端口要求 { #port-requirements }

GPUStack 使用以下端口进行通信：

#### GPUStack Server 端口 { #server-ports }

| 端口 | 说明 |
| --------- | ------------------------------------------------------------ |
| TCP 80 | GPUStack UI 和 API 端点的默认端口 |
| TCP 443 | 启用 TLS 时，GPUStack UI 和 API 端点的默认端口 |
| TCP 10161 | GPUStack Server 指标端点的默认端口 |
| TCP 30080 | GPUStack Server 内部 API 的默认端口 |
| TCP 5432 | 内嵌 Postgres 数据库的默认端口 |

#### Worker 端口 { #worker-ports }

| 端口 | 说明 |
| --------------- | -------------------------------------------------------------- |
| TCP 10150 | GPUStack Worker 的默认端口 |
| TCP 10151 | Worker 指标端点的默认端口 |
| TCP 40000-40063 | 推理服务端口范围 |
| TCP 41000-41999 | Ray 服务端口范围，用于 vLLM 分布式部署 |

##### 使用 Ray 的分布式 vLLM 端口 { #distributed-vllm-with-ray-ports }

使用分布式 vLLM 时，GPUStack 会解析上述 Ray 服务端口范围，并按以下顺序分配：

1. GCS Server 端口，范围内第一个端口
2. Client Server 端口，保留以兼容旧版，已不再使用，见 https://github.com/gpustack/gpustack/issues/4171
3. Dashboard 端口
4. Dashboard gRPC 端口，从 Ray 2.45.0 起不再使用，保留以兼容旧版
5. Dashboard agent gRPC 端口
6. Dashboard agent 监听端口
7. 指标导出端口
8. Node Manager 端口
9. Object Manager 端口
10. Raylet runtime env agent 端口
11. Worker 的最小端口号
12. Worker 的最大端口号，即范围内最后一个端口

Ray 端口详情见 [Ray 文档](https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations)。

#### 内嵌网关端口 { #embedded-gateway-ports }

GPUStack Server 和 Worker 的内嵌网关均使用以下端口进行内部通信。

| 端口 | 主机 | 说明 |
| --------- | --------- | ---------------------------------------------------- |
| TCP 18443 | 127.0.0.1 | 基于文件的 APIServer 通过 HTTPS 提供服务的端口 |
| TCP 15000 | 127.0.0.1 | Envoy 网关管理端口 |
| TCP 15021 | 0.0.0.0 | Envoy 网关健康检查端口 |
| TCP 15090 | 0.0.0.0 | Envoy 网关指标端口 |
| TCP 9876 | 127.0.0.1 | Pilot-discovery 自省端口 |
| TCP 15010 | 127.0.0.1 | Pilot-discovery 通过 HTTP/gRPC 提供 XDS 的端口 |
| TCP 15012 | 127.0.0.1 | Pilot-discovery 通过安全 gRPC 提供 XDS 的端口 |
| TCP 15020 | 0.0.0.0 | Pilot-agent 指标端口 |
| TCP 8888 | 127.0.0.1 | Controller 通过 HTTP 提供 XDS 的端口 |
| TCP 15051 | 127.0.0.1 | Controller 通过 gRPC 提供 XDS 的端口 |
