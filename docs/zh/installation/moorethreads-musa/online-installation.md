# 在线安装

## 支持的设备

- [x] 摩尔线程 GPU（MTT S80、MTT S3000、MTT S4000）

## 支持的平台

| 操作系统 | 架构  | 支持的安装方式                                                                                                      |
| ------ | ----- | ------------------------------------------------------------------------------------------------------------------- |
| Linux  | AMD64 | [Docker 安装](#docker-installation)（推荐）<br>[安装脚本](#installation-script-deprecated)（已弃用） |

## 支持的后端

- [x] llama-box

## 先决条件

- [端口要求](../installation-requirements.md#port-requirements)
- 对 llama-box 后端的 CPU 支持：支持 AVX 的 AMD64

检查 CPU 是否受支持：

```bash
lscpu | grep avx
```

- [MTT S80/S3000/S4000 的驱动](https://developer.mthreads.com/sdk/download/musa)

检查驱动是否已安装：

```bash
mthreads-gmi
```

## Docker 安装 {#docker-installation}

- [Docker](https://docs.docker.com/engine/install/)
- [MT 容器工具包](https://developer.mthreads.com/sdk/download/CloudNative)

检查 MT 容器工具包是否已安装并设置为默认运行时：

```bash
# 进入目录并设置 Docker 运行时
# cd /usr/bin/musa && sudo ./docker setup $PWD
docker info | grep Runtimes | grep mthreads
```

### 运行 GPUStack

运行以下命令启动 GPUStack 服务器并启动内置 Worker（推荐使用 host 网络模式）：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-musa
```

如需更改默认服务器端口 80，请使用 `--port` 参数：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-musa \
    --port 9090
```

如果其他端口存在冲突，或你想自定义启动选项，请参考 [CLI 参考](../../cli-reference/start.md) 了解可用的标志与配置说明。

检查启动日志是否正常：

```bash
docker logs -f gpustack
```

如果日志正常，请在浏览器中打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。你可以运行以下命令获取默认安装的密码：

```bash
docker exec -it gpustack cat /var/lib/gpustack/initial_admin_password
```

### （可选）添加 Worker

你可以向 GPUStack 添加更多 GPU 节点以组成 GPU 集群。需要在其他 GPU 节点上添加 Worker，并通过指定 `--server-url` 和 `--token` 参数加入 GPUStack。

在 GPUStack 的【服务器节点】上运行以下命令以获取添加 Worker 所需的 token：

```bash
docker exec -it gpustack cat /var/lib/gpustack/token
```

在【工作节点】上运行以下命令以 Worker 身份启动 GPUStack，并将其注册到 GPUStack 服务器。请务必将 URL 和 token 替换为你的实际值：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-musa \
    --server-url http://your_gpustack_url --token your_gpustack_token
```

!!! note

    1. 支持异构集群。无论设备类型如何，你都可以通过指定 `--server-url` 和 `--token` 参数将其作为 Worker 加入当前 GPUStack。
    
    2. 你可以在 docker run 命令后追加参数，为 `gpustack start` 命令设置额外的标志。配置详情请参考 [CLI 参考](../../cli-reference/start.md)。

## 安装脚本（已弃用） {#installation-script-deprecated}

!!! warning

    自 0.7 版本起，安装脚本方式已弃用。我们建议在 Linux 上使用 Docker，在 macOS 或 Windows 上使用[桌面安装程序](https://gpustack.ai/)。

### 先决条件

- [MUSA SDK](https://developer.mthreads.com/sdk/download/musa)

### 运行 GPUStack

GPUStack 提供脚本将其以服务方式安装，默认端口为 80。

```bash
curl -sfL https://get.gpustack.ai | sh -s -
```

如需在运行脚本时配置其他环境变量与启动标志，请参考[安装脚本](../installation-script.md)。

安装完成后，确认 GPUStack 启动日志正常：

```bash
tail -200f /var/log/gpustack.log
```

如果启动日志正常，请在浏览器中打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。你可以运行以下命令获取默认安装的密码：

```bash
cat /var/lib/gpustack/initial_admin_password
```

如果你通过 `--data-dir` 参数指定了数据目录，则 `initial_admin_password` 文件位于该目录中。

### （可选）添加 Worker

要向 GPUStack 集群添加 Worker，需要在安装 GPUStack 时指定服务器 URL 和认证 token。

在 GPUStack 的【服务器节点】上运行以下命令以获取添加 Worker 所需的 token：

```bash
cat /var/lib/gpustack/token
```

如果你通过 `--data-dir` 参数指定了数据目录，则 `token` 文件位于该目录中。

在【工作节点】上运行以下命令安装 GPUStack 并以 Worker 身份启动，同时将其注册到 GPUStack 服务器。请务必将 URL 和 token 替换为你的实际值：

```bash
curl -sfL https://get.gpustack.ai | sh -s - --server-url http://your_gpustack_url --token your_gpustack_token
```

安装完成后，确认 GPUStack 启动日志正常：

```bash
tail -200f /var/log/gpustack.log
```