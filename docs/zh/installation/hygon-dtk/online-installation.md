# 在线安装

## 支持的设备

- [x] 海光 DCU（K100_AI（已验证）、Z100/Z100L/K100（未验证））

## 支持的平台

| 操作系统 | 架构  | 支持的方法                                                                                                                |
| ------- | ----- | ------------------------------------------------------------------------------------------------------------------------- |
| Linux   | AMD64 | [Docker 安装](#docker-installation)（推荐）<br>[安装脚本](#installation-script-deprecated)（已弃用） |

## 先决条件

- [端口要求](../installation-requirements.md#port-requirements)
- llama-box 后端的 CPU 要求：支持 AVX 的 AMD64

检查 CPU 是否受支持：

```bash
lscpu | grep avx
```

- [DCU 驱动 rock-6.3](https://developer.sourcefind.cn/tool/)

检查是否已安装驱动：

```bash
lsmod | grep dcu
```

## Docker 安装 {#docker-installation}

### 支持的后端

- [x] vLLM（仅支持 K100_AI）
- [x] llama-box

### 先决条件

- [Docker](https://docs.docker.com/engine/install/)

### 运行 GPUStack

运行以下命令启动 GPUStack 服务端并**内置 worker**（推荐使用 host 网络模式）：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --device=/dev/kfd \
    --device=/dev/mkfd \
    --device=/dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro \
    --network=host \
    --ipc=host \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-dcu
```

如需更改默认的服务器端口 80，请使用 `--port` 参数：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --device=/dev/kfd \
    --device=/dev/mkfd \
    --device=/dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro \
    --network=host \
    --ipc=host \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-dcu \
    --port 9090
```

如果其他端口发生冲突，或需要自定义启动选项，请参考 [CLI 参考](../../cli-reference/start.md) 了解可用的标志与配置说明。

检查启动日志是否正常：

```bash
docker logs -f gpustack
```

如果日志正常，在浏览器中打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。可运行以下命令获取默认安装的密码：

```bash
docker exec -it gpustack cat /var/lib/gpustack/initial_admin_password
```

###（可选）添加 Worker

你可以向 GPUStack 添加更多 GPU 节点以组成 GPU 集群。需要在其他 GPU 节点上添加 worker，并指定 `--server-url` 与 `--token` 参数加入 GPUStack。

在 GPUStack【服务器节点】上运行以下命令获取用于添加 worker 的 token：

```bash
docker exec -it gpustack cat /var/lib/gpustack/token
```

在【工作节点】上以 worker 方式启动 GPUStack，并**注册到 GPUStack 服务器**，运行以下命令。请将 URL 和 token 替换为你的实际值：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --device=/dev/kfd \
    --device=/dev/mkfd \
    --device=/dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro \
    --network=host \
    --ipc=host \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-dcu \
    --server-url http://your_gpustack_url --token your_gpustack_token
```

!!! note

    1. 支持异构集群。无论设备类型如何，都可以通过指定 `--server-url` 与 `--token` 参数将其作为 worker 加入当前 GPUStack。

    2. 可以在 docker run 命令后追加参数，为 `gpustack start` 命令设置额外标志。配置详情请参阅 [CLI 参考](../../cli-reference/start.md)。

    3. 可以使用 `--ipc=host` 或 `--shm-size` 以允许容器访问宿主机的共享内存。vLLM 和 PyTorch 在底层会用它在进程间共享数据，尤其是在进行张量并行推理时。

## 安装脚本（已弃用） {#installation-script-deprecated}

!!! warning

    自 0.7 版本起，安装脚本方法已弃用。我们推荐在 Linux 上使用 Docker，在 macOS 或 Windows 上使用[桌面安装程序](https://gpustack.ai/)。

### 支持的后端

- [x] llama-box

### 先决条件

- [DCU 工具包 25.04](https://developer.sourcefind.cn/tool/)

检查 GPU 是否被列为 agent：

```bash
rocminfo
```

检查 `hy-smi`：

```bash
/opt/hyhal/bin/hy-smi -i --showmeminfo vram --showpower --showserial --showuse --showtemp --showproductname --showuniqueid --json
```

### 运行 GPUStack

GPUStack 提供脚本将其安装为服务，默认端口为 80。

```bash
curl -sfL https://get.gpustack.ai | sh -s -
```

如需在运行脚本时配置额外的环境变量和启动标志，请参阅[安装脚本](../installation-script.md)。

安装完成后，确认 GPUStack 启动日志正常：

```bash
tail -200f /var/log/gpustack.log
```

如果启动日志正常，在浏览器中打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。可运行以下命令获取默认安装的密码：

```bash
cat /var/lib/gpustack/initial_admin_password
```

如果通过 `--data-dir` 参数设置了数据目录，`initial_admin_password` 文件将位于该目录中。

###（可选）添加 Worker

要向 GPUStack 集群添加 worker，需要在 worker 上安装 GPUStack 时指定服务器 URL 和认证 token。

在 GPUStack【服务器节点】上运行以下命令获取用于添加 worker 的 token：

```bash
cat /var/lib/gpustack/token
```

如果通过 `--data-dir` 参数设置了数据目录，`token` 文件将位于该目录中。

要安装 GPUStack 并以 worker 身份启动，且**注册到 GPUStack 服务器**，请在【工作节点】上运行以下命令。请将 URL 和 token 替换为你的实际值：

```bash
curl -sfL https://get.gpustack.ai | sh -s - --server-url http://your_gpustack_url --token your_gpustack_token
```

安装完成后，确认 GPUStack 启动日志正常：

```bash
tail -200f /var/log/gpustack.log
```