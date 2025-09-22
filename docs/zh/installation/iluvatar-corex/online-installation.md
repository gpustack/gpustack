# 在线安装

## 支持的设备

- [x] Iluvatar GPU（MR-V100、MR-V50、BI-V100、BI-V150）

## 支持的平台

| 操作系统 | 架构  | 支持的方法                                   |
| ------- | ----- | -------------------------------------------- |
| Linux   | AMD64 | [Docker 安装](#docker-installation) |

## 支持的后端

- [x] vLLM

## 前置条件

- [MR-V100 MR-V50 BI-V100 BI-V150 的驱动](https://support.iluvatar.com/#/ProductLine?id=2)

检查是否已安装驱动：

```bash
ixsmi
```

## Docker 安装 {#docker-installation}

- [Docker](https://support.iluvatar.com/#/ProductLine?id=2)
- [Corex 容器工具包](https://support.iluvatar.com/#/ProductLine?id=2)

### 运行 GPUStack

运行以下命令以启动 GPUStack 服务器及内置 Worker（推荐使用 host 网络模式）：

```bash
docker run -d --name gpustack \
    -v /lib/modules:/lib/modules \
    -v /dev:/dev \
    --privileged \
    --cap-add=ALL \
    --pid=host \
    --restart=unless-stopped \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-corex
```

如果需要修改默认的服务器端口 80，请使用 `--port` 参数：

```bash
docker run -d --name gpustack \
    -v /lib/modules:/lib/modules \
    -v /dev:/dev \
    --privileged \
    --cap-add=ALL \
    --pid=host \
    --restart=unless-stopped \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-corex \
    --port 9090
```

如果其他端口发生冲突，或你想自定义启动选项，请参考[命令行参考](../../cli-reference/start.md)了解可用标志与配置说明。

检查启动日志是否正常：

```bash
docker logs -f gpustack
```

如果日志正常，在浏览器中打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。你可以运行以下命令获取默认设置的密码：

```bash
docker exec -it gpustack cat /var/lib/gpustack/initial_admin_password
```

### （可选）添加 Worker

你可以向 GPUStack 添加更多 GPU 节点以组成 GPU 集群。需要在其他 GPU 节点上添加 Worker，并指定 `--server-url` 与 `--token` 参数以加入 GPUStack。

在 GPUStack「服务器节点」上运行以下命令以获取用于添加 Worker 的 token：

```bash
docker exec -it gpustack cat /var/lib/gpustack/token
```

在「Worker 节点」上运行以下命令，以 Worker 方式启动 GPUStack，并将其注册到 GPUStack 服务器。请将 URL 与 token 替换为你的实际值：

```bash
docker run -d --name gpustack \
    -v /lib/modules:/lib/modules \
    -v /dev:/dev \
    --privileged \
    --cap-add=ALL \
    --pid=host \
    --restart=unless-stopped \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-corex \
    --server-url http://your_gpustack_url --token your_gpustack_token
```

!!! note

    1. 支持异构集群。无论设备类型如何，只需指定 `--server-url` 和 `--token` 参数，即可将其作为 Worker 加入当前 GPUStack。
    
    2. 你可以通过在 docker run 命令后追加参数，为 `gpustack start` 命令设置更多标志。配置详情请参考[命令行参考](../../cli-reference/start.md)。