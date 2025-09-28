# 在线安装

## 支持的设备

- [x] 寒武纪 MLU

## 支持的平台

| 操作系统 | 架构  | 支持的方法                         |
| ------- | ----- | ---------------------------------- |
| Linux   | AMD64 | [pip 安装](#pip-installation)      |

## 支持的后端

- [x] vLLM

## 前提条件

- 寒武纪驱动

检查是否已安装寒武纪驱动：

```bash
cnmon
```

- 寒武纪 Pytorch Docker 镜像

请联系寒武纪工程师获取寒武纪 Pytorch Docker 镜像。

<a id="pip-installation"></a>

## pip 安装

使用寒武纪 Pytorch Docker 镜像，并激活 `pytorch_infer` 虚拟环境：

```bash
source /torch/venv3/pytorch_infer/bin/activate
```

### 安装 GPUStack

运行以下命令安装 GPUStack。

```bash
# vLLM has been installed in Cambricon Pytorch docker
pip install "gpustack[audio]"
```

运行以下命令验证安装：

```bash
gpustack version
```

### 运行 GPUStack

运行以下命令启动 GPUStack 服务器并启动内置 Worker：

```bash
gpustack start
```

如果启动日志正常，在浏览器中打开 `http://your_host_ip` 访问 GPUStack UI。使用用户名 `admin` 和默认密码登录。你可以通过以下命令获取默认安装的密码：

```bash
cat /var/lib/gpustack/initial_admin_password
```

默认情况下，GPUStack 使用 `/var/lib/gpustack` 作为数据目录，因此需要 `sudo` 或相应权限。你也可以通过以下命令设置自定义数据目录：

```bash
gpustack start --data-dir mypath
```

可参考 [CLI 参考](../../cli-reference/start.md) 了解可用的 CLI 标志。

### （可选）添加 Worker

要向 GPUStack 集群添加 Worker，需要指定服务器 URL 和认证令牌（token）。

在 GPUStack 服务器节点上运行以下命令以获取用于添加 Worker 的 token：

```bash
cat /var/lib/gpustack/token
```

在 Worker 节点上运行以下命令将 GPUStack 以 Worker 模式启动，并注册到 GPUStack 服务器。请将 URL、token 和节点 IP 替换为你的实际值：

```bash
gpustack start --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_worker_host_ip
```