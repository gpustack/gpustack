# 在线安装

## 支持的设备

- [x] Ascend 910B 系列（910B1 ~ 910B4）
- [x] Ascend 310P3

## 支持的平台

| 操作系统 | 架构  | 支持的方法                                                          |
| -------- | ----- |----------------------------------------------------------------|
| Linux    | ARM64 | [Docker 安装](#docker-installation)（推荐）<br>[安装脚本](#install-scripts)（已弃用） |

## 先决条件

- [端口要求](../installation-requirements.md#port-requirements)
- llama-box 后端的 CPU 要求：具备 NEON 的 ARM64

检查 CPU 是否受支持：

```bash
grep -E -i "neon|asimd" /proc/cpuinfo
```

- [NPU 驱动与固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.2.RC1&driver=Ascend+HDK+25.2.0)（必须支持 CANN 8.2.RC1）

检查是否已安装 NPU 驱动：

```bash
npu-smi info
```

<a id="docker-installation"></a>

## Docker 安装

### 支持的后端

- [x] llama-box（仅支持 FP16 精度）
- [x] MindIE
- [x] vLLM

### 前置条件

- [Docker](https://docs.docker.com/engine/install/)

### 运行 GPUStack

运行以下命令以启动 GPUStack 服务器和内置 worker（推荐使用 host 网络模式）。将 `--device /dev/davinci{index}` 设置为所需的 GPU 索引：

=== "Ascend 910B"

    按以下步骤在 Ascend 910B 上安装 GPUStack：
    
    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-npu
    ```
    
    如果需要修改默认的服务器端口 80，请使用 `--port` 参数：
    
    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-npu \
        --port 9090
    ```

=== "Ascend 310P"

    按以下步骤在 Ascend 310P 上安装 GPUStack：
    
    === "主机网络"
    
    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-npu-310p
    ```
    
    如果需要修改默认的服务器端口 80，请使用 `--port` 参数：
    
    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-npu-310p \
        --port 9090
    ```

如果其他端口发生冲突，或你想自定义启动选项，请参考[CLI 参考](../../cli-reference/start.md)以获取可用的标志与配置说明。

检查启动日志是否正常：

```bash
docker logs -f gpustack
```

如果日志正常，在浏览器中打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。你可以运行以下命令获取默认安装的密码：

```bash
docker exec -it gpustack cat /var/lib/gpustack/initial_admin_password
```

### （可选）添加 Worker

你可以向 GPUStack 添加更多 GPU 节点以组成 GPU 集群。需要在其他 GPU 节点上添加 worker，并指定 `--server-url` 和 `--token` 参数以加入 GPUStack。

获取用于添加 worker 的 token，请在 GPUStack 的服务器节点上运行以下命令：

```bash
docker exec -it gpustack cat /var/lib/gpustack/token
```

要将 GPUStack 作为 worker 启动，并将其注册到 GPUStack 服务器（将 `--device /dev/davinci{index}` 设置为所需的 GPU 索引），请在 worker 节点上运行以下命令。请务必将 URL 和 token 替换为你的实际值：

=== "Ascend 910B"

    按以下步骤在 Ascend 910B 上添加 worker：
    
    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-npu \
        --server-url http://your_gpustack_url --token your_gpustack_token
    ```

=== "Ascend 310P"

    按以下步骤在 Ascend 310P 上添加 worker：
    
    ```bash
    docker run -d --name gpustack \
        --restart=unless-stopped \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        --network=host \
        --ipc=host \
        -v gpustack-data:/var/lib/gpustack \
        gpustack/gpustack:latest-npu-310p \
        --server-url http://your_gpustack_url --token your_gpustack_token
    ```

!!! note

    1. 支持异构集群。无论设备类型为何，通过指定 `--server-url` 和 `--token` 参数，都可以将其作为 worker 加入当前 GPUStack。
    
    2. 你可以在 docker run 命令后追加参数，为 `gpustack start` 命令设置额外的标志。配置详情请参考[CLI 参考](../../cli-reference/start.md)。
    
    3. 你可以使用 `--ipc=host` 或 `--shm-size` 使容器访问宿主机的共享内存。vLLM 和 PyTorch 会在底层使用其在进程间共享数据，尤其是在进行张量并行推理时。

<a id="install-scripts"></a>

## 安装脚本（已弃用）

!!! warning

    自 0.7 版本起，安装脚本方式已弃用。我们建议在 Linux 上使用 Docker，在 macOS 或 Windows 上使用[桌面安装程序](https://gpustack.ai/)。

### 支持的后端

- [x] llama-box（仅支持 Ascend 910B，且仅支持 FP16 精度）

### 先决条件

- [Ascend CANN Toolkit 8.2.RC1.beta1 与 Kernels](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)

检查是否已安装 CANN，并确认其版本为 8.2.RC1：

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

检查是否已安装 CANN 内核，并确认其版本为 8.2.RC1：

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg | grep opp
```

### 运行 GPUStack

GPUStack 提供了一个脚本，可将其安装为服务，默认端口为 80。

```bash
curl -sfL https://get.gpustack.ai | sh -s -
```

若需在运行脚本时配置额外的环境变量和启动标志，请参考[安装脚本](../installation-script.md)。

安装完成后，确认 GPUStack 启动日志正常：

```bash
tail -200f /var/log/gpustack.log
```

如果启动日志正常，在浏览器中打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。你可以运行以下命令获取默认安装的密码：

```bash
cat /var/lib/gpustack/initial_admin_password
```

如果你通过 `--data-dir` 参数指定了数据目录，`initial_admin_password` 文件将位于该目录中。

### （可选）添加 Worker

要向 GPUStack 集群添加 worker，需要在 worker 上安装 GPUStack 时指定服务器 URL 和认证 token。

获取用于添加 worker 的 token，请在 GPUStack 的服务器节点上运行以下命令：

```bash
cat /var/lib/gpustack/token
```

如果你通过 `--data-dir` 参数指定了数据目录，`token` 文件将位于该目录中。

要安装 GPUStack、将其作为 worker 启动并注册到 GPUStack 服务器，请在 worker 节点上运行以下命令。请务必将 URL 和 token 替换为你的实际值：

```bash
curl -sfL https://get.gpustack.ai | sh -s - --server-url http://your_gpustack_url --token your_gpustack_token
```

安装完成后，确认 GPUStack 启动日志正常：

```bash
tail -200f /var/log/gpustack.log
```