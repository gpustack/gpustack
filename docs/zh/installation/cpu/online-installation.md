# 在线安装

在 GPUStack 中，`llama-box` 和 `vox-box` 后端支持在 CPU 上进行推理。但与 GPU 相比，CPU 性能明显较低，因此仅建议用于测试或小规模场景。

## 支持的设备

- [x] CPU（支持 AVX 的 AMD64 或支持 NEON 的 ARM64）

## 支持的平台

| 操作系统 | 架构           | 支持的方法                                                                                                                                                                  |
| -------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux    | AMD64<br>ARM64 | [Docker 安装](#docker-installation)（推荐）<br>[pip 安装](#pip-installation)<br>[安装脚本](#installation-script-deprecated)（已弃用）                                      |
| Windows  | AMD64<br>ARM64 | [桌面安装程序](../desktop-installer.md)（推荐）<br>[pip 安装](#pip-installation)<br>[安装脚本](#installation-script-deprecated)（已弃用）                                  |

## 支持的后端

- [x] llama-box
- [x] vox-box

## 前置条件

- [端口要求](../installation-requirements.md#port-requirements)
- CPU（支持 AVX 的 AMD64 或支持 NEON 的 ARM64）

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

    Windows 用户需要手动按照上述说明进行支持性检查。

## Docker 安装 {#docker-installation}

### 前置条件

- [Docker](https://docs.docker.com/engine/install/)

### 运行 GPUStack

运行以下命令以启动 GPUStack 服务器和内置 Worker（推荐使用 host 网络模式）：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --network=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-cpu
```

如果需要修改默认的服务器端口 80，请使用 `--port` 参数：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --network=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-cpu \
    --port 9090
```

如遇端口冲突，或希望自定义启动选项，请参考 [CLI 参考](../../cli-reference/start.md) 获取可用标志与配置说明。

检查启动日志是否正常：

```bash
docker logs -f gpustack
```

如果日志正常，在浏览器中打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。你可以运行以下命令获取默认安装的密码：

```bash
docker exec -it gpustack cat /var/lib/gpustack/initial_admin_password
```

### （可选）添加 Worker

你可以向 GPUStack 添加更多 CPU 节点。需要在其他 CPU 节点上添加 Worker，并指定 `--server-url` 和 `--token` 参数以加入 GPUStack。

在 GPUStack「服务器节点」上运行以下命令以获取用于添加 Worker 的令牌：

```bash
docker exec -it gpustack cat /var/lib/gpustack/token
```

在 Worker 节点上运行以下命令，将 GPUStack 作为 Worker 启动，并注册到 GPUStack 服务器。请将 URL 和 token 替换为你的实际值：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --network=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-cpu \
    --server-url http://your_gpustack_url --token your_gpustack_token
```

!!! note

    1. 支持异构集群。无论设备类型为何，只需指定 `--server-url` 与 `--token` 参数，即可将其作为 Worker 加入当前 GPUStack。
    
    2. 你可以在 docker run 命令末尾追加参数，为 `gpustack start` 命令设置额外标志。配置细节请参考 [CLI 参考](../../cli-reference/start.md)。

## pip 安装 {#pip-installation}

### 前置条件

- Python 3.10 ~ 3.12

查看 Python 版本：

```bash
python -V
```

### 安装 GPUStack

运行以下命令安装 GPUStack：

```bash
pip install "gpustack[audio]"
```

如果不需要音频模型支持，运行：

```bash
pip install gpustack
```

验证安装：

```bash
gpustack version
```

### 运行 GPUStack

运行以下命令启动 GPUStack 服务器和内置 Worker：

```bash
gpustack start
```

如果启动日志正常，在浏览器中打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。你可以运行以下命令获取默认安装的密码：

=== "Linux"

    ```bash
    cat /var/lib/gpustack/initial_admin_password
    ```

=== "Windows"

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\initial_admin_password" -Raw
    ```

默认情况下，GPUStack 使用 `/var/lib/gpustack` 作为数据目录，因此需要 `sudo` 或相应权限。你也可以通过以下命令设置自定义数据目录：

```bash
gpustack start --data-dir mypath
```

可用的 CLI 标志可参考 [CLI 参考](../../cli-reference/start.md)。

### （可选）添加 Worker

向 GPUStack 集群添加 Worker 需要指定服务器 URL 和认证令牌。

在 GPUStack「服务器节点」上运行以下命令以获取用于添加 Worker 的令牌：

=== "Linux"

    ```bash
    cat /var/lib/gpustack/token
    ```

=== "Windows"

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\token" -Raw
    ```

在「Worker 节点」上运行以下命令，将 GPUStack 作为 Worker 启动，并注册到 GPUStack 服务器。请将 URL、token 和节点 IP 替换为你的实际值：

```bash
gpustack start --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_worker_host_ip
```

### 将 GPUStack 作为系统服务运行

推荐做法是将 GPUStack 配置为开机自启动服务。例如，使用 systemd：

在 `/etc/systemd/system/gpustack.service` 创建服务文件：

```bash
sudo tee /etc/systemd/system/gpustack.service > /dev/null <<EOF
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

并确认 GPUStack 启动日志正常：

```bash
tail -200f /var/log/gpustack.log
```

## 安装脚本（已弃用） {#installation-script-deprecated}

!!! warning

    自 0.7 版本起，安装脚本方式已弃用。我们建议在 Linux 上使用 Docker，在 macOS 或 Windows 上使用[桌面安装程序](https://gpustack.ai/)。

GPUStack 提供脚本，以默认端口 80 将其安装为服务。

=== "Linux"

    - 安装服务器

    ```bash
    curl -sfL https://get.gpustack.ai | sh -s -
    ```

    若需在运行脚本时配置额外环境变量和启动标志，请参考[安装脚本](../installation-script.md)。

    安装完成后，确认 GPUStack 启动日志正常：

    ```bash
    tail -200f /var/log/gpustack.log
    ```

    如果启动日志正常，在浏览器中打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。你可以运行以下命令获取默认安装的密码：

    ```bash
    cat /var/lib/gpustack/initial_admin_password
    ```

    如果通过 `--data-dir` 参数指定了数据目录，则 `initial_admin_password` 文件位于指定目录中。

    - （可选）添加 Worker

    向 GPUStack 集群添加 Worker 时，需要在 Worker 上安装 GPUStack 时指定服务器 URL 和认证令牌。

    在 GPUStack「服务器节点」上运行以下命令以获取用于添加 Worker 的令牌：

    ```bash
    cat /var/lib/gpustack/token
    ```

    如果通过 `--data-dir` 参数指定了数据目录，则 `token` 文件位于指定目录中。

    在「Worker 节点」上运行以下命令安装 GPUStack 并作为 Worker 启动，且注册到 GPUStack 服务器。请将 URL 和 token 替换为你的实际值：

    ```bash
    curl -sfL https://get.gpustack.ai | sh -s - --server-url http://your_gpustack_url --token your_gpustack_token
    ```

    安装完成后，确认 GPUStack 启动日志正常：

    ```bash
    tail -200f /var/log/gpustack.log
    ```

=== "Windows"

    - 安装服务器

    ```powershell
    Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content
    ```

    若需在运行脚本时配置额外环境变量和启动标志，请参考[安装脚本](../installation-script.md)。

    安装完成后，确认 GPUStack 启动日志正常：

    ```powershell
    Get-Content "$env:APPDATA\gpustack\log\gpustack.log" -Tail 200 -Wait
    ```

    如果启动日志正常，在浏览器中打开 `http://your_host_ip` 访问 GPUStack 界面。使用用户名 `admin` 和默认密码登录。你可以运行以下命令获取默认安装的密码：

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\initial_admin_password" -Raw
    ```

    如果通过 `--data-dir` 参数指定了数据目录，则 `initial_admin_password` 文件位于指定目录中。

    - （可选）添加 Worker

    向 GPUStack 集群添加 Worker 时，需要在 Worker 上安装 GPUStack 时指定服务器 URL 和认证令牌。

    在 GPUStack「服务器节点」上运行以下命令以获取用于添加 Worker 的令牌：

    ```powershell
    Get-Content -Path "$env:APPDATA\gpustack\token" -Raw
    ```

    如果通过 `--data-dir` 参数指定了数据目录，则 `token` 文件位于指定目录中。

    在 Worker 节点上运行以下命令安装 GPUStack 并作为 Worker 启动，且注册到 GPUStack 服务器。请将 URL 和 token 替换为你的实际值：

    ```powershell
    Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --server-url http://your_gpustack_url --token your_gpustack_token"
    ```

    安装完成后，确认 GPUStack 启动日志正常：

    ```powershell
    Get-Content "$env:APPDATA\gpustack\log\gpustack.log" -Tail 200 -Wait
    ```