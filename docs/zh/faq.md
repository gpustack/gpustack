# 常见问题

## 支持矩阵

### 混合集群支持

GPUStack 支持混合的 Linux、Windows 与 macOS 节点，以及 x86_64 与 arm64 架构。此外，还支持多种 GPU/加速器，包括 NVIDIA、Apple
Metal、AMD、Ascend、Hygon、Moore Threads、Iluvatar 和 Cambricon。

### 分布式推理支持

**单节点多 GPU**

- [x] vLLM
- [x] MindIE
- [x] llama-box（不支持图像生成模型）
- [ ] vox-box

**多节点多 GPU**

- [x] vLLM
- [x] MindIE
- [x] llama-box

**异构节点多 GPU**

- [x] llama-box

> 相关文档：
>
> **vLLM**：[分布式推理与服务](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
>
> **MindIE**：[多节点推理](https://www.hiascend.com/document/detail/zh/mindie/20RC2/envdeployment/instg/mindie_instg_0027.html)
>
> **llama-box**：[使用 llama.cpp 进行分布式 LLM 推理](https://github.com/ggml-org/llama.cpp/tree/master/examples/rpc)
>

## 安装

### 如何更改 GPUStack 默认端口？

默认情况下，GPUStack 服务使用 80 端口。你可以通过以下方式修改：

#### Docker

在 `docker run` 命令末尾添加 `--port` 参数，例如：

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

#### Installer

点击（macOS 菜单栏或 Windows 系统托盘中的）`GPUStack 图标`，选择 `Quick Config`，修改端口后重启服务生效。

![Quick Config](../assets/faq/quick-config.png)

#### pip

在 `gpustack start` 命令末尾添加 `--port` 参数：

```bash
gpustack start --port 9090
```

#### Script (Legacy)

- Linux

```bash
sudo vim /etc/systemd/system/gpustack.service
```

添加 `--port` 参数：

```bash
ExecStart=/root/.local/bin/gpustack start --port 9090
```

保存并重启 GPUStack：

```bash
sudo systemctl daemon-reload && sudo systemctl restart gpustack
```

- macOS

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
sudo vim /Library/LaunchDaemons/ai.gpustack.plist
```

添加 `--port` 参数：

```bash
  <array>
    <string>/Users/gpustack/.local/bin/gpustack</string>
    <string>start</string>
    <string>--port</string>
    <string>9090</string>
  </array>
```

保存并启动 GPUStack：

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

```powershell
nssm edit GPUStack
```

在 `start` 后添加参数：

```
start --port 9090
```

保存并重启 GPUStack：

```powershell
Restart-Service -Name "GPUStack"
```

---

### 如何更改已注册的 worker 名称？

运行 GPUStack 时可通过 `--worker-name` 参数设置为自定义名称：

#### Docker

在 `docker run` 命令末尾添加 `--worker-name` 参数，例如：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack \
    --worker-name New-Name
```

#### Installer

- macOS

点击菜单栏中的 `GPUStack 图标`，选择 `Config Directory` 打开 Finder。编辑 `config.yaml` 文件，添加 `worker_name` 参数：

```yaml
worker_name: New-Name
```

保存后再次点击 `GPUStack 图标`，进入 `Status - Restart` 以应用配置。

- Windows

右键系统托盘中的 `GPUStack 图标`，选择 `Config Directory` 打开资源管理器。编辑 `config.yaml` 文件，添加 `worker_name` 参数：

```yaml
worker_name: New-Name
```

保存后再次右键 `GPUStack 图标`，进入 `Status - Restart` 以应用配置。

#### pip

在 `gpustack start` 命令末尾添加 `--worker-name` 参数：

```bash
gpustack start --worker-name New-Name
```

#### Script (Legacy)

- Linux

```bash
sudo vim /etc/systemd/system/gpustack.service
```

添加 `--worker-name` 参数：

```bash
ExecStart=/root/.local/bin/gpustack start --worker-name New-Name
```

保存并重启 GPUStack：

```bash
sudo systemctl daemon-reload && sudo systemctl restart gpustack
```

- macOS

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
sudo vim /Library/LaunchDaemons/ai.gpustack.plist
```

添加 `--worker-name` 参数：

```bash
  <array>
    <string>/Users/gpustack/.local/bin/gpustack</string>
    <string>start</string>
    <string>--worker-name</string>
    <string>New-Name</string>
  </array>
```

保存并启动 GPUStack：

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

```powershell
nssm edit GPUStack
```

在 `start` 后添加参数：

```
start --worker-name New-Name
```

保存并重启 GPUStack：

```powershell
Restart-Service -Name "GPUStack"
```

---

### 如何更改已注册的 worker IP？

运行 GPUStack 时可通过 `--worker-ip` 参数设置为自定义 IP：

#### Docker

在 `docker run` 命令末尾添加 `--worker-ip` 参数，例如：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack \
    --worker-ip xx.xx.xx.xx
```

#### Installer

- macOS

点击菜单栏中的 `GPUStack 图标`，选择 `Config Directory` 打开 Finder。编辑 `config.yaml` 文件，添加 `worker_ip` 参数：

```yaml
worker_ip: xx.xx.xx.xx
```

保存后再次点击 `GPUStack 图标`，进入 `Status - Restart` 以应用配置。

- Windows

右键系统托盘中的 `GPUStack 图标`，选择 `Config Directory` 打开资源管理器。编辑 `config.yaml` 文件，添加 `worker_ip` 参数：

```yaml
worker_ip: xx.xx.xx.xx
```

保存后再次右键 `GPUStack 图标`，进入 `Status - Restart` 以应用配置。

#### pip

在 `gpustack start` 命令末尾添加 `--worker-ip` 参数：

```bash
gpustack start --worker-ip xx.xx.xx.xx
```

#### Script (Legacy)

- Linux

```bash
sudo vim /etc/systemd/system/gpustack.service
```

添加 `--worker-ip` 参数：

```bash
ExecStart=/root/.local/bin/gpustack start --worker-ip xx.xx.xx.xx
```

保存并重启 GPUStack：

```bash
sudo systemctl daemon-reload && sudo systemctl restart gpustack
```

- macOS

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
sudo vim /Library/LaunchDaemons/ai.gpustack.plist
```

添加 `--worker-ip` 参数：

```bash
  <array>
    <string>/Users/gpustack/.local/bin/gpustack</string>
    <string>start</string>
    <string>--worker-ip</string>
    <string>xx.xx.xx.xx</string>
  </array>
```

保存并启动 GPUStack：

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

```powershell
nssm edit GPUStack
```

在 `start` 后添加参数：

```
start --worker-ip xx.xx.xx.xx
```

保存并重启 GPUStack：

```powershell
Restart-Service -Name "GPUStack"
```

---

### GPUStack 的数据存储在哪里？

#### Docker

运行 GPUStack 容器时，通过 `-v` 参数挂载 Docker 卷。默认数据路径位于 Docker 数据目录下的 volumes 子目录中，默认路径为：

```bash
/var/lib/docker/volumes/gpustack-data/_data
```

你可以通过以下方式查看：

```bash
docker volume ls
docker volume inspect gpustack-data
```

如需改为自定义路径，请在运行容器时修改挂载配置。例如，将主机目录 `/data/gpustack` 进行挂载：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v /data/gpustack:/var/lib/gpustack  \
    gpustack/gpustack
```

#### Installer

默认路径如下：

- macOS

```bash
/Library/Application Support/GPUStack
```

- Windows

```powershell
C:\ProgramData\GPUStack
```

使用安装包时不允许修改数据目录。

#### pip

默认路径如下：

- Linux & macOS

```bash
/var/lib/gpustack
```

- Windows

```powershell
"$env:APPDATA\gpustack"
```

运行 GPUStack 时可通过 `--data-dir` 参数设置为自定义路径：

```bash
gpustack start --data-dir /data/gpustack-data
```

#### Script (Legacy)

- Linux

默认路径如下：

```bash
/var/lib/gpustack
```

运行 GPUStack 时可通过 `--data-dir` 参数设置为自定义路径：

```bash
sudo vim /etc/systemd/system/gpustack.service
```

添加 `--data-dir` 参数：

```bash
ExecStart=/root/.local/bin/gpustack start --data-dir /data/gpustack-data
```

保存并重启 GPUStack：

```bash
sudo systemctl daemon-reload && sudo systemctl restart gpustack
```

- macOS

默认路径如下：

```bash
/var/lib/gpustack
```

运行 GPUStack 时可通过 `--data-dir` 参数设置为自定义路径：

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
sudo vim /Library/LaunchDaemons/ai.gpustack.plist
```

```bash
  <array>
    <string>/Users/gpustack/.local/bin/gpustack</string>
    <string>start</string>
    <string>--data-dir</string>
    <string>/Users/gpustack/data/gpustack-data</string>
  </array>
```

保存并启动 GPUStack：

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

默认路径如下：

```powershell
"$env:APPDATA\gpustack"
```

运行 GPUStack 时可通过 `--data-dir` 参数设置为自定义路径：

```powershell
nssm edit GPUStack
```

在 `start` 后添加参数：

```
start --data-dir D:\gpustack-data
```

保存并重启 GPUStack：

```powershell
Restart-Service -Name "GPUStack"
```

---

### 模型文件存储在哪里？

#### Docker

运行 GPUStack 容器时，通过 `-v` 参数挂载 Docker 卷。默认缓存路径位于 Docker 数据目录下的 volumes 子目录中，默认路径为：

```bash
/var/lib/docker/volumes/gpustack-data/_data/cache
```

你可以通过以下方式查看：

```bash
docker volume ls
docker volume inspect gpustack-data
```

如需改为自定义路径，请在运行容器时修改挂载配置。

> **注意**：如果已挂载数据目录，缓存目录不应挂载在该数据目录内。需要使用 `--cache-dir` 参数指定不同的路径。

例如，将主机目录 `/data/model-cache` 进行挂载：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v /data/gpustack:/var/lib/gpustack  \
    -v /data/model-cache:/data/model-cache \
    gpustack/gpustack \
    --cache-dir /data/model-cache
```

#### Installer

- macOS

默认路径如下：

```bash
/Library/Application Support/GPUStack/cache
```

如需改为自定义路径，点击菜单栏中的 `GPUStack 图标`，选择 `Config Directory` 打开 Finder。编辑 `config.yaml` 文件，添加
`cache_dir` 参数：

```yaml
cache_dir: /Users/gpustack/data/model-cache
```

保存后再次点击 `GPUStack 图标`，进入 `Status - Restart` 以应用配置。

- Windows

默认路径如下：

```powershell
C:\ProgramData\GPUStack\cache
```

如需改为自定义路径，点击菜单栏中的 `GPUStack 图标`，选择 `Config Directory` 打开 Finder。编辑 `config.yaml` 文件，添加
`cache_dir` 参数：

```yaml
cache_dir: D:\model-cache
```

保存后再次右键 `GPUStack 图标`，进入 `Status - Restart` 以应用配置。

#### pip

默认路径如下：

- Linux & macOS

```bash
/var/lib/gpustack/cache
```

- Windows

```powershell
"$env:APPDATA\gpustack\cache"
```

运行 GPUStack 时可通过 `--cache-dir` 参数设置为自定义路径：

```bash
gpustack start --cache-dir /data/model-cache
```

#### Script (Legacy)

- Linux

默认路径如下：

```bash
/var/lib/gpustack/cache
```

运行 GPUStack 时可通过 `--cache-dir` 参数设置为自定义路径：

```bash
sudo vim /etc/systemd/system/gpustack.service
```

添加 `--cache-dir` 参数：

```bash
ExecStart=/root/.local/bin/gpustack start --cache-dir /data/model-cache
```

保存并重启 GPUStack：

```bash
sudo systemctl daemon-reload && sudo systemctl restart gpustack
```

- macOS

默认路径如下：

```bash
/var/lib/gpustack/cache
```

运行 GPUStack 时可通过 `--cache-dir` 参数设置为自定义路径：

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
sudo vim /Library/LaunchDaemons/ai.gpustack.plist
```

```bash
  <array>
    <string>/Users/gpustack/.local/bin/gpustack</string>
    <string>start</string>
    <string>--cache-dir</string>
    <string>/Users/gpustack/data/model-cache</string>
  </array>
```

保存并启动 GPUStack：

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

默认路径如下：

```powershell
"$env:APPDATA\gpustack\cache"
```

运行 GPUStack 时可通过 `--cache-dir` 参数设置为自定义路径：

```powershell
nssm edit GPUStack
```

在 `start` 后添加参数：

```
start --cache-dir D:\model-cache
```

保存并重启 GPUStack：

```powershell
Restart-Service -Name "GPUStack"
```

---

### 启动 GPUStack 时可以设置哪些参数？

请参阅：[gpustack start](cli-reference/start.md)

---

## 升级

### 如何升级内置的 vLLM？

GPUStack 支持多个版本的推理后端。部署模型时，可在 `Edit Model` → `Advanced` → `Backend Version`
中指定后端版本，以使用[新发布的 vLLM 版本](https://github.com/vllm-project/vllm/releases)。GPUStack 将通过 pipx
自动创建虚拟环境并安装：

![pin-vllm-backend-version](../assets/faq/pin-vllm-backend-version.png)

也可以手动安装自定义版本（包括使用自定义 PyPI 源），然后将可执行文件链接到 `/var/lib/gpustack/bin/`，之后按上述方式配置使用。

```bash
python3 -m venv $(pipx environment --value PIPX_LOCAL_VENS)/vllm-v0-8-5-post1
$(pipx environment --value PIPX_LOCAL_VENVS)/vllm-v0-8-5-post1/bin/python -m pip install vllm==v0.8.5.post1 -i https://mirrors.aliyun.com/pypi/simple
sudo mkdir -p /var/lib/gpustack/bin
sudo ln -s $(pipx environment --value PIPX_LOCAL_VENVS)/vllm-v0-8-5-post1/bin/vllm /var/lib/gpustack/bin/vllm_v0.8.5.post1
```

如果仍需升级内置的 vLLM，可按以下方式在所有 worker 节点上升级 vLLM：

**Docker**

```bash
docker exec -it gpustack bash
pip list | grep vllm
pip install -U vllm
```

**pip**

```bash
pip list | grep vllm
pip install -U vllm
```

**Script (Legacy)**

```bash
pipx runpip gpustack list | grep vllm
pipx runpip gpustack install -U vllm
```

---

### 如何升级内置的 Transformers？

**Docker**

```bash
docker exec -it gpustack bash
pip list | grep transformers
pip install -U transformers
```

**pip**

```bash
pip list | grep transformers
pip install -U transformers
```

**Script (Legacy)**

```bash
pipx runpip gpustack list | grep transformers
pipx runpip gpustack install -U transformers
```

---

### 如何升级内置的 llama-box？

GPUStack 支持多个版本的推理后端。部署模型时，可在 `Edit Model` → `Advanced` → `Backend Version`
中指定后端版本，以使用[新发布的 llama-box 版本](https://github.com/gpustack/llama-box/releases)。GPUStack 会自动下载并配置：

![pin-llama-box-backend-version](../assets/faq/pin-llama-box-backend-version.png)

如果使用分布式推理，应按以下方式在所有 worker 节点升级 llama-box：

从 [llama-box releases](https://github.com/gpustack/llama-box/releases) 下载最新的 llama-box 二进制文件。

你需要先停止 GPUStack，再替换二进制文件，最后重启 GPUStack。可以通过以下目录定位文件，例如：

**Docker**

```bash
docker exec -it gpustack bash
ps -ef | grep llama-box
```

**pip & Script (Legacy)**

```bash
ps -ef | grep llama-box
```

---

## 查看日志

### 如何查看 GPUStack 日志？

GPUStack 日志会包含启动状态、模型资源需求计算等信息。查看 GPUStack
日志的方法见：[故障排查](troubleshooting.md#view-gpustack-logs)。

---

### 如何开启 GPUStack 的调试模式？

你可以在不中断 GPUStack 服务的情况下临时开启调试模式。请参考：[故障排查](troubleshooting.md#configure-log-level)。

如果希望持久开启调试模式，server 与 worker 在运行 GPUStack 时都可添加 `--debug` 参数：

#### Docker

在 `docker run` 命令末尾添加 `--debug` 参数，例如：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack \
    --debug
```

#### Installer

- macOS

点击菜单栏中的 `GPUStack 图标`，选择 `Config Directory` 打开 Finder。编辑 `config.yaml` 文件，添加 `debug` 参数：

```yaml
debug: true
```

保存后再次点击 `GPUStack 图标`，进入 `Status - Restart` 以应用配置。

- Windows

右键系统托盘中的 `GPUStack 图标`，选择 `Config Directory` 打开资源管理器。编辑 `config.yaml` 文件，添加 `debug` 参数：

```yaml
debug: true
```

保存后再次右键 `GPUStack 图标`，进入 `Status - Restart` 以应用配置。

#### pip

在 `gpustack start` 命令末尾添加 `--debug` 参数：

```bash
gpustack start --debug
```

#### Script (Legacy)

- Linux

```bash
sudo vim /etc/systemd/system/gpustack.service
```

```bash
ExecStart=/root/.local/bin/gpustack start --debug
```

保存并重启 GPUStack：

```bash
sudo systemctl daemon-reload && sudo systemctl restart gpustack
```

- macOS

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
sudo vim /Library/LaunchDaemons/ai.gpustack.plist
```

```bash
  <array>
    <string>/Users/gpustack/.local/bin/gpustack</string>
    <string>start</string>
    <string>--debug</string>
  </array>
```

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

```powershell
nssm edit GPUStack
```

在 `start` 后添加参数：

```
start --debug
```

保存并重启 GPUStack：

```powershell
Restart-Service -Name "GPUStack"
```

---

### 如何查看 RPC Server 日志？

RPC Server 用于 GGUF 模型的分布式推理。如果模型启动异常或分布式推理出现问题，可在对应节点查看 RPC Server 日志：

#### Docker

默认路径如下。如果设置了 `--data-dir` 或 `--log-dir` 参数，请替换为你实际配置的路径：

```bash
docker exec -it gpustack tail -200f /var/lib/gpustack/log/rpc_server/gpu-0.log
```

每块 GPU 对应一个 RPC Server。其他 GPU 索引请按实际修改：

```bash
docker exec -it gpustack tail -200f /var/lib/gpustack/log/rpc_server/gpu-n.log
```

#### Installer

- macOS

默认路径如下。如果设置了 `--log-dir` 参数，请替换为你实际配置的路径：

```bash
tail -200f /Library/Application\ Support/GPUStack/log/rpc_server/gpu-0.log
```

- Windows

默认路径如下。如果设置了 `--log-dir` 参数，请替换为你实际配置的路径：

```powershell
Get-Content "C:\ProgramData\GPUStack\log\rpc_server\gpu-0.log" -Tail 200 -Wait
```

每块 GPU 对应一个 RPC Server。其他 GPU 索引请按实际修改：

```powershell
Get-Content "C:\ProgramData\GPUStack\log\rpc_server\gpu-n.log" -Tail 200 -Wait
```

#### "pip & Script (Legacy)"

- Linux & macOS

默认路径如下。如果设置了 `--data-dir` 或 `--log-dir` 参数，请替换为你实际配置的路径：

```bash
tail -200f /var/lib/gpustack/log/rpc_server/gpu-0.log
```

每块 GPU 对应一个 RPC Server。其他 GPU 索引请按实际修改：

```bash
tail -200f /var/lib/gpustack/log/rpc_server/gpu-n.log
```

- Windows

默认路径如下。如果设置了 `--data-dir` 或 `--log-dir` 参数，请替换为你实际配置的路径：

```powershell
Get-Content "$env:APPDATA\gpustack\log\rpc_server\gpu-0.log" -Tail 200 -Wait
```

每块 GPU 对应一个 RPC Server。其他 GPU 索引请按实际修改：

```powershell
Get-Content "$env:APPDATA\gpustack\log\rpc_server\gpu-n.log" -Tail 200 -Wait
```

### 如何查看 Ray 日志？

Ray 用于 vLLM 的分布式推理。如果模型启动异常或分布式推理出现问题，可以在服务端与对应的 worker 上查看 Ray
日志。若通过容器安装，以下路径指的是容器内目录。

默认路径如下。如果设置了 `--data-dir` 或 `--log-dir` 参数，请替换为你实际配置的路径：

Ray head 启动日志：

```bash
tail -200f /var/lib/gpustack/log/ray-head.log
```

Ray worker 启动日志：

```bash
tail -200f /var/lib/gpustack/log/ray-worker.log
```

Ray 组件日志：

```bash
/tmp/ray/session_*/logs
```

### 模型日志存放在哪里？

#### Docker

模型实例日志存放在对应 worker 容器的 `/var/lib/gpustack/log/serve/` 目录下，日志文件名为 `id.log`，其中 id 为模型实例
ID。若设置了 `--data-dir` 或 `--log-dir` 参数，则按参数指定的实际路径存放。

#### "pip & Script (Legacy)"

- Linux & macOS

模型实例日志存放在对应 worker 节点的 `/var/lib/gpustack/log/serve/` 目录下，日志文件名为 `id.log`，其中 id 为模型实例
ID。若设置了 `--data-dir` 或 `--log-dir` 参数，则按参数指定的实际路径存放。

- Windows

模型实例日志存放在对应 worker 节点的 `$env:APPDATA\gpustack\log\serve\` 目录下，日志文件名为 `id.log`，其中 id 为模型实例
ID。若设置了 `--data-dir` 或 `--log-dir` 参数，则按参数指定的实际路径存放。

---

### 如何开启后端调试模式？

#### "vLLM"

在 `Edit Model` → `Advanced` → `Environment Variables` 中添加 `VLLM_LOGGING_LEVEL=DEBUG` 环境变量，并重新创建模型实例：

![enable-vllm-debug-mode](../assets/faq/enable-vllm-debug-mode.png)

#### "llama-box"

在 `Edit Model` → `Advanced` → `Backend Parameters` 中添加 `--verbose` 参数，并重新创建模型实例：

![enable-llama-box-debug-mode](../assets/faq/enable-llama-box-debug-mode.png)

---

## 管理 Worker

### Worker 一直处于 `Unreachable` 状态怎么办？

在服务端尝试访问错误信息中显示的 URL。如果服务端运行在容器中，需要先进入服务器容器再执行命令：

```
curl http://10.10.10.1:10150/healthz
```

---

### Worker 处于 `NotReady` 状态怎么办？

查看对应 worker 上的 GPUStack 日志：[这里](troubleshooting.md#view-gpustack-logs)。若日志无异常，检查所有节点的时区与系统时间是否一致。

---

## 检测 GPU

### 为什么无法识别 Ascend NPU？

检查在容器中是否能执行 `npu-smi`：

```bash
docker exec -it gpustack bash
npu-smi info
```

若出现如下错误，表示有其他容器也挂载了 NPU 设备，且不支持共享：

```bash
dcmi model initialized failed, because the device is used. ret is -8020
```

检查主机上是否有容器挂载了 NPU 设备：

```bash
if [ $(docker ps | wc -l) -gt 1 ]; then docker ps | grep -v CONT | awk '{print $1}' | xargs docker inspect --format='{{printf "%.5s" .ID}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' | sort -k2; fi; echo ok
```

仅挂载未被其他容器占用的 NPU，使用 `--device` 显式指定：

```bash
docker run -d --name gpustack \
--restart=unless-stopped \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
... \
gpustack/gpustack:latest-npu
```

---

## 管理模型

### 如何部署模型？

#### 如何从 Hugging Face 部署模型？

要从 Hugging Face 部署模型，服务端与调度模型实例的 worker 节点需能够访问 Hugging Face，或使用镜像站点。

例如，配置 `hf-mirror.com` 镜像：

#### Docker

运行容器时添加 `HF_ENDPOINT` 环境变量，如下所示：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    -e HF_ENDPOINT=https://hf-mirror.com \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

#### Installer

点击（macOS 菜单栏或 Windows 系统托盘中的）`GPUStack 图标`，选择 `Quick Config - Environments`，添加环境变量
`HF_ENDPOINT=https://hf-mirror.com`，并重启服务生效。

#### pip

```bash
HF_ENDPOINT=https://hf-mirror.com gpustack start
```

#### Script (Legacy)

- Linux

在**所有节点**创建或编辑 `/etc/default/gpustack`，添加 `HF_ENDPOINT` 环境变量以使用 `https://hf-mirror.com` 作为 Hugging
Face 镜像：

```bash
vim /etc/default/gpustack
```

```bash
HF_ENDPOINT=https://hf-mirror.com
```

保存并重启 GPUStack：

```bash
systemctl restart gpustack
```

- macOS

在**所有节点**创建或编辑 `/etc/default/gpustack`，添加 `HF_ENDPOINT` 环境变量以使用 `https://hf-mirror.com` 作为 Hugging
Face 镜像：

```bash
vim /etc/default/gpustack
```

```bash
HF_ENDPOINT=https://hf-mirror.com
```

然后使用最初相同的配置选项重新运行安装脚本：

```bash
curl -sfL https://get.gpustack.ai | <EXISTING_INSTALL_ENV> sh -s - <EXISTING_GPUSTACK_ARGS>
```

- Windows

创建或编辑 `$env:APPDATA\gpustack\gpustack.env` 并添加如下配置：

```powershell
notepad $env:APPDATA\gpustack\gpustack.env
```

```powershell
HF_ENDPOINT=https://hf-mirror.com
```

然后使用最初相同的配置选项重新运行安装脚本：

```powershell
curl -sfL https://get.gpustack.ai | <EXISTING_INSTALL_ENV> sh -s - <EXISTING_GPUSTACK_ARGS>
```

---

#### 如何从本地路径部署模型？

从本地路径部署模型时，建议将模型文件上传到每个节点，并保持相同的绝对路径。否则应通过手动调度或标签选择，将模型实例调度到已存在该模型文件的节点。也可以挂载跨节点共享存储。

从本地路径部署 GGUF 模型时，路径必须指向 `.gguf` 文件的绝对路径；对于分片模型文件，使用第一个 `.gguf`
文件（00001）的绝对路径。若使用容器安装，必须将模型文件挂载进容器，路径应指向容器内路径，而非宿主机路径。

从本地路径部署 Safetensors 模型时，路径必须指向包含 `*.safetensors`、`config.json`
等文件的模型目录的绝对路径。若使用容器安装，必须将模型文件挂载进容器，路径应指向容器内路径，而非宿主机路径。

![deploy-model-from-local-path](../assets/faq/deploy-model-from-local-path.png)

---

### 模型一直处于 `Pending` 状态怎么办？

`Pending` 表示当前没有满足模型需求的 worker。将鼠标悬停在 `Pending` 状态上可查看原因。

首先检查 `Resources` - `Workers`，确保 worker 状态为 Ready。

然后针对不同后端：

#### "vLLM"

vLLM 默认要求所有 GPU 的可用显存大于 90%（由 `--gpu-memory-utilization` 参数控制）。需确保可分配的 GPU 显存超过
90%。注意，即使其他模型处于 `Error` 或 `Downloading` 状态，显存也已被占用。

若所有 GPU 可用显存均大于 90% 但仍显示 `Pending`，说明显存仍不足。对于 BF16 格式的 `safetensors` 模型，可按如下估算所需 GPU
显存（GB）：

```
GPU 显存（GB） = 模型权重大小（GB） * 1.2 + 2
```

若可分配显存不足 90%，但你确定模型在更低的分配比例也能运行，可调整 `--gpu-memory-utilization` 参数。例如，在 `Edit Model` →
`Advanced` → `Backend Parameters` 中添加 `--gpu-memory-utilization=0.5`，以分配 50% GPU 显存。

注意：如果模型运行后报错 `CUDA: out of memory`，则说明分配的显存不足，需要进一步调低 `--gpu-memory-utilization`
、增加资源或部署更小的模型。

模型的上下文长度也会影响所需显存。你可以通过 `--max-model-len` 参数设置更小的上下文。在 GPUStack 中，若未显式设置，该参数默认值为
8192；若在后端参数中指定，则以实际设置为准。

可根据需要调小上下文，例如 `--max-model-len=2048`。但要注意，每次推理请求的最大 token 数不能超过 `--max-model-len`
，设置过小可能导致推理被截断。

`--enforce-eager` 参数也有助于降低显存占用。但它会强制 vLLM 以 eager 模式执行，即操作调用后立即执行，而不是进行图优化的延迟执行，这可能更易调试但会降低性能。

#### "llama-box"

llama-box 使用 [GGUF Parser](https://github.com/gpustack/gguf-parser-go) 计算模型的内存需求。需确保可分配显存大于解析得出的需求。注意，即使其他模型处于
`Error` 或 `Downloading`
状态，显存也已被占用。如果不确定模型所需显存，可使用 [GGUF Parser](https://github.com/gpustack/gguf-parser-go) 进行计算。

模型的上下文长度也会影响所需显存。你可以通过 `--ctx-size` 参数设置更小的上下文。在 GPUStack 中，若未显式设置，该参数默认值为
`8192`；若在后端参数中指定，则以实际设置为准。

可在 `Edit Model` → `Advanced` → `Backend Parameters` 中按需调小，例如 `--ctx-size=2048`。但需注意，单次推理请求的最大
tokens 同时受 `--ctx-size` 与 `--parallel` 影响：
`最大 tokens = 上下文长度 / parallel`

`--parallel` 的默认值为 `4`，此时最大 tokens 为 `512`。若 token 数超过最大值，输出会被截断。

另一方面，`--parallel` 表示解码的并行序列数，可粗略理解为模型的并发请求能力。

因此，需要合理设置 `--ctx-size` 与 `--parallel`，既保证单次请求的最大 tokens 在限制内，又确保可用显存能支撑设定的上下文长度。

若显存不足，可尝试以更低配置启动：

```
--ctx-size=2048
--parallel=1
```

---

### 模型一直处于 `Scheduled` 状态怎么办？

尝试在调度到的节点上[重启 GPUStack 服务](faq.md#how-can-i-manage-the-gpustack-service)。若问题仍存，在对应 worker
的日志中排查原因：[这里](troubleshooting.md#view-gpustack-logs)。

---

### 模型一直处于 `Error` 状态怎么办？

将鼠标悬停在 `Error` 状态上查看原因。若出现 `View More` 按钮，点击进入模型日志查看具体错误信息并分析原因。

---

### 如何解决 \*.so: cannot open shared object file: No such file or directory 错误？

如果模型启动时报错无法打开某个 `.so` 文件，例如：

```bash
llama-box: error while loading shared libraries: libcudart.so.12: cannot open shared object file: No such file or directory
```

原因是 GPUStack 未识别 `LD_LIBRARY_PATH` 环境变量，可能是环境变量缺失或安装 GPUStack 时未正确配置相关工具包（如 CUDA、CANN
等）。

!!! note

该问题仅发生在非容器化环境。安装脚本已被弃用，强烈建议使用 Docker 方式安装。

检查环境变量是否已设置：

```bash
echo $LD_LIBRARY_PATH
```

若未配置，以下以 CUDA 为例。

确保可执行 `nvidia-smi` 且 NVIDIA 驱动版本为 `550` 或更高：

```bash
nvidia-smi
```

配置 CUDA 环境变量。若未安装，请安装 `CUDA 12.4` 或更高版本：

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/targets/x86_64-linux/lib
export PATH=$PATH:/usr/local/cuda/bin
echo $LD_LIBRARY_PATH
echo $PATH
```

创建或编辑 `/etc/default/gpustack`，添加 `PATH` 与 `LD_LIBRARY_PATH` 环境变量：

```bash
vim /etc/default/gpustack
```

```bash
LD_LIBRARY_PATH=......
PATH=......
```

保存并重启 GPUStack：

```bash
systemctl restart gpustack
```

---

### 使用本地路径时为什么加载模型失败？

从本地路径部署模型并出现 `failed to load model` 错误时，需要检查模型文件是否存在于模型实例调度到的节点上，且绝对路径是否正确。

对于 GGUF 模型，需要指定 `.gguf` 文件的绝对路径；若为分片模型，使用第一个 `.gguf` 文件（通常为 00001）的绝对路径。

若使用 Docker 安装，必须将模型文件挂载到容器内，且提供的路径应为容器内路径，而非宿主机路径。

![deploy-model-from-local-path](../assets/faq/deploy-model-from-local-path.png)

---

### 为什么删除模型后磁盘空间没有释放？

这是为了避免重新部署时重复下载模型。需要在 `Resources - Model Files` 中手动清理。

---

### 为什么默认每块 GPU 都有一个 llama-box 进程？

该进程是 llama-box 进行分布式推理所使用的 RPC Server。如果确定不需要 llama-box 的分布式推理，可在运行 GPUStack 时添加
`--disable-rpc-servers` 参数以禁用该服务。

---

### 后端参数

#### 如何了解后端参数的作用？

- [vLLM](https://docs.vllm.ai/en/stable/cli/serve.html)

- [MindIE](https://github.com/gpustack/gpustack/blob/main/gpustack/worker/backends/ascend_mindie.py#L103-L473)

- [llama-box](https://github.com/gpustack/llama-box?tab=readme-ov-file#usage)

---

#### 如何设置模型的上下文长度？

#### "vLLM"

GPUStack 将模型默认上下文长度设置为 8K。你可以使用 `--max-model-len` 参数自定义上下文长度，但不能超过模型的最大上下文长度：

![set-the-model-context-length-for-vllm](../assets/faq/set-the-model-context-length-for-vllm.png)

编辑后保存，并重新创建模型实例生效。

#### "MindIE"

GPUStack 将模型默认上下文长度设置为 8K。你可以使用 `--max-seq-len` 参数自定义上下文长度，但不能超过模型的最大上下文长度：

![set-the-model-context-length-for-mindie](../assets/faq/set-the-model-context-length-for-mindie.png)

编辑后保存，并重新创建模型实例生效。

#### "llama-box"

GPUStack 将模型默认上下文长度设置为 8K。你可以使用 `--ctx-size` 参数自定义上下文长度，但不能超过模型的最大上下文长度：

![set-the-model-context-length-for-llama-box](../assets/faq/set-the-model-context-length-for-llama-box.png)

编辑后保存，并重新创建模型实例生效。

---

## 使用模型

### 使用视觉语言模型

#### 如何解决 “At most 1 image(s) may be provided in one request” 错误？

这是 vLLM 的限制。你可以在 `Edit Model - Advanced - Backend Parameters` 中按需调整 `--limit-mm-per-prompt` 参数。例如，
`--limit-mm-per-prompt=image=4` 表示每次推理请求最多支持 4
张图片，详见[此处](https://docs.vllm.ai/en/latest/configuration/engine_args.html#-limit-mm-per-prompt)。

---

## 管理 GPUStack

<a id="how-can-i-manage-the-gpustack-service"></a>

### 如何管理 GPUStack 服务？

#### Docker

重启 GPUStack 容器：

```bash
docker restart gpustack
```

#### Installer

点击（macOS 菜单栏或 Windows 系统托盘中的）`GPUStack 图标`，选择 `Status - Start/Stop/Restart` 控制服务。

#### Script (Legacy)

- Linux

停止 GPUStack：

```bash
sudo systemctl stop gpustack
```

启动 GPUStack：

```bash
sudo systemctl start gpustack
```

重启 GPUStack：

```bash
sudo systemctl restart gpustack
```

- macOS

停止 GPUStack：

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
```

启动 GPUStack：

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

重启 GPUStack：

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

可以通过右键系统托盘中的 GPUStack 图标控制服务状态，或使用 PowerShell 命令。

以管理员身份运行 PowerShell（不要使用 PowerShell ISE）。

停止 GPUStack：

```powershell
Stop-Service -Name "GPUStack"
```

启动 GPUStack：

```powershell
Start-Service -Name "GPUStack"
```

重启 GPUStack：

```powershell
Restart-Service -Name "GPUStack"
```

---

### 如何在代理环境下使用 GPUStack？

#### Docker

运行 GPUStack 时传入环境变量：

```bash
docker run -e HTTP_PROXY="http://username:password@proxy-server:port" \
           -e HTTPS_PROXY="http://username:password@proxy-server:port" \
           -e ALL_PROXY="socks5://username:password@proxy-server:port" \
           -e NO_PROXY="localhost,127.0.0.1,192.168.0.0/24,172.16.0.0/16,10.0.0.0/8" \
           ...
```

#### Installer

点击（macOS 菜单栏或 Windows 系统托盘中的）`GPUStack 图标`，选择 `Quick Config - Environments`，添加所需环境变量，重启服务生效。

#### pip

- Linux & macOS

设置代理环境变量：

```bash
export HTTP_PROXY="http://username:password@proxy-server:port"
export HTTPS_PROXY="http://username:password@proxy-server:port"
export ALL_PROXY="socks5://username:password@proxy-server:port"
export NO_PROXY="localhost,127.0.0.1,192.168.0.0/24,172.16.0.0/16,10.0.0.0/8"
```

然后启动 GPUStack：

```bash
gpustack start
```

- Windows

设置代理环境变量：

```powershell
$env:HTTP_PROXY = "http://username:password@proxy-server:port"
$env:HTTPS_PROXY = "http://username:password@proxy-server:port"
$env:ALL_PROXY = "socks5://username:password@proxy-server:port"
$env:NO_PROXY = "localhost,127.0.0.1,192.168.0.0/24,172.16.0.0/16,10.0.0.0/8"
```

然后启动 GPUStack：

```powershell
gpustack start
```

#### Script (Legacy)

- Linux

创建或编辑 `/etc/default/gpustack`，添加代理配置：

```bash
vim /etc/default/gpustack
```

```bash
HTTP_PROXY="http://username:password@proxy-server:port"
HTTPS_PROXY="http://username:password@proxy-server:port"
ALL_PROXY="socks5://username:password@proxy-server:port"
NO_PROXY="localhost,127.0.0.1,192.168.0.0/24,172.16.0.0/16,10.0.0.0/8"
```

保存并重启 GPUStack：

```bash
systemctl restart gpustack
```

- macOS

创建或编辑 `/etc/default/gpustack`，添加代理配置：

```bash
vim /etc/default/gpustack
```

```bash
HTTP_PROXY="http://username:password@proxy-server:port"
HTTPS_PROXY="http://username:password@proxy-server:port"
ALL_PROXY="socks5://username:password@proxy-server:port"
NO_PROXY="localhost,127.0.0.1,192.168.0.0/24,172.16.0.0/16,10.0.0.0/8"
```

然后使用最初相同的配置选项重新运行安装脚本：

```bash
curl -sfL https://get.gpustack.ai | <EXISTING_INSTALL_ENV> sh -s - <EXISTING_GPUSTACK_ARGS>
```

- Windows

创建或编辑 `$env:APPDATA\gpustack\gpustack.env`，添加代理配置：

```powershell
notepad $env:APPDATA\gpustack\gpustack.env
```

```powershell
HTTP_PROXY="http://username:password@proxy-server:port"
HTTPS_PROXY="http://username:password@proxy-server:port"
ALL_PROXY="socks5://username:password@proxy-server:port"
NO_PROXY="localhost,127.0.0.1,192.168.0.0/24,172.16.0.0/16,10.0.0.0/8"
```

然后使用最初相同的配置选项重新运行安装脚本：

```powershell
curl -sfL https://get.gpustack.ai | <EXISTING_INSTALL_ENV> sh -s - <EXISTING_GPUSTACK_ARGS>
```