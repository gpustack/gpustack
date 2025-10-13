# Apple Metal 安装

## 支持的设备

- [x] Apple Metal（M 系列芯片）

## 支持的平台

| 操作系统 | 版本                     | 架构  | 支持的安装方式                                                                                                                                                           |
| ------ | ------------------------ | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| macOS  | 14 Sonoma<br>15 Sequoia | ARM64 | [桌面安装程序](desktop-installer.md)（推荐）<br>[pip 安装](#pip-installation)<br>[安装脚本](#install-scripts)（已弃用） |

## 支持的后端

- [x] llama-box
- [x] vox-box（CPU 后端）

## 桌面安装程序

请参考桌面安装程序 [文档](desktop-installer.md#desktop-installer)

<a id="pip-installation"></a>

## pip 安装

### 前置条件

- Python 3.10 ~ 3.12

检查 Python 版本：

```bash
python -V
```

### 安装 GPUStack

运行以下命令安装 GPUStack：

```bash
pip install gpustack
```

如需音频模型支持，运行：

```bash
pip install "gpustack[audio]"
```

验证安装，运行：

```bash
gpustack version
```

### 运行 GPUStack

运行以下命令启动 GPUStack 服务器及内置 worker：

```bash
sudo gpustack start
```

如果启动日志正常，在浏览器打开 `http://your_host_ip` 访问 GPUStack UI。使用用户名 `admin` 和默认密码登录。你可以运行以下命令获取默认设置的密码：

```bash
cat /var/lib/gpustack/initial_admin_password
```

如果你通过 `--data-dir` 参数指定了数据目录，`initial_admin_password` 文件将位于该目录中。

默认情况下，GPUStack 使用 `/var/lib/gpustack` 作为数据目录，因此需要 `sudo` 或相应权限。你也可以通过运行以下命令设置自定义数据目录：

```
gpustack start --data-dir mypath
```

可参考 [CLI 参考](../cli-reference/start.md) 了解可用的 CLI 标志。

### （可选）添加 Worker

要向 GPUStack 集群添加 worker，需要指定服务器 URL 和认证令牌。

在 GPUStack 服务器节点运行以下命令以获取用于添加 worker 的令牌：

```bash
cat /var/lib/gpustack/token
```

如果你通过 `--data-dir` 参数指定了数据目录，`token` 文件将位于该目录中。

要启动一个 GPUStack worker 并将其注册到 GPUStack 服务器，请在 worker 节点上运行以下命令。请将 URL 和令牌替换为你的实际值：

```bash
sudo gpustack start --server-url http://your_gpustack_url --token your_gpustack_token
```

### 以 Launchd 服务方式运行 GPUStack

推荐的方式是将 GPUStack 作为开机启动服务运行。例如，使用 launchd：

在 `/Library/LaunchDaemons/ai.gpustack.plist` 创建服务文件：

```bash
sudo tee /Library/LaunchDaemons/ai.gpustack.plist > /dev/null <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.gpustack</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(command -v gpustack)</string>
    <string>start</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>EnableTransactions</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/var/log/gpustack.log</string>
  <key>StandardErrorPath</key>
  <string>/var/log/gpustack.log</string>
</dict>
</plist>
EOF
```

然后启动 GPUStack：

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

查看服务状态：

```bash
sudo launchctl print system/ai.gpustack
```

并确认 GPUStack 启动日志正常：

```bash
tail -200f /var/log/gpustack.log
```

<a id="install-scripts"></a>

## 安装脚本（已弃用）

!!! warning

    从 0.7 版本起，安装脚本方式已被弃用。我们建议在 Linux 上使用 Docker，在 macOS 或 Windows 上使用桌面安装程序。

GPUStack 提供了脚本，可将其安装为服务，默认端口为 80。

### 前置条件

- Python 3.10 ~ 3.12

检查 Python 版本：

```bash
python -V
```

### 运行 GPUStack

```bash
curl -sfL https://get.gpustack.ai | sh -s -
```

如需音频模型支持，运行：

```bash
curl -sfL https://get.gpustack.ai | INSTALL_SKIP_BUILD_DEPENDENCIES=0 sh -s -
```

若需在运行脚本时配置其他环境变量和启动参数，请参阅[安装脚本](installation-script.md)。

安装完成后，确认 GPUStack 启动日志正常：

```bash
tail -200f /var/log/gpustack.log
```

如果启动日志正常，在浏览器打开 `http://your_host_ip` 访问 GPUStack UI。使用用户名 `admin` 和默认密码登录。你可以运行以下命令获取默认设置的密码：

```bash
cat /var/lib/gpustack/initial_admin_password
```

如果你通过 `--data-dir` 参数指定了数据目录，`initial_admin_password` 文件将位于该目录中。

### （可选）添加 Worker

要向 GPUStack 集群添加 worker，在工作节点安装 GPUStack 时需要指定服务器 URL 和认证令牌。

在 GPUStack 服务器节点运行以下命令以获取用于添加 worker 的令牌：

```bash
cat /var/lib/gpustack/token
```

如果你通过 `--data-dir` 参数指定了数据目录，`token` 文件将位于该目录中。

要安装 GPUStack 并以 worker 身份启动，同时注册到 GPUStack 服务器，请在 worker 节点运行以下命令。请将 URL 和令牌替换为你的实际值：

```bash
curl -sfL https://get.gpustack.ai | sh -s - --server-url http://your_gpustack_url --token your_gpustack_token
```

如需音频模型支持，运行：

```bash
curl -sfL https://get.gpustack.ai | INSTALL_SKIP_BUILD_DEPENDENCIES=0 sh -s - --server-url http://your_gpustack_url --token your_gpustack_token
```

安装完成后，确认 GPUStack 启动日志正常：

```bash
tail -200f /var/log/gpustack.log
```