# 桌面端设置

!!! note

    自 v0.7.0 起支持

你可以通过安装程序在 macOS 和 Windows 上安装 GPUStack。安装过程中会同时安装用于管理 GPUStack 的系统托盘应用程序 GPUStackHelper。请按照[桌面端安装程序](../installation/desktop-installer.md)指南完成安装。

## GPUStackHelper

GPUStackHelper 提供以下用于配置和管理 GPUStack 的功能。

![托盘菜单](../../assets/desktop-installer/to-upgrade-darwin.png)

- 状态 - 显示 GPUStack 服务的当前状态。状态子菜单支持启动、停止和重启操作。
- Web 控制台 - 如果服务正在运行，在默认浏览器中打开 GPUStack 门户。
- 随系统启动 - 配置 GPUStack 是否在系统启动时自动运行。
- 快速配置 - 打开[快速配置对话框](#quick-config-dialog)。
- 配置目录 - 打开包含 GPUStack 配置文件 `config.yaml` 的目录。快速配置中不可用的配置可以在此处手动修改。
- 复制令牌 - 复制工作节点注册所需的令牌。
- 显示日志 - 打开控制台或记事本窗口查看 GPUStack 日志。
- 关于 - 显示版本信息。

<a id="configuration"></a>

## 配置

可以通过 GPUStackHelper 以多种方式配置 GPUStack。推荐使用配置文件，而不是命令行参数或环境变量。部分 GPUStack 参数是固定的，由 Helper 管理，不应手动修改。

例如，在 macOS 的 GPUStack `launchd` 配置中（`/Library/LaunchDaemons/ai.gpustack.plist`）：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>EnableTransactions</key>
    <true/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>/Library/Application Support/GPUStack/root</string>
    </dict>
    <key>KeepAlive</key>
    <true/>
    <key>Label</key>
    <string>ai.gpustack</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Applications/GPUStack.app/Contents/MacOS/gpustack</string>
        <string>start</string>
        <string>--config-file=/Library/Application Support/GPUStack/config.yaml</string>
        <string>--data-dir=/Library/Application Support/GPUStack</string>
    </array>
    <key>RunAtLoad</key>
    <false/>
    <key>StandardErrorPath</key>
    <string>/var/log/gpustack.log</string>
    <key>StandardOutPath</key>
    <string>/var/log/gpustack.log</string>
</dict>
</plist>
```

- ProgramArguments 指定了 GPUStack 的固定参数，如 `--config-file` 和 `--data-dir`。在 Windows 上，NSSM 服务中也有类似的固定配置。GPUStack 安装并启动后，你可以使用 `nssm dump gpustack` 查看。
- EnvironmentVariables 中的 `HOME` 变量用于避免将 root 用户的主目录作为默认缓存路径。此设置仅适用于 macOS。

<a id="quick-config-dialog"></a>

### 快速配置对话框

![通用](../../assets/desktop-installer/quickconfig-general.png)

“通用”选项卡包含常用配置。更多细节请参见[配置文件](../cli-reference/start.md#config-file)文档。

- 服务器角色 - 参见 `disable_worker` 配置。

    - 全部（同时作为服务器和工作节点）
    - 工作节点
    - 仅服务器

- 服务器 URL - 参见 `server_url` 配置。
- 令牌 - 参见 `token` 配置。
- 端口 - 参见 `port` 配置。

![环境变量](../../assets/desktop-installer/quickconfig-env-var.png)

“环境变量”选项卡用于为 GPUStack 服务设置环境变量。常见环境变量已在键名选项中列出：

- HF_ENDPOINT - Hugging Face Hub 端点，例如 `https://hf-mirror.com`。
- HF_TOKEN - Hugging Face 令牌。
- HTTP_PROXY、HTTPS_PROXY 和 NO_PROXY - 与代理相关的环境变量。

你也可以手动输入自定义变量的键和值。

!!! note

    GPUStack 支持通过以 `GPUSTACK_` 开头的环境变量进行配置。但仍推荐使用配置文件。比如，如果通过环境变量设置了 `GPUSTACK_PORT`，`Web 控制台` 功能将无法识别该端口，从而导致浏览器打开默认端口或配置文件中指定的端口。

## 状态与启动

GPUStack 可能显示以下状态之一：

- 已停止
- 运行中
- 待升级 - 表示应当从脚本式安装升级 GPUStack。
- 待重启|运行中 - 表示由于配置更改需要重启 GPUStack。

### 平台说明

- 在 Windows 上，GPUStackHelper 需要 UAC（用户帐户控制）确认才能启动，因为它需要权限来操作 GPUStack 服务。  
- 在 macOS 上，GPUStackHelper 以当前登录用户的身份在后台运行，在启动/停止/重启 GPUStack 服务时会提示获取 root 权限，如下图所示。

![权限提示](../../assets/desktop-installer/prompt-root-privileges.png){width=30%}