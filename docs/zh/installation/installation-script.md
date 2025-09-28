# 安装脚本（已弃用）

!!! note

    自 0.7 版本起，安装脚本方式已被弃用。我们建议在 Linux 上使用 Docker，在 macOS 或 Windows 上使用[桌面安装程序](https://gpustack.ai/)。

## Linux 和 macOS

你可以使用位于 `https://get.gpustack.ai` 的安装脚本，在基于 systemd 和 launchd 的系统上将 GPUStack 安装为服务。

运行该脚本时，你可以设置额外的环境变量和 CLI 标志。以下是使用不同配置运行安装脚本的示例：

```bash
# 使用内置 worker 运行服务器。
curl -sfL https://get.gpustack.ai | sh -s -

# 使用非默认端口运行服务器。
curl -sfL https://get.gpustack.ai | sh -s - --port 8080

# 使用自定义数据路径运行服务器。
curl -sfL https://get.gpustack.ai | sh -s - --data-dir /data/gpustack-data

# 不启用内置 worker 运行服务器。
curl -sfL https://get.gpustack.ai | sh -s - --disable-worker

# 启用 TLS 运行服务器。
curl -sfL https://get.gpustack.ai | sh -s - --ssl-keyfile /path/to/keyfile --ssl-certfile /path/to/certfile

# 使用外部 PostgreSQL 数据库运行服务器。
curl -sfL https://get.gpustack.ai | sh -s - --database-url "postgresql://username:password@host:port/database_name"

# 以指定 IP 运行 worker。
curl -sfL https://get.gpustack.ai | sh -s - --server-url http://myserver --token mytoken --worker-ip 192.168.1.100

# 使用自定义 PyPI 镜像安装。
curl -sfL https://get.gpustack.ai | INSTALL_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple sh -s -

# 安装来自 pypi.org 以外来源的自定义 wheel 包。
curl -sfL https://get.gpustack.ai | INSTALL_PACKAGE_SPEC=https://repo.mycompany.com/my-gpustack.whl sh -s -

# 安装带有额外音频依赖的特定版本。
curl -sfL https://get.gpustack.ai | INSTALL_PACKAGE_SPEC=gpustack[audio]==0.6.0 sh -s -
```

## Windows

你可以使用位于 `https://get.gpustack.ai` 的安装脚本，通过 Windows 服务管理器将 GPUStack 安装为服务。

运行该脚本时，你可以设置额外的环境变量和 CLI 标志。以下是使用不同配置运行安装脚本的示例：

```powershell
# 使用内置 worker 运行服务器。
Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content

# 使用非默认端口运行服务器。
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --port 8080"

# 使用自定义数据路径运行服务器。
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --data-dir 'D:\gpustack-data'"

# 不启用内置 worker 运行服务器。
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --disable-worker"

# 启用 TLS 运行服务器。
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --ssl-keyfile 'C:\path\to\keyfile' --ssl-certfile 'C:\path\to\certfile'"

# 使用外部 PostgreSQL 数据库运行服务器。
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --database-url 'postgresql://username:password@host:port/database_name'"

# 以指定 IP 运行 worker。
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --server-url 'http://myserver' --token 'mytoken' --worker-ip '192.168.1.100'"

# 以自定义预留资源运行 worker。
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --server-url 'http://myserver' --token 'mytoken' --system-reserved '{""ram"":5, ""vram"":5}'"

# 使用自定义 PyPI 镜像安装。
$env:INSTALL_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content

# 安装来自 pypi.org 以外来源的自定义 wheel 包。
$env:INSTALL_PACKAGE_SPEC = "https://repo.mycompany.com/my-gpustack.whl"
Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content

# 安装带有额外音频依赖的特定版本。
$env:INSTALL_PACKAGE_SPEC = "gpustack[audio]==0.6.0"
Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content
```

!!! warning

    请避免使用 PowerShell ISE，它与安装脚本不兼容。

## 安装脚本可用的环境变量

| 名称                              | 默认值                               | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| --------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `INSTALL_INDEX_URL`               | （空）                               | Python 软件包索引的基础 URL。                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `INSTALL_PACKAGE_SPEC`            | `gpustack[all]` 或 `gpustack[audio]` | 要安装的包规范。安装脚本会根据平台自动决定。它支持 PYPI 包名、URL 和本地路径。详情参见 [pip install 文档](https://pip.pypa.io/en/stable/cli/pip_install/#pip-install)。<ul><li>`gpustack[all]`：包含所有推理后端：llama-box、vllm、vox-box。</li><li>`gpustack[vllm]`：包含推理后端：llama-box、vllm。</li><li>`gpustack[audio]`：包含推理后端：llama-box、vox-box。</li></ul> |
| `INSTALL_SKIP_POST_CHECK`         | （空）                               | 如果设为 1，安装脚本将跳过安装后的检查。                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `INSTALL_SKIP_BUILD_DEPENDENCIES` | `1`                                  | 如果设为 1，将跳过构建依赖安装。                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `INSTALL_SKIP_IOGPU_WIRED_LIMIT`  | （空）                               | 如果设为 1，将跳过在 macOS 上设置 GPU 有线内存限制。                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `INSTALL_IOGPU_WIRED_LIMIT_MB`    | （空）                               | 用于设置在 macOS 上 GPU 可分配的有线内存上限（MB）。                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

## 为 GPUStack 服务设置环境变量

你可以在以下位置的环境文件中为 GPUStack 服务设置环境变量：

- Linux 和 macOS：`/etc/default/gpustack`
- Windows：`$env:APPDATA\gpustack\gpustack.env`

以下是该文件内容的示例：

```bash
HF_TOKEN="your_hf_token"
HF_ENDPOINT="https://your_hf_endpoint"
```

!!!note

    与 Systemd 不同，Launchd 和 Windows 服务并不原生支持从文件读取环境变量。通过环境文件进行配置是由安装脚本实现的：它会读取该文件并将变量应用到服务配置中。在 Windows 和 macOS 上修改环境文件后，你需要重新运行安装脚本以将更改应用到 GPUStack 服务。

## 可用的 CLI 标志

安装脚本后追加的 CLI 标志会直接作为 `gpustack start` 命令的参数传递。你可以参阅[CLI 参考](../cli-reference/start.md)了解详情。