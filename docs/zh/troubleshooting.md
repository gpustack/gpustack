# 故障排除

<a id="view-gpustack-logs"></a>

## 查看 GPUStack 日志

如果你使用 Docker 或桌面安装程序安装了 GPUStack，在默认配置下可使用以下命令查看 GPUStack 日志：

=== "Linux"

    ```bash
    docker logs -f gpustack
    ```

=== "macOS"

    ```bash
    tail -200f /var/log/gpustack.log
    ```

=== "Windows"

    ```powershell
    Get-Content "C:\ProgramData\GPUStack\log\gpustack.log" -Tail 200 -Wait
    ```

<a id="configure-log-level"></a>

## 配置日志级别

通过设置 `--debug` 参数，可为 `gpustack start` 启用 DEBUG 日志级别。

你可以在服务器节点上运行以下命令，在运行时配置 GPUStack 服务器的日志级别：

=== "Linux & macOS"

    ```bash
    curl -X PUT http://localhost/debug/log_level -d "debug"
    ```

    同样适用于 GPUStack 工作节点：

    ```bash
    curl -X PUT http://localhost:10150/debug/log_level -d "debug"
    ```

=== "Windows"

    ```powershell
    curl.exe -X PUT http://localhost/debug/log_level -d "debug"
    ```

    同样适用于 GPUStack 工作节点：

    ```powershell
    curl.exe -X PUT http://localhost:10150/debug/log_level -d "debug"
    ```

可用的日志级别包括：`trace`，`debug`，`info`，`warning`，`error`，`critical`。

## 重置管理员密码

=== "Linux"

    如果你忘记了管理员密码，可在**服务器容器**内运行以下命令进行重置：

    ```bash
    gpustack reset-admin-password
    ```

    如果默认端口已更改，请使用 `--server-url` 参数指定 GPUStack 的 URL。该命令必须在服务器本机上运行，并通过 `localhost` 访问：

    ```bash
    gpustack reset-admin-password --server-url http://localhost:9090
    ```

=== "macOS"

    如果你忘记了管理员密码，可在**服务器节点**上运行以下命令进行重置：

    ```bash
    /Applications/GPUStack.app/Contents/MacOS/gpustack reset-admin-password
    ```

    如果默认端口已更改，请使用 `--server-url` 参数指定 GPUStack 的 URL。该命令必须在服务器本机上运行，并通过 `localhost` 访问：

    ```bash
    /Applications/GPUStack.app/Contents/MacOS/gpustack reset-admin-password --server-url http://localhost:9090
    ```

=== "Windows"

    如果你忘记了管理员密码，可在**服务器节点**上运行以下命令进行重置：

    ```powershell
    & "C:\Program Files\GPUStack\gpustack.exe" reset-admin-password
    ```

    如果默认端口已更改，请使用 `--server-url` 参数指定 GPUStack 的 URL。该命令必须在服务器本机上运行，并通过 `localhost` 访问：

    ```powershell
    & "C:\Program Files\GPUStack\gpustack.exe" reset-admin-password --server-url http://localhost:9090
    ```