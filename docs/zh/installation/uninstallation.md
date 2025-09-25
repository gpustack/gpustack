# 卸载

## Docker

如果你使用 Docker 安装了 GPUStack，以下是用于卸载 GPUStack 的示例命令；你可以根据你的环境进行修改：

```bash
# 移除容器
docker rm -f gpustack
# 移除数据卷
docker volume rm gpustack-data
```

## pip

如果你通过 pip 安装了 GPUStack，以下是用于卸载 GPUStack 的示例命令；你可以根据你的环境进行修改：

```bash
# 停止并移除服务
systemctl stop gpustack.service
rm -f /etc/systemd/system/gpustack.service
systemctl daemon-reload
# 卸载 CLI
pip uninstall gpustack
# 删除数据目录
rm -rf /var/lib/gpustack
```

## 脚本

!!! warning

    卸载脚本会删除本地数据存储（sqlite）中的数据、配置、模型缓存，以及所有脚本和 CLI 工具。它不会从外部数据存储中删除任何数据。

如果你使用安装脚本安装了 GPUStack，安装过程中会生成一个用于卸载 GPUStack 的脚本。

运行以下命令以卸载 GPUStack：

=== "Linux"

    ```bash
    sudo /var/lib/gpustack/uninstall.sh
    ```

=== "macOS"

    ```bash
    sudo /var/lib/gpustack/uninstall.sh
    ```

=== "Windows"

    ```powershell
    Set-ExecutionPolicy Bypass -Scope Process -Force; & "$env:APPDATA\gpustack\uninstall.ps1"
    ```