# 从旧版脚本安装迁移

如果你之前通过旧版安装脚本安装了 GPUStack，请按照以下说明迁移到受支持的方法。

!!! note

    在执行迁移之前，强烈建议备份你的数据库。对于默认安装，先停止 GPUStack 服务器，然后备份位于 `/var/lib/gpustack/database.db` 的文件。

## Linux 迁移

### 第 1 步：找到你现有的数据目录

找到旧版安装使用的数据目录路径。默认路径为：

```bash
/var/lib/gpustack
```

下文将其记作 `${your-data-dir}`。

### 第 2 步：通过 Docker 重新安装 GPUStack

如果你使用的是 Nvidia GPU，运行以下 Docker 命令迁移你的 GPUStack 服务器，将卷挂载路径替换为你的数据目录位置。

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v ${your-data-dir}:/var/lib/gpustack \
    gpustack/gpustack
```

这将通过 Docker 启动 GPUStack，同时保留你现有的数据。

对于工作节点和其他 GPU 硬件平台，请参考[安装文档](installation/installation-requirements.md)中的命令。

## macOS / Windows 迁移

通过[桌面安装程序](installation/desktop-installer.md#download-installer)下载并安装新版 GPUStack。

!!!note

    安装器升级目前仅测试了从 v0.6.2 升级到 v0.7.0。理论上可以从 v0.6.2 之前的版本直接升级到安装器 v0.7.0，但建议先升级到 v0.6.2，再使用安装器进行迁移升级。

1. 启动 GPUStack，系统托盘会出现图标。如果检测到已安装旧版本的 GPUStack，将显示 `To Upgrade` 状态。

   ![darwin 待升级状态](../assets/desktop-installer/to-upgrade-darwin.png)

1. 要升级并迁移到新版本 GPUStack，可在 `Status` 的子菜单中点击 `Start`。
1. 原有配置会根据所运行的操作系统迁移到相应位置。详细配置可参见[桌面配置](user-guide/desktop-setup.md#configuration)

   - macOS
     - 通过启动参数设置的配置将合并到单一配置文件 `~/Library/Application Support/GPUStackHelper/config.yaml`。
     - 通过环境变量设置的配置将迁移到 `launchd` 的 plist 配置 `~/Library/Application Support/GPUStackHelper/ai.gpustack.plist`。
     - GPUStack 数据将移动到新的数据位置 `/Library/Application Support/GPUStack`。
   - Windows
     - 通过启动参数设置的配置将合并到单一配置文件 `C:\Users\<Name>\AppData\Roaming\GPUStackHelper\config.yaml`。
     - 服务配置（如环境变量）不会被合并，因为将复用系统服务 `GPUStack`。
     - GPUStack 数据将移动到新的数据位置 `C:\ProgramData\GPUStack`。