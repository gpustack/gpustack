# 升级

您可以通过 Docker 升级 GPUStack，或手动安装所需版本的 GPUStack Python 包来升级。

!!! note

    1. 升级时，请先升级 GPUStack 服务器，再升级工作节点（workers）。
    2. 请勿（DO NOT）从/到 main（dev）版本或候选发布（rc）版本进行升级，因为它们可能包含破坏性变更。如果您想尝试 main 或 rc 版本，请进行全新安装。

!!! note

    在进行升级之前，强烈建议先备份数据库。对于默认安装，请停止 GPUStack 服务器，并备份位于 `/var/lib/gpustack/database.db` 的文件。

## Docker 升级

如果您是通过 Docker 安装的 GPUStack，可通过拉取带有所需版本标签的 Docker 镜像来升级到新版本。

例如：

```bash
docker pull gpustack/gpustack:vX.Y.Z
```

然后使用新镜像重启 GPUStack 服务。

## pip 升级

如果您使用 pip 手动安装了 GPUStack，请使用常见的 `pip` 工作流程进行升级。

例如，将 GPUStack 升级到最新版本：

```bash
pip install --upgrade gpustack
```

## 从旧版脚本升级

如果您是使用安装脚本安装的 GPUStack，请按照[迁移指南](migration.md)进行升级并保留现有数据。