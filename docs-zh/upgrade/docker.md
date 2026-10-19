# 通过 Docker 升级 { #upgrade-via-docker }

对于使用 Docker 安装的 GPUStack，可以通过拉取新镜像并重新创建容器进行升级。

以下升级说明仅适用于 GPUStack v2.0 及更高版本。

对于 v0.7 之前的安装，请参阅[迁移指南](../migration.md)。

!!! note

    1. 升级时，请先升级 GPUStack Server，再升级 Worker。

    2. 请**不要**从 main（dev）版本或候选发布（rc）版本升级，也不要升级到这些版本，因为它们可能包含不兼容更改。如需试用 main 或 rc 版本，请全新安装。

!!! warning

    **先备份：** 升级之前，强烈建议备份数据库。

    对于使用内嵌 PostgreSQL 数据库的默认安装，请停止 GPUStack Server，并备份容器内以下位置的 PostgreSQL 数据库目录：

    ```
    /var/lib/gpustack/postgresql/data
    ```

    如果使用外部数据库，请改用数据库提供方的备份流程。

先升级 **GPUStack Server**，再升级 **Worker**。GPUStack Server 的升级方式为：拉取新镜像，可以指定版本标签或使用 `latest` 标签；移除旧容器；使用与之前**相同的参数和数据卷**，通过新镜像启动新容器。

## 升级 GPUStack Server { #upgrade-the-server }

```bash
docker pull gpustack/gpustack:latest  # or: docker pull gpustack/gpustack:vx.y.z

docker stop gpustack
docker rm gpustack

docker run -d --name gpustack \
  ... \
  gpustack/gpustack:latest
  ...
```

## 升级 Worker { #upgrade-the-workers }

GPUStack Server 使用新版本启动并正常运行后，升级各集群中的 Worker。Docker、Kubernetes 和云集群的操作流程见[升级集群部署](cluster.md)。

## 通过 Docker Compose 升级 { #upgrade-via-docker-compose }

如果使用从仓库克隆的 Docker Compose 部署，请切换到新发布标签，并使用更新后的镜像重新创建容器：

```bash
cd gpustack

# Fetch the tags and check out the target stable release
git fetch --tags
git checkout <new-tag>   # e.g. vx.y.z

cd docker-compose

# Pull the new image and recreate the containers
sudo docker compose -f docker-compose.server.yaml pull
sudo docker compose -f docker-compose.server.yaml up -d
```
