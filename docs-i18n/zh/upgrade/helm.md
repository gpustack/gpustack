# 通过 Helm 升级 { #upgrade-via-helm }

对于基于 Kubernetes 安装的 GPUStack，可以使用 `helm upgrade` 升级。

以下升级说明仅适用于 GPUStack v2.2.0 及更高版本，这是首个支持通过 Helm 部署的版本。

!!! note

    使用 Higress 在 Kubernetes 上部署和升级 GPUStack 目前属于**实验性**功能。

!!! warning

    **先备份：** 升级之前，强烈建议备份数据库。

    如果使用内置的内嵌 PostgreSQL 数据库，其数据位于 GPUStack Server 的 PVC 中。PVC 名称为 `gpustack-data-dir-<release-name>-server-0`，例如默认 `gpustack` Release 对应 `gpustack-data-dir-gpustack-server-0`。该 PVC 在 GPUStack Server Pod 内挂载到 `/var/lib/gpustack`，内嵌 PostgreSQL 数据目录位于挂载路径下的 `postgresql/data`。升级前请备份。

    如果使用外部数据库（`server.externalDatabaseURL`），请改用数据库提供方的备份流程。

## 升级 Release { #upgrade-the-release }

始终通过 `--version <chart-version>` 固定目标版本，以确保升级到已知且可重复部署的版本。Chart 版本与 GPUStack 发布版本对应，例如 `2.2.0`。可以通过传入安装时使用的相同 `-f values.yaml` 和/或 `--set` 参数复用配置，也可以使用 `--reuse-values` 保留之前应用的值：

```bash
helm upgrade gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system \
  --version <chart-version> \
  -f values.yaml
```

!!! warning

    省略 `--version` 时，Helm 会自动选择最新稳定 Chart 版本。升级时应避免这样做，因为您无法控制最终升级到哪个版本。始终使用 `--version` 指定目标发布版本。

## GPUStack Server 和 Worker 的升级顺序 { #server-and-worker-ordering }

GPUStack Server 部署为 `StatefulSet`，Chart 管理的 Worker 部署为 `DaemonSet`，因此 Release 升级时，两者都会滚动更新到新镜像。由于 GPUStack Server 和 Worker 属于**同一个 Release**，单次 `helm upgrade` 无法保证“先 GPUStack Server，后 Worker”的顺序。如果环境要求严格顺序，请先等待 GPUStack Server Pod 就绪，再允许 Worker Pod 滚动更新：

```bash
kubectl rollout status statefulset/<release-name>-server -n gpustack-system
```

通过 GPUStack 集群在 Chart 之外注册的 Worker，例如通过界面添加的 Docker 或 Kubernetes 集群，不受此 Helm Release 管理，必须按照[升级集群部署](cluster.md)中的步骤升级。
