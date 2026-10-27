# 升级集群部署 { #upgrade-a-cluster-deployment }

GPUStack 将 Worker 组织为[集群](../user-guide/cluster-management.md)。本页介绍在 GPUStack Server 升级后，如何升级集群中的 Worker 部署。

!!! note

    1. 先升级 GPUStack **Server**，见[通过 Docker 升级](docker.md)或[通过 Helm 升级](helm.md)，再升级集群部署。

    2. 将 Worker 升级为与 GPUStack Server **相同的版本**。请勿将 `main`（dev）或候选发布（rc）Worker 镜像与稳定版 GPUStack Server 混用。

升级流程与集群最初的部署方式一致，参见[集群管理](../user-guide/cluster-management.md)：对于 Docker 集群，使用新镜像重新运行部署命令；对于 Kubernetes 集群，使用新镜像重新应用清单。

## Docker 集群 { #docker-cluster }

Docker 集群中的 Worker 由集群 `Add Worker` 步骤中的 `docker run` 命令启动，以容器形式运行。升级每个 Worker 时，请拉取新镜像、移除旧容器，并使用与之前**相同的参数和数据卷**重新创建：

```bash
docker pull gpustack/gpustack:vx.y.z   # match the server version

docker stop gpustack-worker
docker rm gpustack-worker

docker run -d --name gpustack-worker \
  ... \
  gpustack/gpustack:vx.y.z \
  --server-url http://your_gpustack_server_url \
  --token your_worker_token \
  --advertise-address your_worker_ip
```

!!! tip

    可以从 GPUStack UI 获取目标版本的最新 `docker run` 命令：在 `Clusters` 页面打开集群的 `Add Worker` 步骤，复制生成的 `Run command`。

## Kubernetes 集群 { #kubernetes-cluster }

Kubernetes 集群中的 Worker 部署为 DaemonSet，由 GPUStack Operator 管理，来自集群注册时应用的清单。如需升级，请重新应用为新版本生成的清单：

1. 进入 `Clusters` 页面，找到要升级的 Kubernetes 集群。
2. 单击操作列的省略号按钮，然后选择 `Register Cluster`。
3. 按照步骤进入 `Run command`，复制生成的命令。
4. 在已安装 `kubectl` 且 `kubeconfig` 配置为目标集群的环境中运行该命令。此操作重新应用更新后的清单，Operator 将 Worker DaemonSet 滚动更新到新版本。

Worker Pod 达到 `Ready` 状态后，集群即运行新版本。

## 云提供方集群 { #cloud-provider-cluster }

对于通过云提供方创建的集群，例如 DigitalOcean，每个 Worker 由 cloud-init 初始化，将 Worker 配置写入 `/var/lib/gpustack/config.yaml`，并通过 `/opt/gpustack-run-worker.sh` 启动 `gpustack-worker` 容器。如需升级现有节点，请通过 SSH 登录，使用相同的配置和数据卷，以新镜像重新创建容器：

```bash
# SSH into the worker node, then:
NEW_IMAGE=gpustack/gpustack:vx.y.z   # match the server version

docker pull "$NEW_IMAGE"

docker rm -f gpustack-worker

docker run -d --name gpustack-worker \
  -e "GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME=gpustack-worker" \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  -v /var/lib/gpustack:/var/lib/gpustack \
  -v /var/run/docker.sock:/var/run/docker.sock \
  "$NEW_IMAGE" \
  --config-file=/var/lib/gpustack/config.yaml
```

!!! note

    为使**新创建**的节点，例如 Worker Pool 扩容新增的节点，也使用新版本，请同时在界面中更新集群镜像配置。参见[集群管理](../user-guide/cluster-management.md#operating-worker-pools)。
