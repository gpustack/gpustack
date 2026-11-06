# 通过 Helm 安装 { #installation-via-helm }

从 v2.2.0 起，可以使用官方 Helm Chart 在 Kubernetes 上部署 GPUStack，并使用集群内 [Higress](https://higress.io/) API 网关。

!!! note

    使用 Higress 在 Kubernetes 上部署 GPUStack 目前属于**实验性**功能。继续之前，请查看[限制](#limitations)。

## 前提条件 { #prerequisites }

| 组件 | 版本 |
| ---------- | ---------- |
| Helm       | >= v3.18.4 |
| Kubernetes | >= v1.30.0 |
| GPUStack   | >= v2.2.0  |

此外：

- 集群必须配置默认 `StorageClass`，用于 GPUStack Server 数据卷，k3s 中默认为 `local-path`。也可以设置 `server.dataVolume.hostPath`，使用主机路径卷代替 PVC。
- 对于 GPU Worker，确保节点已安装相应 GPU 驱动和容器工具包，见[安装要求](./requirements.md)；同时安装 [Node Feature Discovery（NFD）](https://kubernetes-sigs.github.io/node-feature-discovery/)，为 GPU 节点添加标签。

## 限制 { #limitations }

- GPUStack Server 部署为 `StatefulSet`，目前**不支持**多副本。
- 默认使用内置的内嵌 `PostgreSQL` 数据库。生产环境建议通过 `server.externalDatabaseURL` 指定外部数据库。
- Higress 插件由与 GPUStack 一同安装的专用 `gpustack/higress-plugins` Deployment 提供。Higress 网关重启时会从此服务下载插件；如果该服务不可用，网关会等待插件可访问后才能启动。
- 内置 `higress-core` 子 Chart 会将 Higress 部署为集群的 Ingress Controller。如果已有其他 Ingress Controller，请设置 `higress-core.enabled=false`，并将 `gateway.ingressClassname` 指向已有 Higress 实例。

## 安装 k3s（可选） { #install-k3s-optional }

以下步骤以 k3s 作为 Kubernetes 发行版示例。其他符合 CNCF 标准的发行版，如 RKE2、基于 kubeadm 的集群或托管云 Kubernetes，只要满足上述版本要求，也可以使用。

安装 k3s 时禁用 Traefik，因为将使用 Higress 作为 Ingress Controller。高可用 k3s 集群的配置见 [k3s 文档](https://docs.k3s.io/datastore/ha)。

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.30.11+k3s1 INSTALL_K3S_EXEC="--disable=traefik" sh -
```

验证安装：

```bash
kubectl version
```

## 使用 Helm 安装 GPUStack { #install-gpustack-with-helm }

Chart 作为 OCI 产物发布在 Docker Hub 的 `oci://registry-1.docker.io/gpustack/gpustack-chart`。Chart 版本与 GPUStack 发布版本对应，例如 Chart `2.2.0` 包含 GPUStack `v2.2.0`。

安装最新稳定版本。省略 `--version` 时，Helm 会自动选择最新稳定 Chart 版本，跳过开发版和预发布版：

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace
```

如需固定版本，请附加 `--version <chart-version>`，Chart 版本与 GPUStack 发布版本一致，例如 `2.2.0`：

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --version <chart-version>
```

默认启用 `higress-core` 子 Chart，并与 GPUStack 一同部署。如果集群中已安装 Higress，请禁用内置 Higress，并将 GPUStack 指向现有实例：

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set higress-core.enabled=false \
  --set gateway.ingressClassname=<your-higress-ingressclass>
```

如需自定义参数，请在安装时使用 `--set key=value` 或 `-f your-values.yaml`。见下文 [Chart 参数](#chart-parameters)。

### 从克隆的仓库安装 { #installing-from-a-cloned-repository }

也可以克隆仓库，从本地 Chart 目录安装，适用于离线环境或需要修改 Chart 的场景：

```bash
git clone https://github.com/gpustack/gpustack.git
cd gpustack/charts
helm dependency update ./gpustack-chart
helm install gpustack ./gpustack-chart \
  --namespace gpustack-system --create-namespace
```

### 单独安装 Higress { #installing-higress-separately }

如果设置 `higress-core.enabled=false`，请在部署 GPUStack 前安装兼容的 Higress 实例：

```bash
# Add the Higress Helm repository
helm repo add higress.io https://higress.io/helm-charts

# Install higress-core (match the version pinned by the GPUStack chart)
helm install higress higress.io/higress-core \
  --namespace higress-system --create-namespace \
  --version 2.1.9
```

确认 IngressClass 可用，并且名称与 `gateway.ingressClassname` 一致：

```bash
kubectl get ingressclass higress
# NAME      CONTROLLER                      PARAMETERS   AGE
# higress   higress.io/higress-controller   <none>       3m46s
```

Higress 自定义配置见 [Higress 文档](https://higress.cn/en/docs/latest/ops/deploy-by-helm)。

## 访问 GPUStack { #accessing-gpustack }

等待 GPUStack Server Pod 就绪：

```bash
kubectl get pods -n gpustack-system -w
```

获取初始管理员密码：

```bash
kubectl exec -it -n gpustack-system gpustack-server-0 -- \
  cat /var/lib/gpustack/initial_admin_password
```

如果未设置 `server.ingress.hostname`，请从 Ingress 获取 GPUStack UI 地址：

```bash
kubectl get ingress -n gpustack-system gpustack \
  -o jsonpath="{.status.loadBalancer.ingress[0].ip}"
```

在浏览器中打开该地址，使用用户名 `admin` 和上面获取的密码登录。

## 常用配置 { #common-configuration }

### 启用 Worker DaemonSet { #enabling-worker-daemonsets }

Worker DaemonSet 默认禁用（`worker.enabled=false`）。启用后，Chart 会生成一个 CPU Worker DaemonSet（`<release>-worker`），并为 `worker.gpuVendors` 列出的每个 GPU 厂商生成一个 DaemonSet，名称为 `<release>-worker-<vendor>`。

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set worker.enabled=true \
  --set 'worker.gpuVendors={nvidia}'
```

支持的 `worker.gpuVendors` 值：`nvidia`、`mthreads`、`amd`、`ascend`、`hygon`、`metax`、`iluvatar`、`cambricon`、`thead`。

!!! note

    每个 GPU DaemonSet 都会自动获得 Node Feature Discovery 上报的 PCI 设备存在性 `nodeSelector` 标签，例如 NVIDIA 对应 `feature.node.kubernetes.io/pci-10de.present: "true"`。集群中**必须安装 NFD**，否则没有节点包含所需标签，所有 Worker Pod 都会停留在 `Pending`。

只要列出了至少一个 GPU 厂商，即 `worker.gpuVendors` 非空，每个 Worker Pod 还会获得必需的 `podAntiAffinity`（topologyKey=hostname），防止两个 Worker 运行在同一节点上，避免 `hostNetwork: true` 导致端口冲突。

CPU Worker DaemonSet 通过 `nodeSelector` 条件 `feature.gpustack.ai/acceleratable: "false"`，覆盖未匹配任何 GPU 厂商的节点。设置 `worker.cpuEnabled=false` 可让这些节点不运行 Worker，通常适用于控制平面与 GPU 节点位于同一集群、且不应运行 Worker 的场景：

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set worker.enabled=true \
  --set worker.cpuEnabled=false \
  --set 'worker.gpuVendors={nvidia}'
```

此时，`worker.gpuVendors` 必须指定至少一个受支持的厂商；否则 Release 不会生成任何 Worker DaemonSet，安装会被拒绝。无论是否启用 CPU Worker，GPU DaemonSet 都保留 `-<vendor>` 后缀，因此关闭 CPU Worker 不会将任何 GPU DaemonSet 重命名为 `<release>-worker`。

也可以在安装后，通过界面的 **Clusters** 和 **Workers** 页面添加 GPU 集群和 Worker。

### 使用外部数据库 { #using-an-external-database }

GPUStack 默认使用内嵌 PostgreSQL 数据库。如需使用外部 PostgreSQL 或 MySQL 数据库，请设置 `server.externalDatabaseURL`：

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set server.externalDatabaseURL="postgresql://user:password@host:port/dbname"
```

### 使用自定义证书启用 HTTPS { #enabling-https-with-a-custom-certificate }

通过 `server.ingress.tls` 提供 PEM 格式的证书和密钥内容。两者均设置后，Ingress 协议将变为 HTTPS：

```yaml
# values.yaml
server:
  ingress:
    hostname: gpustack.example.com
    tls:
      cert: |-
        -----BEGIN CERTIFICATE-----
        MIID...
        -----END CERTIFICATE-----
      key: |-
        -----BEGIN PRIVATE KEY-----
        MIIE...
        -----END PRIVATE KEY-----
```

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  -f values.yaml
```

### 从私有镜像仓库拉取镜像 { #pulling-images-from-a-private-registry }

如需从私有镜像仓库拉取所有镜像，包括 GPUStack Server/Worker、higress-plugins 和内置 higress-core gateway/controller/pilot，请覆盖 `global.hub`。Helm 的全局值传播机制可让一个设置应用于所有镜像：

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set global.hub=myregistry.example.com
```

`global.hub` 还会作为 `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY` 传入 GPUStack Server，确保 vLLM、llama.cpp 等推理引擎镜像从相同仓库拉取。

为提供拉取凭据，Chart 可以创建名为 `gpustack-image-pull-secret` 的 `docker-registry` Secret，并关联到所有 Pod：

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set imagePullSecret.credentials.registry=registry.example.com \
  --set imagePullSecret.credentials.username=myuser \
  --set imagePullSecret.credentials.password=mypassword
```

如需引用自己预先创建的 Secret，请替换 `global.imagePullSecrets`：

```yaml
global:
  imagePullSecrets:
    - name: my-existing-secret
```

## Chart 参数 { #chart-parameters }

下表列出了最常用的参数。完整权威列表见 Chart 的 [`values.yaml`](https://github.com/gpustack/gpustack/blob/main/charts/gpustack-chart/values.yaml) 和 [README](https://github.com/gpustack/gpustack/blob/main/charts/gpustack-chart/README.md)。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `debug` | `false` | 启用调试模式。 |
| `registrationToken` | `null` | Worker 注册令牌；为 `null` 时生成随机令牌并复用。 |
| `clusterDomain` | `cluster.local` | Kubernetes 集群服务域名后缀。 |
| `global.hub` | `docker.io` | 容器镜像仓库主机；使用私有仓库时覆盖此值。 |
| `global.imagePullSecrets` | `[gpustack-image-pull-secret]` | 关联到所有 Pod 并传播到子 Chart 的镜像拉取 Secret。 |
| `global.nodeSelector` | `{}` | 所有组件默认使用的 nodeSelector；可由组件级设置替换。 |
| `image.repository` | `gpustack/gpustack` | 包含命名空间的镜像仓库；最终引用为 `{global.hub}/{repository}:{tag}`。 |
| `image.tag` | `null` | 镜像标签；默认为 Chart 的 `appVersion`。 |
| `image.pullPolicy` | `IfNotPresent` | 镜像拉取策略。 |
| `imagePullSecret.credentials.registry` | `docker.io` | Chart 创建镜像拉取 Secret 时使用的仓库主机。 |
| `imagePullSecret.credentials.username` | `null` | 仓库用户名；与密码同时设置时创建 Secret。 |
| `imagePullSecret.credentials.password` | `null` | 仓库密码；与用户名同时设置时创建 Secret。 |
| `server.ingress.hostname` | `null` | Server 的 Ingress 主机名。 |
| `server.ingress.tls.cert` | `null` | Ingress TLS 证书（PEM）；与密钥同时设置时启用 HTTPS。 |
| `server.ingress.tls.key` | `null` | Ingress TLS 私钥（PEM）。 |
| `server.externalDatabaseURL` | `null` | 外部数据库连接字符串（PostgreSQL 或 MySQL）。 |
| `server.dataVolume.hostPath` | `null` | Server 数据卷的主机路径；使用 hostPath 代替 PVC。 |
| `server.dataVolume.size` | `100Gi` | Server 数据卷大小（PVC）。 |
| `server.apiPort` | `30080` | API 服务端口。 |
| `server.metricsPort` | `10161` | Server 指标端口。 |
| `server.environmentConfig` | `{}` | GPUStack Server 的额外环境变量。 |
| `server.nodeSelector` | `{}` | Server Pod 的 nodeSelector；非空时替换 `global.nodeSelector`。 |
| `gateway.ingressClassname` | `higress` | Higress IngressClass 名称；找到对应项时启用集群内网关模式。 |
| `higress-core.enabled` | `true` | 部署内置 Higress 网关；若已安装，则禁用。 |
| `worker.enabled` | `false` | 生成 Worker DaemonSet。 |
| `worker.gpuVendors` | `[nvidia]` | GPU 厂商；每个厂商对应一个 DaemonSet，此外还有 CPU DaemonSet。 |
| `worker.cpuEnabled` | `true` | 生成 CPU Worker DaemonSet；设为 `false` 时必须指定 GPU 厂商。 |
| `worker.nodeSelector` | `{}` | Worker 的基础 nodeSelector；非空时替换 `global.nodeSelector`。 |
| `worker.port` | `10150` | Worker 服务端口。 |
| `worker.metricsPort` | `10151` | Worker 指标端口。 |
| `worker.dataDir` | `/var/lib/gpustack` | 在每个 Worker Pod 内挂载到 `/var/lib/gpustack` 的主机路径。 |
| `worker.environmentConfig` | `{}` | GPUStack Worker 的额外环境变量。 |

## 卸载 { #uninstallation }

Release 部署的不仅是 Server：`gpustack-operator` 子 Chart 还会部署 Kueue 和 Node Feature Discovery 控制器，Operator 创建的自定义资源带有 finalizer。`helm uninstall` 会将控制器与其他资源一并移除，因此必须**先**在控制器仍运行时删除自定义资源，否则这些对象会一直停留在 `Terminating`，`gpustack-system` 命名空间也无法完成删除。请按以下顺序操作：

1. 在**所有**控制器仍运行时清理工作负载。Operator 会清除其 `Instance` 对象上的 finalizer；而只要已准入的 `Workload` 仍占用 `ClusterQueue`，Kueue 就不会释放该队列，因此必须在停止任何控制器之前完成这些清理。`Instance` 属于命名空间级资源，位于各租户的命名空间中，因此请使用 `-A` 列出所有对象。从 Server 删除模型，或直接删除其实例；每个实例的 Pod 会随实例终止，Workload 也会随 Instance 删除。不要一次性删除命名空间内的所有 Pod，否则 Operator、Kueue 和 NFD 控制器也会被关闭，而后续清理需要这些控制器完成：

    ```bash
    kubectl delete instances.worker.gpustack.ai --all -A
    kubectl get instances.worker.gpustack.ai -A; kubectl get workloads.kueue.x-k8s.io -A   # both empty before moving on
    ```

    如果 Kueue 与其他应用共用，请勿在此处或第 3 步使用 `--all`，只删除 GPUStack 的对象。Operator 在 Kueue 中创建的所有对象名称均以 `gpustack` 开头，例如 `clusterqueue/gpustack--…`、`admissioncheck/gpustack-node-devices`。集群级资源只需按名称选择；命名空间级资源还需要附带命名空间，因为 `-o name` 不包含命名空间：

    ```bash
    kubectl get clusterqueues.kueue.x-k8s.io -o name | grep '/gpustack' \
      | xargs -I {} kubectl delete {}                # clusterqueues, resourceflavors, admissionchecks
    kubectl get localqueues.kueue.x-k8s.io -A -o jsonpath='{range .items[*]}{.metadata.namespace} {.metadata.name}{"\n"}{end}' \
      | awk '$2 ~ /^gpustack/' | while read ns name; do \
        kubectl -n "$ns" delete localqueues.kueue.x-k8s.io "$name"; done
    ```

2. 停止 Operator 自身的控制器，但保留 Release，让 Kueue 控制器继续运行，以完成下一步的删除。只要 `gpustack-operator-worker` 和 `gpustack-operator-device-manager-*` DaemonSet 仍在运行，它们就会在您删除 `LocalQueue`、`ClusterQueue` 和 `gpustack-node-devices` AdmissionCheck 后立即重新创建这些资源，使资源数量始终无法归零：

    ```bash
    kubectl -n gpustack-system scale deploy/gpustack-operator-worker --replicas=0
    kubectl -n gpustack-system get ds -l app.kubernetes.io/name=gpustack-operator-device-manager -o name | xargs -I {} \
      kubectl -n gpustack-system patch {} -p '{"spec":{"template":{"spec":{"nodeSelector":{"gpustack.ai/uninstall-pause":"true"}}}}}'
    ```

3. 按依赖顺序删除 Kueue 资源：先删除 `LocalQueue`，再删除 `ClusterQueue`、`AdmissionCheck`，最后删除 `ResourceFlavor`。当使用某对象的资源仍存在时，Kueue 会保留该对象上的 `kueue.x-k8s.io/resource-in-use` finalizer：只有所有 ClusterQueue 都已删除，`gpustack-node-devices` AdmissionCheck 才能完成删除；每个 ClusterQueue 也必须等待其 LocalQueue 和已准入的 Workload 删除。确认每类资源均已清空后，再处理下一类。在共享集群中，请对每条命令应用第 1 步的 `gpustack` 名称前缀范围限制，不要使用 `--all`：

    ```bash
    kubectl delete localqueues.kueue.x-k8s.io --all -A
    kubectl delete clusterqueues.kueue.x-k8s.io --all
    kubectl delete admissionchecks.kueue.x-k8s.io --all
    kubectl delete resourceflavors.kueue.x-k8s.io --all
    ```

4. 卸载 Release。不要传入 `--wait`：Node Feature Discovery 的 post-delete prune hook 会阻塞删除等待，超过其 `--timeout`；为退出等待而终止 Helm，会留下没有归属的 prune RBAC 资源。第 5 步会手动移除这些资源，即使未使用 `--wait`，也应执行第 5 步：

    ```bash
    helm uninstall gpustack -n gpustack-system
    ```

5. 移除 `helm uninstall` 未删除的资源。Helm 从不删除 CRD，Operator 注册的准入 Webhook 和 APIService 也不属于任何 Release。这两类资源都会因 Operator Service 已不存在而拒绝请求；在删除 Webhook 之前，它们还会拒绝下文移除 finalizer 的操作：

    ```bash
    kubectl delete mutatingwebhookconfiguration gpustack-worker-mutation --ignore-not-found
    kubectl delete validatingwebhookconfiguration gpustack-worker-validation --ignore-not-found
    kubectl delete apiservice v1.gpustack.ai v1.worker.gpustack.ai --ignore-not-found
    kubectl get instances.worker.gpustack.ai -A -o jsonpath='{range .items[*]}{.metadata.namespace} {.metadata.name}{"\n"}{end}' \
      | while read ns name; do [ -n "$name" ] && kubectl -n "$ns" patch instances.worker.gpustack.ai "$name" \
        --type=json -p='[{"op":"remove","path":"/metadata/finalizers"}]' 2>/dev/null; done
    kubectl get instancetypes.worker.gpustack.ai -o name | xargs -I {} \
      kubectl patch {} --type=json -p='[{"op":"remove","path":"/metadata/finalizers"}]' 2>/dev/null
    kubectl delete crd devices.worker.gpustack.ai instances.worker.gpustack.ai \
      instancetypes.worker.gpustack.ai --ignore-not-found
    kubectl delete clusterrole node-feature-discovery-prune --ignore-not-found
    kubectl delete clusterrolebinding node-feature-discovery-prune --ignore-not-found
    ```

    `worker.gpustack.ai` 组始终由 GPUStack 独占，没有其他组件使用，因此可以直接删除该组的 CRD。

6. 删除 Release 遗留的 CRD，**仅限此 Release 安装的 CRD**。删除 CRD 会删除整个集群中该类型的所有对象；如果集群在安装 GPUStack 前已运行 Kueue、Node Feature Discovery 或 Higress（参见上文 `higress-core.enabled`），且仍有其他组件使用它们，请保留其 CRD，并跳过对应组的清理。请查询此 Release 拥有的资源，不要凭猜测判断：

    ```bash
    kubectl get crd -o json | jq -r '.items[]
      | select(((.metadata.annotations // {})["meta.helm.sh/release-name"] // "") == "gpustack")
      | .metadata.name'
    ```

    在默认安装中，该命令会列出 Kueue 的 CRD；如果 Chart 还安装了 NFD 和 Higress，也会列出它们的 CRD：

    ```bash
    kubectl get crd -o name | grep 'kueue\.x-k8s\.io' | xargs -I {} kubectl delete {} --ignore-not-found
    kubectl delete crd nodefeatures.nfd.k8s-sigs.io nodefeaturerules.nfd.k8s-sigs.io \
      nodefeaturegroups.nfd.k8s-sigs.io --ignore-not-found
    kubectl delete crd envoyfilters.networking.istio.io http2rpcs.networking.higress.io \
      mcpbridges.networking.higress.io wasmplugins.extensions.higress.io --ignore-not-found
    ```

    集群提供方自身的 CRD 不属于 GPUStack，不应删除。例如，k3s 集群中的 `*.k3s.cattle.io` 和 `helm.cattle.io` 属于集群自身。

7. *（可选）* 删除命名空间，同时移除其中的 Higress 对象、引导 ConfigMap、ServiceAccount 和注册令牌 Secret。请了解同时会被删除的数据：命名空间中还包含 Server 的 PVC（`gpustack-data-dir-gpustack-server-0`，命名规则为 `<volumeClaimTemplate>-<statefulset>-<ordinal>`），其中保存了 Server 的持久化数据，即包含用户、模型和集群信息的内嵌 PostgreSQL 数据库。在 `Delete` 回收策略下（`local-path` 的默认策略），底层卷也会被删除。如需保留数据以便重新安装，请保留命名空间；否则可删除命名空间，如果回收策略未自动移除 PVC，请先删除 PVC：

    ```bash
    kubectl delete pvc gpustack-data-dir-gpustack-server-0 -n gpustack-system
    kubectl delete namespace gpustack-system
    ```
