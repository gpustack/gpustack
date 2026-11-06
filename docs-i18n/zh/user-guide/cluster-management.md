---
translation_status: translated
---

# 集群管理 { #cluster-management }

GPUStack 支持按集群管理 Worker，并提供多种集群类型。可以通过 `DigitalOcean` 等 `Cloud Provider` 创建集群，也可以创建自托管集群并使用 `Docker` 运行命令添加 Worker。此外，还可以将自托管 `Kubernetes` 集群中的全部节点注册为 GPUStack Worker。

## 创建集群 { #create-cluster }

1. 进入 `Clusters` 页面。
2. 单击 `Add Cluster` 按钮。
3. 选择集群提供商。`Self-Host` 提供商支持 `Docker` 和 `Kubernetes`，`Cloud Provider` 支持 `DigitalOcean` 和 `SHUIHUA FUTURE`。
4. 根据提供商的不同，需要在 `Base Configuration` 和 `Add Worker` 步骤中设置不同选项。
5. Base Configuration 中的 `Advanced` 集群设置允许通过 `Worker Configuration YAML` 预配置 Worker 选项。

### 创建 Docker 集群 { #create-docker-cluster }

1. 在 `Basic Configuration` 步骤中，`Name` 为必填字段，`Description` 为可选字段。
2. 单击 `Save`。
3. 在 `Add Worker` 步骤中，通过 `docker run` 命令添加 Worker 前，需要设置一些选项并执行验证。
4. `Select the GPU vendor`。已测试的厂商包括 `Nvidia`、`AMD` 和昇腾（`Ascend`）。实验性支持的厂商包括海光（`Hygon`）、摩尔线程（`Moore Threads`）、天数智芯（`Iluvatar`）、寒武纪（`Cambricon`）和沐曦（`Metax`）。选择厂商后单击 `Next`。
5. `Check Environment`。页面提供 shell 命令，用于验证环境是否已具备添加 Worker 的条件。复制脚本并在你的环境中运行。脚本返回 OK 后单击 `Next`。
6. 为要添加的 Worker 执行 `Specify arguments`。提供以下参数，然后单击 `Next`：
   - `Specify the worker IP`，或让 Worker `Auto-detect the Worker IP`。确保 GPUStack Server 可以访问 Worker IP。
   - 为 Worker 容器指定 `Additional Volume Mount`。此挂载路径可以用于复用已有模型文件。

7. 执行 `Run command`，创建并启动 Worker 容器。复制 Bash 脚本并在你的环境中运行。

集群创建后也可以添加 Worker。

1. 进入 `Clusters` 页面。
2. 找到要添加 Worker 的集群。
3. 单击操作列中的省略号按钮，然后选择 `Add Worker`。
4. 选择添加 Worker 的选项。按照上文从 `Select the GPU vendor` 到 `Run command` 的相同步骤操作。

### 注册 Kubernetes 集群 { #register-kubernetes-cluster }

1. 在 `Basic Configuration` 步骤中，`Name` 为必填字段，`Description` 为可选字段。
2. 单击 `Save`。
3. `Select the GPU vendor`。已测试的厂商包括 `Nvidia`、`AMD` 和昇腾（`Ascend`）。实验性支持的厂商包括海光（`Hygon`）、摩尔线程（`Moore Threads`）、天数智芯（`Iluvatar`）、寒武纪（`Cambricon`）和沐曦（`Metax`）。对于 `Kubernetes` 集群，可以选择多个 GPU 厂商；对于纯 CPU 集群，也可以不选。每个所选厂商都会生成一个 Worker DaemonSet，每个 DaemonSet 的 `nodeSelector` 会在生成清单时根据厂商的 PCI 存在性标签推导。选择厂商后单击 `Next`。
4. `Check environment`。页面提供 shell 命令，用于验证环境是否已具备添加 Worker 的条件。复制脚本并在你的环境中运行。脚本返回 OK 后单击 `Next`。
5. 执行 `Run command` 应用 Worker 清单。复制 Bash 脚本，并在已安装 `kubectl` 且已配置 `kubeconfig` 的环境中运行。

集群创建后也可以注册 Kubernetes。

1. 进入 `Clusters` 页面。
2. 找到要注册 Kubernetes 集群的集群记录。
3. 单击操作列中的省略号按钮，然后选择 `Register Cluster`。
4. 选择注册集群的选项。按照上文从 `Select the GPU vendor` 到 `Run command` 的相同步骤操作。

#### Kubernetes 集群选项 { #kubernetes-cluster-options }

创建或编辑 `Kubernetes` 集群时，`Basic Configuration` 步骤除了 `Name` 和 `Description` 外，还提供以下选项：

- `Cluster Type`（必填）——选择集群用途：
    - `Model Service`——用于 LLM 推理和 API 服务，例如暴露模型 API 和基于 token 的服务。
    - `GPU Service`——用于按需 GPU 计算，例如交互式开发、训练任务或自定义环境。

`Advanced` 设置提供以下 Kubernetes 部署选项：

- `Namespace`——生成的集群清单使用的 Kubernetes 命名空间。留空时使用 `gpustack-system`。
- `Volume Mounts`——挂载到每个 Worker Pod 的额外卷。为每个挂载指定 `Volume Name`、`Container Path` 和 `Read Only` 标记，然后选择 `Source Type`：
    - `Host Path`——节点上的路径，并指定 `Path Type`（例如 `Directory`、`Directory (create if not exists)`、`File`、`Socket`、`Character Device`、`Block Device`）。
    - `Persistent Volume Claim (PVC)`——已有的 `PVC Name`，可选只读。
    - `ConfigMap`——`ConfigMap Name`，可标记为可选。
- `Image Credentials`——从私有镜像仓库拉取 GPUStack 镜像时使用的镜像拉取 Secret。为每个条目指定 `Registry`、`Username` 和 `Password`。
- `Node Selector`——将此集群的 GPUStack 工作负载限制到标签匹配的节点。它作为 chart 的 `global.nodeSelector` 传递，各组件会回退使用此值，因此会作用于 Worker DaemonSet、Operator，以及支持此设置的 Operator 版本所部署的组件（Kueue、Node Feature Discovery 控制平面、CSI 控制器）。两类工作负载有意排除：一类是必须覆盖所服务节点的组件（Node Feature Discovery 标签组件、CSI 节点插件），因为限制它们会导致其他节点没有标签，包括 Worker 自身选择所依据的标签；另一类是 chart 安装钩子，它们在 release 存在之前运行，无法等待 release 自身应用的标签。Worker DaemonSet 会同时保留 GPU 厂商标签和这些标签，节点必须同时匹配。
- `Default Container Registry`——解析此集群 GPUStack 镜像所使用的默认镜像仓库。未设置时回退到 GPUStack Server 默认值（占位值为 `docker.io`）。
- `Operator Image`——GPUStack Operator 容器镜像覆盖值。留空时使用 GPUStack Server 默认值。
- `GPU Service Static Access Address`——仅当 `Cluster Type` 为 `GPU Service` 时显示。Operator 用于访问此集群中 GPU 实例的静态地址（例如 LoadBalancer VIP），可选。
- `Worker Configuration YAML`——参见下方 [Worker 配置 YAML](#worker-configuration-yaml)。

#### 命名空间与 Pod Security Admission { #namespaces-and-pod-security-admission }

GPUStack 会创建自己管理的命名空间：`gpustack-system`（或 `Namespace` 设置的名称），以及使用此集群的每个组织对应的 `gpustack-<org>`；并为每个命名空间设置 `privileged` 级别的 [Pod Security Admission](https://kubernetes.io/docs/concepts/security/pod-security-admission/) 标签：

```yaml
pod-security.kubernetes.io/enforce: privileged
pod-security.kubernetes.io/audit: privileged
pod-security.kubernetes.io/warn: privileged
```

**为什么需要它。** Pod Security Admission 是 Kubernetes 内置功能，在 Pod **创建时**执行：不符合相应级别的 Pod 会直接被拒绝，而不是停留在 Pending。模型、缓存服务和基准测试 Pod 确实需要 `baseline` 和 `restricted` 级别禁止的能力：

| 需求 | 需要它的功能 |
| --- | --- |
| 主机网络 | RDMA 将 GID 绑定到网卡地址 |
| hostPort | 分离部署的一对实例向对端提供的侧通道 |
| 主机 IPC | 通过 CUDA-IPC 共享 KV 缓存缓冲区 |
| 设备挂载 | 加速器设备本身 |

预填充/解码部署同时需要这四项，因此缺少标签时，它会最先且最严重地失败。

**为什么设置三个键。** 如果 `warn` 和 `audit` 保持集群默认值，每次创建 Pod 仍会返回警告并写入审计注解。对于本就应获得豁免的命名空间，这些信息会产生噪声，掩盖其他位置真正需要关注的警告。

**影响范围。** `privileged` 表示这些命名空间中的每个 Pod 都免于 Pod Security Admission 限制。因此它们应由 GPUStack 创建并独占管理，不应与自己的工作负载共享，也不应调度其他工作负载进入其中。这些标签也不会影响 Kyverno、Gatekeeper 或 OPA；它们是独立的准入 webhook，如果其策略禁止主机网络或设备挂载，仍会拒绝这些 Pod。集群运行此类策略时，也请为 GPUStack 命名空间配置允许规则。

**检查更严格级别会拒绝什么。** 针对自己运行中的工作负载执行以下命令，不会更改任何内容：

```bash
kubectl label --dry-run=server --overwrite ns <namespace> \
  pod-security.kubernetes.io/enforce=baseline
```

API 服务器会根据该级别评估命名空间内现有 Pod，对每个可能被拒绝的 Pod 给出警告，并指出违反的控制项。

#### Chart values { #chart-values }

注册 `Kubernetes` 集群会在其中安装 GPUStack Helm chart，上述选项会转换为此 chart 的 values。对于这些选项未覆盖的配置，可以通过 `helmValues` 直接设置同一 chart：键直接采用 chart 原有键，因此 chart 及其子 chart 提供的任何功能都能配置，无需等待此处增加专用选项。

UI 尚未提供对应字段，请通过 API 设置。`PUT` 会替换整个集群，因此应先读取集群，再将 `helmValues` 添加到 `k8sOptions` 下后提交回去：

```bash
# Read the cluster.
curl -sS -H "Authorization: Bearer <api-key>" \
  http://<server>/v2/clusters/<id> > cluster.json

# Add the values. This one keeps the chart from installing the S3 CSI driver,
# for a cluster that already provides its own storage.
jq '.k8sOptions.helmValues = {
      "gpustack-operator": {"csi-driver-s3": {"enabled": false}}
    }' cluster.json > updated.json

# Send it back.
curl -sS -X PUT -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" --data @updated.json \
  http://<server>/v2/clusters/<id>
```

这些 values 会按键逐层、深度优先地合并覆盖从集群推导的配置；列表采用替换而非扩展，与 Helm 本身一致。各键的说明参见 chart 文档：GPUStack chart 的 `values.yaml`；对于 `gpustack-operator` 下的配置，参见 [GPUStack Operator chart](https://github.com/gpustack/gpustack-operator)。

某些路径不会合并，而是会被拒绝：这些路径由 GPUStack Server 根据此集群的注册信息推导，用于确保 release 与注册所针对的集群一致，包括 Worker 上报位置、运行镜像、镜像仓库及携带令牌的 Secret。API 会指出被拒绝的路径，`helmValues` 字段说明中包含当前列表；本页不重复列举，避免二者不一致。

!!! note

    仅保存集群不会自行更改 Kubernetes 中的内容。请重新执行 `Register Cluster` 并应用生成的清单；集群内 Job 会比较清单要求的配置与 release 安装时的配置，仅在不同的情况下升级。

!!! note

    如果升级改变了 release 部署的内容，例如为由纯 CPU 节点承载控制平面的集群关闭 CPU Worker DaemonSet（在 `helmValues` 中设置 `worker.cpuEnabled=false`），或取消选择集群不再使用的 GPU 厂商，升级流程会自行移除不再生成的 Worker DaemonSet。`kubectl apply` 不会删除清单中省略的对象，因此升级 Job 会删除 release 不再生成的每个 `<release>-worker[-<vendor>]` DaemonSet，无论它是否曾归某个 Helm revision 管理。

    在此清理机制出现之前遗留的对象也会采用相同方式移除：应用一次当前清单，Job 就会清理，即使此后配置没有变化。只有完全无法运行 Job 的集群，才需要手动操作：

    ```bash
    kubectl delete daemonset/gpustack-worker -n gpustack-system
    kubectl delete daemonset/gpustack-worker-<vendor> -n gpustack-system
    ```

    如果只是想升级版本，请保持 `Register Cluster` 参数（GPU 厂商、选项）与上次应用的清单完全一致。其他任何变动都属于配置更改，而非单纯升级。

!!! warning

    Kueue 和 Node Feature Discovery 不是可选组件。Operator 依赖它们构建调度链，并在启动时等待其 CRD；如果禁用集群尚未自行提供的组件，Operator 将无法启动。关闭当前 release 安装的组件会将其移除；Kueue 的 CRD 来自其 chart，所有 `Workload` 和 `ClusterQueue` 也会随 CRD 一并移除。

#### 卸载 Kubernetes 集群 { #uninstalling-a-kubernetes-cluster }

卸载包括对注册时安装的 release 执行 `helm uninstall`，以及清理从未由 Helm 管理的对象。能否顺利完成取决于一个准备步骤：Operator 组件树（Kueue、Node Feature Discovery、CSI 驱动、设备管理器）部署的控制器，其自定义资源带有 finalizer；卸载会移除控制器，所以必须先删除这些资源，否则其所在命名空间（`gpustack-system`、`gpustack-default`）会一直处于 `Terminating`。按以下顺序操作：

1. 先在 GPUStack Server 上移除集群运行的内容：删除或迁移部署到该集群的模型，再删除集群记录，确保 GPUStack Server 停止向即将消失的 Worker 调度。
2. **在控制器仍运行时**删除自定义资源，并等待清理完成。请查询此集群 release 实际管理的资源，不要自行猜测：

    ```bash
    kubectl get crd -o json | jq -r '.items[]
      | select(((.metadata.annotations // {})["meta.helm.sh/release-name"] // "")
        | . == "gpustack" or . == "gpustack-kueue"
          or . == "gpustack-node-feature-discovery"
          or . == "gpustack-csi-driver-nfs" or . == "gpustack-csi-driver-s3"
          or . == "gpustack-operator-device-manager")
      | .metadata.name'
    ```

    对列出的每个 CRD，删除其全部对象，并等待没有对象剩余：

    ```bash
    kubectl delete <crd-name> --all -A          # e.g. clusterqueues.kueue.x-k8s.io
    kubectl get <crd-name> -A                   # empty before you move on
    ```

3. 卸载 release。主 release 为 `gpustack`；在 chart 接管其对象之前注册的集群中，还存在五个 `gpustack-*` 应用 release。对不存在的 release 执行 `helm uninstall` 会报错，因此请先检查：

    ```bash
    for release in gpustack gpustack-kueue gpustack-node-feature-discovery \
                   gpustack-csi-driver-nfs gpustack-csi-driver-s3 \
                   gpustack-operator-device-manager; do
      kubectl get secret -n gpustack-system \
        --selector "owner=helm,name=${release}" >/dev/null 2>&1 || continue
      helm uninstall "${release}" -n gpustack-system --timeout 10m
    done
    ```

    此处不要传入 `--wait`。使用 `--wait` 时，卸载会运行 Node Feature Discovery 的 **post-delete prune hook**，创建该钩子的 `node-feature-discovery-prune` ServiceAccount 和 ClusterRole，然后阻塞在删除等待中，且等待会超过 `--timeout`（使用 `helm --debug` 已验证：超时后数分钟仍停留在 `waiting for resources to be deleted count=1`，刚创建的 ServiceAccount 也仍存在）。为退出阻塞而终止 Helm，会导致 prune RBAC 无归属地遗留，因此步骤 4 会手动移除这些对象。没有 `--wait` 时，卸载会及时返回，钩子资源与其他资源一起清理，对象同样会被移除；步骤 4 会清理从未由 Helm 管理的残留对象。

4. 移除 `helm uninstall` 未处理的内容。Helm 从不删除 CRD，Operator 还会注册三类没有任何 release 管理的集群级对象：

    - **Operator 自身的 CRD**——`worker.gpustack.ai` 下的 `devices`、`instances` 和 `instancetypes`。它们由 Operator 在启动时创建，而非由 chart 创建，因此没有 `meta.helm.sh/release-name` 注解，也不会出现在步骤 2 的发现查询中。`instancetype` 对象保留 `gpustack.ai/controlled` finalizer；请在删除 CRD 前或删除过程中移除它（见下文），否则 CRD 会一直处于删除状态，该组 API 也不会消失。
    - **Operator 的准入 webhook**——`gpustack-worker-mutation`（Mutating）和 `gpustack-worker-validation`（Validating）。它们将请求转发到 Operator 的 Service，失败时拒绝请求，因此在删除之前，对 `worker.gpustack.ai` 对象的每次写入（包括下方移除 finalizer 的操作）都会被拒绝，并报告 `service "gpustack-operator-worker" not found`。
    - **Operator 的 APIService**——`v1.gpustack.ai` 和 `v1.worker.gpustack.ai`，同样由 Operator 的 Service 支撑。它们移除前，卡在删除中的命名空间会针对这些组报告 `stale GroupVersion discovery`，且无法完成删除。

    ```bash
    kubectl delete mutatingwebhookconfiguration gpustack-worker-mutation --ignore-not-found
    kubectl delete validatingwebhookconfiguration gpustack-worker-validation --ignore-not-found
    kubectl delete apiservice v1.gpustack.ai v1.worker.gpustack.ai --ignore-not-found
    kubectl delete crd devices.worker.gpustack.ai instances.worker.gpustack.ai \
      instancetypes.worker.gpustack.ai --ignore-not-found
    ```

    Node Feature Discovery 的 prune RBAC 也可能在 `helm uninstall` 后残留。`ClusterRole/node-feature-discovery-prune` 及其绑定属于 post-delete prune hook 资源（参见上文 `--wait` 说明），在钩子执行中途终止 Helm 时会无归属地遗留。后续注册会拒绝安装，而不会接管它们，因此请一并移除：

    ```bash
    kubectl delete clusterrole node-feature-discovery-prune --ignore-not-found
    kubectl delete clusterrolebinding node-feature-discovery-prune --ignore-not-found
    ```

    然后移除集群级 bootstrap RBAC，最后移除命名空间。`gpustack-system` 还存放 bootstrap ConfigMap、ServiceAccount 和注册令牌 Secret，删除命名空间也会将它们删除：

    ```bash
    kubectl delete clusterrolebinding gpustack-bootstrap --ignore-not-found
    kubectl delete namespace gpustack-default gpustack-system
    ```

5. *（可选）* 清理各节点仍保留的数据。Worker Pod 从主机挂载 `/var/lib/gpustack`，上述步骤都不会触及它；其中保存下载的模型权重和缓存。保留它不影响重新安装：在相同节点上再次注册集群时，会重新使用此目录，跳过已有内容的下载，因此仅在存储空间需要用于其他用途时才删除：

    ```bash
    rm -rf /var/lib/gpustack
    ```

使用 `kubectl get all -n gpustack-system`（应返回 `NotFound`）、`kubectl get clusterrolebinding | grep gpustack`、`kubectl get ds -A | grep -E 'gpustack|csi'` 验证；对于只有步骤 4 才会移除的对象，使用 `kubectl get apiservice,mutatingwebhookconfiguration,validatingwebhookconfiguration 2>/dev/null | grep gpustack` 和 `kubectl get crd | grep -E 'gpustack|worker\.gpustack'`（二者都应无输出）。

如果命名空间已卡在 `Terminating`，说明其中的对象仍带有 finalizer，但对应控制器已不存在。找到这些对象，再手动移除 finalizer；仅在你正在移除的集群中这样做才安全。请先删除准入 webhook（步骤 4），否则补丁操作本身也会被 Service 已不存在的 webhook 拦截：

```bash
kubectl api-resources --verbs=list --namespaced -o name |
  xargs -n1 -I{} sh -c 'kubectl get {} -n gpustack-default -o name 2>/dev/null'
kubectl patch <kind>/<name> -n gpustack-default \
  -p '{"metadata":{"finalizers":null}}' --type=merge
```

相同的移除方式也适用于集群级残留对象。保留 `gpustack.ai/controlled` finalizer 的 `instancetype.worker.gpustack.ai` 对象，会使其 CRD（以及整个 `worker.gpustack.ai` 组）一直处于删除状态：`kubectl patch instancetypes.worker.gpustack.ai <name> -p '{"metadata":{"finalizers":null}}' --type=merge`。

### 创建 DigitalOcean 集群 { #creating-digitalocean-cluster }

1. 在 `Basic Configuration` 步骤中，`Name` 为必填字段，`Description` 为可选字段。创建或选择用于访问 DigitalOcean API 的云凭据。选择支持 GPU Droplet 的区域。还必须配置新建 DigitalOcean Droplet 可以访问的 `GPUStack Server URL`。
2. 单击 `Next`。
3. 添加一个或多个 `Worker Pools`。每个 Worker Pool 可指定 `Name`、`Instance Type`、`OS Image`、`Replicas`、`Batch Size`、`Labels` 和 `Volumes`。
4. 配置完 Worker Pool 后，单击 `Save`。

集群创建后也可以添加更多 Worker Pool。

1. 进入 `Clusters` 页面。
2. 找到要添加 Worker Pool 的 `DigitalOcean` 集群。
3. 单击操作列中的省略号按钮，然后选择 `Add Worker Pool`。
4. 使用上方步骤 3 中的选项添加新 Worker Pool。

### 创建 SHUIHUA FUTURE 集群 { #creating-shuihua-future-cluster }

1. 在 `Basic Configuration` 步骤中，`Name` 为必填字段，`Description` 为可选字段。创建或选择用于访问 Shuihua API 的云凭据。Shuihua 不区分区域，因此无需选择区域。还必须配置新建实例可以访问的 `GPUStack Server URL`。
2. 此提供商必须设置 `Default Container Registry`：Shuihua 实例无法访问 Docker Hub，因此解析到 Docker Hub 的镜像仓库会被拒绝。字段建议 `quay.io` 和 `swr.cn-south-1.myhuaweicloud.com`，也接受其他可访问的仓库，例如自建 Harbor 或镜像站。
3. 单击 `Next`。
4. 添加一个或多个 `Worker Pools`。每个 Worker Pool 可指定 `Name`、`Instance Type`、`OS Image`、`Replicas`、`Batch Size` 和 `Labels`。`Instance Type` 是 Shuihua 规格模板，列表显示 GPU 型号、每小时价格和剩余库存；售罄模板不可选择。不提供卷选项，因为 Shuihua 没有块存储 API。
5. 配置完 Worker Pool 后，单击 `Save`。

集群创建后也可以添加更多 Worker Pool，方式与 DigitalOcean 集群相同。

Shuihua 实例位于 NAT 后，仅映射端口 22 和 80，因此集群默认使用 `tunnel` 代理模式，Worker 通过 GPUStack Server 的 WebSocket 隧道提供推理服务。列表中的 IP 是实例私有地址；请在 Worker 上使用 `View SSH Access` 获取实际可访问的端点。完整步骤参见[使用 Shuihua 添加 GPU 集群](../tutorials/adding-gpucluster-using-shuihua.md)。

### 管理 Worker Pool { #operating-worker-pools }

可以在 `Clusters` 页面管理云提供商集群的 Worker Pool：

1. 进入 `Clusters` 页面。
2. 找到要管理的云提供商集群，展开查看其 Worker Pool。
3. 要编辑 Worker 的副本数，请直接在 Worker 列修改。
4. 要编辑 Worker Pool，单击 `Edit` 按钮，并按需更新 `Name`、`Replica`、`Batch Size` 和 `Labels`。
5. 要删除 Worker Pool，单击其操作列中的省略号按钮，然后选择 `Delete`。

## 更新集群 { #update-cluster }

1. 进入 `Clusters` 页面。
2. 找到要编辑的集群。
3. 单击 `Edit` 按钮。
4. 根据需要更新 `Name`、`Description` 和 `Worker Configuration YAML`。
5. 单击 `Save` 按钮。

## 删除集群 { #delete-cluster }

1. 进入 `Clusters` 页面。
2. 找到要删除的集群。
3. 单击操作列中的省略号按钮，然后选择 `Delete`。
4. 确认删除。
5. 集群中仍有模型或 Worker 时，无法删除集群。

## Worker 配置 YAML { #worker-configuration-yaml }

创建或更新集群时，可以使用以下 YAML 示例预定义 Worker 配置：

```yaml
# ========= log level & tools ===========
debug: false
tools_download_base_url: https://mirror.your_company.com
# ========= directories ===========
pipx_path: "/usr/local/bin/pipx"
cache_dir: "/var/lib/gpustack/cache"
log_dir: "/var/lib/gpustack/log"
bin_dir: "/var/lib/gpustack/bin"
# ========= container & image ===========
image_name_override: "gpustack/gpustack:main"
image_repo: "gpustack/gpustack"
# ========= service & networking ===========
worker_ifname: en0
worker_port: 10150
worker_metrics_port: 10151
disable_worker_metrics: false
service_port_range: "40000-40063"
ray_port_range: "41000-41999"
proxy_mode: worker
# ========= system reserved resources ===========
system_reserved:
  ram: 0
  vram: 0
# ========= huggingface ===========
huggingface_token: xxxxxx
enable_hf_transfer: false
# ========= TLS ===========
insecure_tls: false
```

上方 YAML 列出了 `Worker Configuration YAML` 当前支持的全部选项。各选项的含义参见完整的 GPUStack [配置文件文档](../cli-reference/start.md#config-file)。`proxy_mode` 选项控制 GPUStack Server 访问 Worker 的方式，参见 [Worker 连接模式](#worker-connection-modes)。

`Docker` 和 `Kubernetes` 集群的默认容器镜像仓库都不再于此设置；请通过 `Advanced` 设置中的集群级 `Default Container Registry` 选项配置。对于 `Kubernetes` 集群，命名空间同样通过 `Namespace` 选项配置（参见 [Kubernetes 集群选项](#kubernetes-cluster-options)）。

## Worker 连接模式 { #worker-connection-modes }

要将推理请求转发给 Worker 上的模型实例，GPUStack Server 和 API 网关需要能够通过网络访问该 Worker。Worker 的 `proxy_mode` 选项通过 [Worker 配置 YAML](#worker-configuration-yaml) 或 `--proxy-mode` 参数设置，用于选择此路径：

- `direct`——GPUStack Server 和网关直接连接 Worker 公布的地址和端口。开销最低，但 GPUStack Server 必须能够直接访问 Worker。
- `worker`——请求通过 Worker 内置的 HTTP 反向代理，再由其转发给本地推理进程。GPUStack Server 仍须能访问 Worker，但只需暴露 Worker 端口。
- `tunnel`——Worker 保持一条到 GPUStack Server 的**出站** WebSocket 连接，GPUStack Server 仅通过该隧道访问 Worker。Worker 无法接受来自 GPUStack Server 的入站连接时，请使用此模式。

GPUStack Server 内嵌 Worker 默认为 `direct`，独立 Worker 默认为 `worker`。

### 隧道模式（Worker 位于防火墙或 NAT 后） { #tunnel-mode-worker-behind-a-firewall-or-nat }

在 `direct` 和 `worker` 模式下，连接由 GPUStack Server 发起，因此 GPUStack Server 必须能够访问 Worker。但这并非总能实现，例如 Worker 位于防火墙或 NAT 后、处于不同网络，或位于仅允许出站流量的私有子网中。

`tunnel` 模式反转了连接方向：连接**单向**从 Worker 建立到 GPUStack Server。

1. Worker 向 GPUStack Server 建立持久出站 WebSocket 连接，使用与注册相同的 `--server-url` 端点，通过自身令牌认证，并在连接中断时自动重连。
2. GPUStack Server 在代理端口运行 HTTP/HTTPS 代理（默认 `30079`，通过 `--proxy-port` 设置）。
3. 网关需要访问隧道模式 Worker 上的模型实例时，会将请求路由到 GPUStack Server 的代理端口，再通过已有隧道转发到 Worker。

由于 Worker 只主动向外连接，Worker 侧无需开放任何入站端口。它只需能够访问 GPUStack Server 的 API 端口，同时网关必须能够访问 GPUStack Server 的代理端口。

要启用此模式，请将 Worker 的 `proxy_mode` 设置为 `tunnel`：

```yaml
proxy_mode: tunnel
```
