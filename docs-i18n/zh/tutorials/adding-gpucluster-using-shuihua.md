---
translation_status: translated
---

# 使用 Shuihua 添加 GPU 集群 { #adding-a-gpu-cluster-using-shuihua }

创建集群时，GPUStack 可以通过 SHUIHUA FUTURE 创建 Worker，并将其加入 GPUStack 集群。

## 准备工作 { #preparation }

准备 SHUIHUA FUTURE 账号，并在 [Shuihua 控制台](https://hub.do.top/cn/signin)创建 API 密钥。密钥以 `amp_live_` 开头。

![水华 API 密钥](../assets/tutorials/adding-gpucluster-using-shuihua/shuihua-api-key.png)

启动 GPUStack Server 时需要指定 `--server-external-url`。Shuihua 实例通过该地址注册并维持隧道连接，因此必须能够访问该地址。如果 GPUStack Server 位于代理后方，请在此填写代理地址。

## 创建 Shuihua 集群 { #create-shuihua-cluster }

### 创建云凭证 { #create-cloud-credential }

在 `Cloud Credentials` 页面创建 `SHUIHUA FUTURE` 云凭证，只需填写 `API Key`。

![创建云凭据](../assets/tutorials/adding-gpucluster-using-shuihua/create-cloud-credential.png)

### 使用云凭证创建集群 { #create-cluster-with-cloud-credential }

在创建集群页面，将 `Cloud Provider` 设为 `SHUIHUA FUTURE`：

![选择云提供方](../assets/tutorials/adding-gpucluster-using-shuihua/select-cloud-provider.png)

输入名称，选择刚创建的云凭证，并配置 `GPUStack Server URL`。Shuihua 不区分区域，无需选择区域。

此处必须填写 `Default Container Registry`。Shuihua 实例无法访问 Docker Hub，因此 GPUStack Worker 镜像必须来自其他仓库。界面建议使用 `quay.io` 或 `swr.cn-south-1.myhuaweicloud.com`，也支持填写其他仓库，例如自己的 Harbor 或镜像站。解析为 Docker Hub 主机的值会被拒绝。

![集群基本配置](../assets/tutorials/adding-gpucluster-using-shuihua/basic-configuration.png)

> 注意：GPUStack Server 的 `--image-name-override` 优先级高于此字段。配置覆盖值后，会直接使用该镜像名称，仓库设置不再生效，因此覆盖值本身必须指向可访问的仓库。

点击 `Next`，为集群创建 Worker Pool。

### 创建 Worker Pool { #create-worker-pool }

按需填写 `Name`、`Replicas` 和 `Batch Size`。

![Worker Pool 配置](../assets/tutorials/adding-gpucluster-using-shuihua/worker-pool-configuration.png)

然后选择 `Instance Type`，即 Shuihua 的规格模板。每个选项会显示 GPU 型号、每小时价格和剩余库存。售罄的模板仍显示在列表中，但无法选择，以便说明不可用原因。

![选择实例类型](../assets/tutorials/adding-gpucluster-using-shuihua/select-instance-type.png)

接着选择 `OS Image`。

Shuihua 只提供 NVIDIA GPU，其镜像已预装驱动和 container toolkit。因此 cloud-init 只需将 Docker 指向 NVIDIA runtime 并启动 Worker，无需安装软件或重启实例，启动速度比需要构建驱动的 Worker 更快。

该提供商不提供卷，因为 Shuihua 没有块存储 API；支持配置标签。

配置完成后点击 `Save`。

## 等待 Worker 创建完成 { #waiting-for-workers-to-be-provisioned }

保存集群后，进入 `Workers` 页面查看 Shuihua Worker 的创建进度。

![配置中的 Worker](../assets/tutorials/adding-gpucluster-using-shuihua/provisioning-workers.png)

创建流程包括：

1. 生成 SSH 密钥。Shuihua 没有 SSH 密钥 API，因此不会向提供商注册密钥，而是将其写入实例的 cloud-init。
2. 创建实例。创建是异步的，Shuihua 接受请求后依次报告 `creating`、`processing` 和 `active`。
3. 等待实例启动。
4. 等待分配地址。
5. Worker 进入 `Initialized` 状态，等待 Worker 容器启动并连接 GPUStack Server。

Worker 进入 `Ready` 状态后，即可在其上部署模型。

![Worker 配置完成](../assets/tutorials/adding-gpucluster-using-shuihua/worker-provisioned.png)

## 通过 SSH 连接 Worker { #connecting-to-a-worker-over-ssh }

Shuihua 的所有实例共用一个公网 IP，并且只将实例的 22 和 80 端口映射到各自的外部端口。Worker 页面显示的是实例自身的私有地址，可用于标识实例，但不能直接连接。

请在 Worker 操作菜单中选择 `View SSH Access`。其中会显示可实际访问实例的 `Host`、`Port` 和 `Connection Command`，并可下载 `Private Key`，保存为 `worker-<id>-private_key.pem`。

![Worker SSH 连接信息](../assets/tutorials/adding-gpucluster-using-shuihua/worker-ssh-info.png)

由于 Worker 自身的端口没有映射，网关无法主动连入这些 Worker。推理请求通过 GPUStack Server 的 WebSocket 隧道传输，因此 Shuihua 集群默认使用 `tunnel` 代理模式。除非网络可以直接路由到实例，否则请保持默认设置。

## 安全缩容 Shuihua Worker { #safely-scale-down-shuihua-workers }

不再需要 Shuihua Worker 时，按以下步骤安全销毁实例：

- 根据计划删除的 Worker 数量调整 Worker Pool 的副本数。此步骤不会自动删除 Worker。
- 确认待删除 Worker 上没有已部署的模型实例。
- 按需删除 Worker，对应实例也会随之终止。

## 故障排查 { #troubleshooting }

**Worker 一直停留在 `Initialized`，未进入 `Ready`。** 实例已启动，但 Worker 容器没有启动，最常见原因是镜像拉取失败。通过 SSH 连接后检查：

```bash
sudo cloud-init status --long
sudo grep -iE "pull|denied|manifest" /var/log/cloud-init-output.log
docker ps -a
```

**Worker 已是 `Ready`，但未报告 GPU。** Docker 未使用 NVIDIA runtime：

```bash
nvidia-smi
which nvidia-ctk
docker info | grep -i runtime
```

**Worker 创建失败，无法原地重试。** 删除该 Worker，让 Worker Pool 创建替代项。每次创建请求都携带一个由 Worker 数据行派生的防重放密钥，因此创建新实例需要一个新的 Worker。
