---
translation_status: translated
---

# 使用 DigitalOcean 添加 GPU 集群 { #adding-a-gpu-cluster-using-digitalocean }

创建集群时，GPUStack 可以通过 DigitalOcean 创建 Worker，并将其加入 GPUStack 集群。

## 准备工作 { #preparation }

注册 DigitalOcean 账号，并在 API 页面创建 Personal Access Token。

![DigitalOcean 个人访问令牌](../assets/tutorials/adding-gpucluster-using-digitalocean/digital-ocean-pat.png)

> 注意：令牌权限范围必须设为 Full Access。如果通过 Custom Scopes 自选权限，删除 Droplet 时可能遇到问题。

启动 GPUStack Server 时需要指定 `--server-external-url`。Droplet 创建并启动 Worker 后，该参数用于配置 Worker 的 `--server-url`。如果 GPUStack Server 位于代理后方，请填写代理地址，确保公网中的 Droplet 启动后能够通过该地址访问 GPUStack Server API。

## 创建 DigitalOcean 集群 { #create-digitalocean-cluster }

请参考[创建 DigitalOcean 集群](../user-guide/cluster-management.md#creating-digitalocean-cluster)的步骤。

### 创建云凭证 { #create-cloud-credential }

在 `Cloud Credentials` 页面创建 DigitalOcean 云凭证。

![创建 DigitalOcean 云凭据](../assets/tutorials/adding-gpucluster-using-digitalocean/create-cloud-credential.png)

### 使用云凭证创建集群 { #create-cluster-with-cloud-credential }

在创建集群页面，将 `Cloud Provider` 设为 `DigitalOcean`：

![选择云提供方](../assets/tutorials/adding-gpucluster-using-digitalocean/select-cloud-provider.png)

输入名称，选择刚创建的云凭证，并选择支持 GPU Droplet 的区域。

![选择区域](../assets/tutorials/adding-gpucluster-using-digitalocean/select-region.png)

点击 `Next`，为集群创建 Worker Pool。

### 创建 Worker Pool { #create-worker-pool }

按需填写 `Name`、`Replicas` 和 `Batch Size`，然后选择 Worker Pool 的 `Instance Type`，即 DigitalOcean 中的 Droplet 规格。

![选择实例类型](../assets/tutorials/adding-gpucluster-using-digitalocean/select-instance-type.png)

接着选择 `OS Image`。

![选择操作系统镜像](../assets/tutorials/adding-gpucluster-using-digitalocean/select-os-image.png)

目前仅支持 NVIDIA 系列 GPU Droplet，因为 AMD GPU Droplet 经常不可用。仅支持基于 Debian 的操作系统。

- 推荐使用 Nvidia AI/ML Ready：基于 Ubuntu 22.04，已预装驱动、CUDA 和 container-toolkit，无需额外软件包。
- 对于 Debian 发行版，cloud-init 会安装 `nvidia-open` 驱动、CUDA 12.8 和 nvidia-container-toolkit 1.17.8-1。
- 对于 Ubuntu 发行版，cloud-init 会安装 `nvidia-driver-570` 驱动、CUDA 12.8 和 nvidia-container-toolkit 1.17.8-1。

Worker Pool 支持通过标签和卷配置 Worker。创建并挂载到 Droplet 的卷不会自动挂载进 Worker 容器；可按需修改 `/opt/gpustack-run-worker.sh` 中的 Worker 启动脚本来挂载卷。

配置完成后点击 `Save`。

## 等待 Worker 创建完成 { #waiting-for-workers-to-be-provisioned }

保存集群后，进入 `Workers` 页面查看 DigitalOcean Worker 的创建进度。

![配置 Worker](../assets/tutorials/adding-gpucluster-using-digitalocean/provisioning-workers.png)

创建流程包括：

1. 创建 SSH 密钥。
2. 使用该 SSH 密钥创建 Droplet。
3. 等待 Droplet 启动。
4. 等待分配公网 IP。
5. 创建卷并挂载到 Droplet。
6. Worker 进入 `Initialized` 状态，等待 Worker 容器启动并连接 GPUStack Server。

Worker 进入 `Ready` 状态后，即可在其上部署模型。

![Worker 配置完成](../assets/tutorials/adding-gpucluster-using-digitalocean/worker-provisioned.png)

## 安全缩容 DigitalOcean Worker { #safely-scale-down-digitalocean-workers }

不再需要 DigitalOcean Worker 时，按以下步骤安全销毁 Droplet：

- 根据计划删除的 Worker 数量调整 Worker Pool 的副本数。此步骤不会自动删除 Worker。
- 确认待删除 Worker 上没有已部署的模型实例。
- 按需删除 Worker，对应 Droplet 也会随之删除。
