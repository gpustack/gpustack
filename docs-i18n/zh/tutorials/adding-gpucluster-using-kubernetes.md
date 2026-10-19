---
translation_status: translated
---

# 使用 Kubernetes 添加 GPU 集群 { #adding-a-gpu-cluster-using-kubernetes }

GPUStack 支持将 Kubernetes 集群添加为 GPU 集群。

## 准备工作 { #preparation }

先准备 Kubernetes 集群，在所有节点上安装并配置运行时类。本教程以 k3s 集群为例。

## 创建 Kubernetes 集群 { #create-kubernetes-cluster }

请参考[注册 Kubernetes 集群](../user-guide/cluster-management.md#register-kubernetes-cluster)的步骤。

创建 Kubernetes 集群时，可配置以下选项：

![Kubernetes 集群配置选项](../assets/tutorials/adding-gpucluster-using-kubernetes/k8s-options.png)

点击 `Save` 创建集群。

创建 `Kubernetes` 提供商集群后，选择所需的 `GPU Vendors`，可以多选，也可以不选。

![选择 NVIDIA GPU 厂商](../assets/tutorials/adding-gpucluster-using-kubernetes/nvidia-gpu-vendor.png)

点击 `Next` 获取环境检查脚本，验证集群是否满足注册条件。在安装了 k3s 的主机上运行脚本。

![检查 Kubernetes 环境](../assets/tutorials/adding-gpucluster-using-kubernetes/check-environment.png)

请注意，Kubernetes 中的 `runtimeclass` 资源只是一条数据记录，并不代表每个节点上的容器运行时都已正确配置。必须配置好所选 GPU 厂商对应的容器运行时，否则 Worker 无法启动。

复制 `Run Command` 步骤中的脚本，在安装了 k3s 的主机上运行。

![运行集群注册命令](../assets/tutorials/adding-gpucluster-using-kubernetes/run-command.png)

![应用 Kubernetes 清单](../assets/tutorials/adding-gpucluster-using-kubernetes/apply-manifests.png)

## 等待 Worker 创建完成 { #waiting-for-workers-to-be-provisioned }

将清单应用到 Kubernetes 集群后，等待 Worker 出现在 `Workers` 页面中。

![已配置的 Worker](../assets/tutorials/adding-gpucluster-using-kubernetes/provisioned-workers.png)

Worker 进入 `Ready` 状态后，即可在其上部署模型。
