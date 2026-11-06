---
translation_status: translated
---

# GPU 服务存储 { #gpu-service-storage }

GPU 服务存储基于 [GPU 服务存储类型](gpuservice-storage-types.md)，可分配持久化存储，并在多个 GPU 服务实例中挂载。

它适用于需要在实例间共享的数据、模型和其他文件。

GPUStack 可管理多个 Kubernetes 集群，并提供统一界面，在这些集群中共享 GPU 服务存储。

底层每个存储项都对应目标集群中的 Kubernetes `PersistentVolumeClaim`。

## 浏览存储 { #browse-storage }

进入 `GPU Service` > `Storage` 页面，浏览所有存储及其详情。

![截图：存储列表](../assets/gpuservice/storage/list.png)

可以按显示名称或名称筛选存储，即 `Name` 列可能显示的任意一个值。对 `Name` 列排序时，按实际显示的标签排序。

## 添加存储 { #adding-storage }

在 `Storage` 页面，单击 `Add Storage` 打开创建表单。

![截图：添加存储表单](../assets/gpuservice/storage/add.png)

填写 `Name`，选择 `Storage Type`（在 [GPU 服务存储类型](gpuservice-storage-types.md)页面定义），并设置 `Capacity (GB)`，然后单击 `Save`。

## 编辑存储 { #editing-storage }

创建后只能修改显示名称。

!!! note

    后续版本计划支持编辑更多存储配置。

## 删除存储 { #deleting-storage }

单击存储项上的 `Delete` 并确认，该存储随即从列表中移除。

!!! warning

    如果存储仍挂载在 GPU 服务实例上，集群侧的 `PersistentVolumeClaim` 不会立即移除。由于删除采用延迟策略，`PersistentVolumeClaim` 会保留到关联实例删除后才被清理。
