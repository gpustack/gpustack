---
translation_status: translated
---

# GPU 服务存储类型 { #gpu-service-storage-types }

GPU 服务存储类型用于为 GPU 服务实例定义共享文件系统存储后端，包括 NFS 或 S3。

它帮助管理员管理多个 Kubernetes 集群中的存储定义。定义后，可在创建 [GPU 服务存储](gpuservice-storage.md)时选择相应存储类型。

底层每个存储类型都对应目标集群中的 Kubernetes `StorageClass`。

## 浏览存储类型 { #browse-storage-types }

进入 `GPU Service` > `Storage Types` 页面，浏览所有存储类型及其详情。

![截图：存储类型列表](../assets/gpuservice/storage-types/list.png)

可以按显示名称或名称筛选存储类型，即 `Name` 列可能显示的任意一个值。对 `Name` 列排序时，按实际显示的标签排序。

## 添加存储类型 { #adding-a-storage-type }

在 `Storage Types` 页面，单击 `Add Storage Type` 打开创建表单。

![截图：添加存储类型表单](../assets/gpuservice/storage-types/add.png)

填写 `Name` 和可选的 `Display Name`，然后选择 `Type`。其余字段取决于所选类型。

**NFS**——挂载 NFS 服务器导出的共享目录：

- **NFS Server**（必填）：NFS 服务器的主机名或 IP 地址。
- **Share Path**（必填）：NFS 服务器上导出的共享路径。
- **Sub Directory**：用作卷的共享目录内子目录。留空时使用共享目录的根目录。
- **Mount Permissions**：应用于挂载目录的权限，例如 `0777`。默认为 `0`，表示遵循 NFS 服务器配置的权限。
- **Mount Options**：逐项添加的 NFS 挂载选项。默认为 `hard`、`vers=4`、`rsize=1048576`、`wsize=1048576`、`noatime` 和 `nodiratime`。

**S3**——挂载兼容 S3 的对象存储桶：

- **Endpoint**（必填）：兼容 S3 的存储端点 URL。
- **Region**：S3 存储所在区域（如适用）。
- **Bucket**：要使用的存储桶名称。留空时会创建名为 `gpu-instance-pv-{id}` 的默认存储桶。
- **Access Key**（必填）：用于向存储进行身份验证的访问密钥。
- **Secret Key**（必填）：用于向存储进行身份验证的私密密钥。此字段只写，绝不会在 API 响应中返回。
- **Insecure**：连接端点时跳过 TLS/SSL 证书验证。默认禁用。
- **Mount Options**：底层 [GeeseFS](https://github.com/yandex-cloud/geesefs) 驱动的挂载选项，逐项添加。默认值采用**大文件密集写入**预设，适合大多数工作负载。

S3 卷通过 [GeeseFS](https://github.com/yandex-cloud/geesefs) 挂载，`Mount Options` 会直接传给它。请选择符合访问模式的预设；如果端点不是 Yandex Object Storage，还应追加**非 Yandex 的兼容 S3 服务**选项。

??? note "常用 GeeseFS 挂载选项预设"

    | 使用场景 | 挂载选项 |
    | --- | --- |
    | 大文件密集写入（默认） | `--no-checksum` `--memory-limit=4000` `--max-flushers=32` `--max-parallel-parts=32` `--part-sizes=25` |
    | 大文件顺序读取 | `--read-ahead-large=200000` `--large-read-cutoff=10240` `--read-ahead-parallel=40000` `--memory-limit=8000` |
    | 小文件随机读取 | `--read-ahead-small=64` `--small-read-cutoff=64` `--read-ahead=1024` `--stat-cache-ttl=300s` `--entry-limit=200000` |
    | 高可用写入 | `--sdk-max-retries=10` `--read-retry-attempts=5` `--fsync-on-close` `--cache=/mnt/disk-cache` |
    | 非 Yandex 的兼容 S3 服务 | `--list-type=2` `--no-specials` |

填写必填字段后，单击 `Save` 创建存储类型。

## 编辑存储类型 { #editing-a-storage-type }

创建后只能修改显示名称。

!!! note

    后续版本计划支持编辑更多存储类型配置。

## 删除存储类型 { #deleting-a-storage-type }

单击存储类型上的 `Delete` 并确认，该存储类型随即从列表中移除。

!!! warning

    删除存储类型前，请先删除由它创建的所有 [GPU 服务存储](gpuservice-storage.md)。
