---
translation_status: translated
---

# GPU 服务 实例模板 { #gpu-service-instance-templates }

GPU 服务 实例模板用于为 GPU 服务 实例定义可复用模板。

模板一次性记录镜像、命令、端口、环境变量等通用配置，便于使用相同设置创建多个实例。

## 浏览模板 { #browse-templates }

进入 `GPU Service` > `Instance Templates` 页面，浏览所有可用模板及其详情。

![截图：实例模板列表](../assets/gpuservice/instance-templates/list.png)

可以按显示名称或名称（卡片可能显示的任意一个值）筛选模板，也可以按厂商筛选。

## 添加模板 { #adding-a-template }

在 `Instance Templates` 页面，单击 `Add Instance Template` 打开创建表单。

![截图：添加实例模板表单](../assets/gpuservice/instance-templates/add.png)

!!! note

    GPU 服务 实例本质上是运行在 Kubernetes 集群中的 Pod，因此其配置遵循容器运行规则。

模板可指定以下属性：

- **Name**：模板的唯一标识符，创建时设置，之后不可更改。
- **Display Name**：便于识别的模板名称。
- **Vendor**：GPU 厂商（例如 `NVIDIA`），对于基于 CPU 的实例则使用 `CPU`。
- **Image**：从此模板创建的实例所使用的有效容器镜像。
- **Image Pull Policy**：容器镜像拉取策略（`Always`、`IfNotPresent` 或 `Never`）。
- **Container Startup Command**：容器启动时执行的命令。
- **Container Disk**：分配给容器根文件系统的磁盘容量。
- **Mount Path**：工作目录在容器内的挂载路径。
- **Ports**：容器对外暴露的网络端口。
- **Environment Variables**：在容器中设置的环境变量。

填写必填字段后，单击 `Save` 创建模板。

## 生成值占位符 { #generated-value-placeholders }

创建实例时，GPUStack 会解析启动命令和端口访问参数中的 `{{generated_*}}` 占位符，为每个实例生成一个值并保存在实例规格中，因此实例停止再启动后，该值仍保持不变。目前支持：

- `{{generated_token}}`：随机生成的 32 字符十六进制令牌。例如，内置 JupyterLab 模板使用 `--ServerApp.token={{generated_token}}` 作为启动命令，并在 JUPYTER 端口的访问参数中使用相同占位符，因此 `Connect` 列显示的 Web 访问链接会包含 `?token=` 查询参数，以实现自动登录。

占位符和端口访问参数可通过 API 设置（`spec.command` 和 `spec.ports[].accessParams`）；模板表单尚未提供这些配置项。

## 编辑模板 { #editing-a-template }

单击模板卡片上的 `Edit` 打开配置，完成修改后单击 `Save`。

![截图：编辑实例模板表单](../assets/gpuservice/instance-templates/edit.png)

## 删除模板 { #deleting-a-template }

单击模板卡片上的 `Delete` 并确认，模板随即从列表中移除。

## 从模板部署 { #deploying-from-templates }

定义模板后，即可在 [GPU 服务 实例](gpuservice-instances.md)页面从模板部署实例。
