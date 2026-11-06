---
translation_status: translated
---

# GPU 服务 SSH 公钥 { #gpu-service-ssh-public-keys }

GPU 服务 SSH 公钥用于管理访问 GPU 服务实例所需的 SSH 公钥，实现安全的免密码访问。

在实例的 `SSH Access` 配置中分配这些密钥，参见 [GPU 服务实例](gpuservice-instances.md)。

## 浏览 SSH 公钥 { #browse-ssh-public-keys }

进入 `GPU Service` > `SSH Public Keys` 页面，浏览所有已注册密钥。

![截图：SSH 公钥列表](../assets/gpuservice/ssh-public-keys/list.png)

可以按显示名称或名称筛选 SSH 公钥，即 `Name` 列可能显示的任意一个值。对 `Name` 列排序时，按实际显示的标签排序。

## 添加 SSH 公钥 { #adding-an-ssh-public-key }

在 `SSH Public Keys` 页面，单击 `Add SSH Public Key` 打开创建表单。

![截图：添加 SSH 公钥表单](../assets/gpuservice/ssh-public-keys/add.png)

填写 `Name`，然后将公钥粘贴到 `SSH Public Key` 字段。每个公钥必须以 `ssh-` 开头；每行输入一个公钥可一次注册多个。单击 `Save` 完成操作。

!!! tip

    要输出公钥，请运行 `cat ~/.ssh/id_rsa.pub`（RSA）或 `cat ~/.ssh/id_ed25519.pub`（Ed25519）。

## 编辑 SSH 公钥 { #editing-an-ssh-public-key }

单击密钥上的 `Edit` 打开配置，更新设置后单击 `Save`。

![截图：编辑 SSH 公钥表单](../assets/gpuservice/ssh-public-keys/edit.png)

## 删除 SSH 公钥 { #deleting-an-ssh-public-key }

单击密钥上的 `Delete` 并确认，该密钥随即从列表中移除。

!!! note

    如果 SSH 公钥已被 GPU 服务实例使用，在此删除不会从该实例中移除它。其内容已同步到 Worker 侧集群，无法事后从中单独移除一个条目。要撤销运行中实例的密钥，请打开该实例，单击 `Edit`，并在其 `SSH Access` 配置下取消选择该密钥。
