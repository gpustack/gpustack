# 用户管理

GPUStack 支持两种角色的用户：`Admin` 和 `User`。管理员可以监控系统状态，管理模型、用户和系统设置。用户可以管理自己的 API 密钥，并使用 model API。

## 默认管理员

在初始化时，GPUStack 会创建一个默认的管理员用户。默认管理员的初始密码存储在 `<data-dir>/initial_admin_password`。在默认设置下，其路径为 `/var/lib/gpustack/initial_admin_password`。你可以在启动 `gpustack` 时通过设置 `--bootstrap-password` 参数自定义默认管理员密码。

## 创建用户

1. 进入 `Users` 页面。
2. 点击 `Create User` 按钮。
3. 填写 `Name`、`Full Name`、`Password`，并为该用户选择 `Role`。
4. 点击 `Save` 按钮。

## 更新用户

1. 进入 `Users` 页面。
2. 找到需要编辑的用户。
3. 点击 `Operations` 列中的 `Edit` 按钮。
4. 按需更新相关属性。
5. 点击 `Save` 按钮。

## 删除用户

1. 进入 `Users` 页面。
2. 找到需要删除的用户。
3. 点击 `Operations` 列中的省略号按钮，然后选择 `Delete`。
4. 确认删除。