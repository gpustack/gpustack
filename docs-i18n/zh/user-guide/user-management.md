---
translation_status: translated
---

# 用户管理 { #user-management }

GPUStack 有两种用户角色：`Admin` 和 `User`。

管理员可以管理集群、资源、模型、用户和系统设置。

普通用户可以管理自己的 API 密钥并访问模型 API。

## 默认管理员 { #default-admin }

首次初始化时，GPUStack 会创建一个默认管理员用户。

初始密码保存在 `<data-dir>/initial_admin_password` 中。

默认配置下，此文件位于 GPUStack Server 容器内的 `/var/lib/gpustack/initial_admin_password`。

可以在启动 `GPUStack` 时使用 `--bootstrap-password` 参数，为默认管理员设置自定义密码。

## 密码要求 { #password-requirements }

通过 UI 或 API 设置的密码必须至少包含一个大写字母、一个小写字母、一个数字，以及下列特殊字符之一：

```
! @ # $ % ^ & * _ + .
```

UI 还将密码长度限制为 6–64 个字符。

## 创建用户 { #create-user }

1. 进入 `Users` 页面。
2. 单击 `Add User` 按钮。
3. 填写用户的 `Name`、`Full Name`、`Password`，并选择 `Role`。
4. 单击 `Save` 按钮。

## 更新用户 { #update-user }

1. 进入 `Users` 页面。
2. 找到要编辑的用户。
3. 单击 `Operations` 列中的 `Edit` 按钮。
4. 根据需要更新属性。
5. 单击 `Save` 按钮。

## 停用用户 { #deactivate-user }

1. 进入 `Users` 页面。
2. 找到要停用的用户。
3. 单击 `Operations` 列中的 `Deactivate Account` 按钮。

## 启用用户 { #activate-user }

1. 进入 `Users` 页面。
2. 找到要启用的用户。
3. 单击 `Operations` 列中的 `Activate Account` 按钮。

## 删除用户 { #delete-user }

1. 进入 `Users` 页面。
2. 找到要删除的用户。
3. 单击 `Operations` 列中的省略号按钮，然后选择 `Delete`。
4. 确认删除。

!!! note

    无法通过 UI 停用或删除 admin 用户。
