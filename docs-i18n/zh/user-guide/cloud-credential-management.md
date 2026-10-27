---
translation_status: translated
---

# 云凭据管理 { #cloud-credential-management }

GPUStack 支持云凭据管理，用于安全连接外部云提供商。云凭据包含访问 API 所需的提供商信息、密钥和选项。

## 支持的提供商 { #supported-providers }

支持 `DigitalOcean` 和 `SHUIHUA FUTURE`。

## 创建云凭据 { #create-cloud-credential }

1. 进入 `Cloud Credentials` 页面。
2. 单击 `Add Cloud Credential` 下拉菜单并选择提供商。
3. 填写以下信息：

   - `Name`：唯一的凭据名称。
   - 提供商的密钥，这是两者唯一需要的凭据字段：
     - `DigitalOcean`——`Access Token`，即在 DigitalOcean 的 `Applications & API` 页面生成的 API 令牌。令牌作用域必须为 `Full Access`。
     - `SHUIHUA FUTURE`——`API Key`，在[水华控制台](https://hub.do.top/cn/signin)创建，以 `amp_live_` 为前缀。
   - `Description`：云凭据的附加信息。

4. 单击 `Save` 按钮。

## 更新云凭据 { #update-cloud-credential }

1. 进入 `Cloud Credentials` 页面。
2. 找到要编辑的凭据。
3. 单击 `Edit` 按钮。
4. 根据需要更新 `Name`、`Access Token` 和 `Description`。
5. 单击 `Save` 按钮。

## 删除云凭据 { #delete-cloud-credential }

1. 进入 `Cloud Credentials` 页面。
2. 找到要删除的凭据。
3. 单击操作列中的省略号按钮，然后选择 `Delete`。
4. 确认删除。
