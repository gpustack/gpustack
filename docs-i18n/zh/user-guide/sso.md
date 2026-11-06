---
translation_status: translated
---

# 单点登录（SSO）认证 { #single-sign-on-sso-authentication }

GPUStack 支持 OIDC、SAML 和 CAS 等单点登录（SSO）认证方式，让用户可以使用外部身份提供商的现有凭据登录。

同一时间只有一种 SSO 方式处于启用状态，登录页的 SSO 按钮会显示所用协议，例如 `Log in with OIDC`。如果希望显示身份提供商名称，例如 `Log in with Okta`，请设置 `--external-auth-provider-name`。

## OIDC { #oidc }

可以配置任何支持 OIDC 的认证提供商。如果可用，会使用 `email`、`name` 和 `picture` 声明。允许的重定向 URI 应包含 `<server-url>/auth/oidc/callback`。

如果 OIDC 提供商使用由私有或企业 CA 签发的证书，参见[额外受信任的 CA](../installation/installation.md#additional-trusted-cas)，了解如何将 CA 证书挂载到 GPUStack 容器。

以下 CLI 参数可用于 OIDC 配置：

| <div style="width:180px">参数</div> | 描述 |
| --- | --- |
| `--oidc-issuer` | OIDC 签发者 URL。将通过 `<issuer>/.well-known/openid-configuration` 下的 OIDC 发现机制获取 OIDC 配置。 |
| `--oidc-client-id` | OIDC 客户端 ID。 |
| `--oidc-client-secret` | OIDC 客户端密钥。 |
| `--oidc-redirect-uri` | OIDC 应用中配置的重定向 URI，必须设置为 `<server-url>/auth/oidc/callback`。 |
| `--external-auth-provider-name`（可选） | 登录页 SSO 按钮上显示的身份提供商名称，例如 `Okta`。未设置时默认为 `OIDC`。 |
| `--external-auth-name`（可选） | OIDC 用户信息到用户名的映射，例如 `preferred_username`。默认使用 `email` 声明（如果可用）。 |
| `--external-auth-full-name`（可选） | OIDC 用户信息到用户全名的映射。可组合多个元素，例如 `name` 或 `firstName+lastName`。默认使用 `name` 声明。 |
| `--external-auth-avatar-url`（可选） | OIDC 用户信息到用户头像 URL 的映射。默认使用 `picture` 声明（如果可用）。 |
| `--external-auth-default-inactive`（可选） | 阻止新 SSO 用户默认激活。 |
| `--external-auth-post-logout-redirect-key`（可选） | 不同 IdP 注销后重定向使用的通用参数名（例如 Auth0 的 `returnTo`）。同时适用于 OIDC 和 SAML。 |

也可以使用环境变量代替 CLI 参数设置这些选项：

```bash
GPUSTACK_OIDC_ISSUER="your-oidc-issuer-url"
GPUSTACK_OIDC_CLIENT_ID="your-client-id"
GPUSTACK_OIDC_CLIENT_SECRET="your-client-secret"
GPUSTACK_OIDC_REDIRECT_URI="{your-server-url}/auth/oidc/callback"
# Optional
GPUSTACK_EXTERNAL_AUTH_PROVIDER_NAME="Okta"
GPUSTACK_EXTERNAL_AUTH_NAME="email"
GPUSTACK_EXTERNAL_AUTH_FULL_NAME="name"
GPUSTACK_EXTERNAL_AUTH_AVATAR_URL="picture"
GPUSTACK_EXTERNAL_AUTH_DEFAULT_INACTIVE="true"
GPUSTACK_EXTERNAL_AUTH_POST_LOGOUT_REDIRECT_KEY="returnTo"  # e.g., for Auth0
```

### 示例：集成 Auth0 OIDC { #example-integrate-with-auth0-oidc }

要将 Auth0 配置为 GPUStack 的 OIDC 提供商：

1. 进入 [Auth0](https://auth0.com)，创建类型为 `Regular Web Applications` 的新应用。

![创建 OIDC 应用](../assets/sso/create-oidc-app.png)

2. 从应用设置中获取 `Domain`、`Client ID` 和 `Client Secret`。

![Auth0 应用](../assets/sso/auth0-app.png)

3. 在 Allowed Callback URLs 中添加 `<your-server-url>/auth/oidc/callback`。请按 GPUStack Server 的实际 URL 调整。

4. 在 Allowed Logout URLs 中添加 `<your-server-url>/`（或期望的注销后 URL）。

![Auth0 回调](../assets/sso/auth0-callback.png)

然后使用相应 OIDC 配置运行 GPUStack。以下示例使用带 CUDA 的 Docker：

```bash
sudo docker run -d --name gpustack \
    --restart=unless-stopped \
    --privileged \
    --network=host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --volume /path/to/custom-root-ca.crt:/usr/local/share/ca-certificates/custom-root-ca.crt:ro \
    --runtime nvidia \
    -e GPUSTACK_OIDC_ISSUER="https://<your-auth0-domain>" \
    -e GPUSTACK_OIDC_CLIENT_ID="<your-client-id>" \
    -e GPUSTACK_OIDC_CLIENT_SECRET="<your-client-secret>" \
    -e GPUSTACK_OIDC_REDIRECT_URI="<your-server-url>/auth/oidc/callback" \
    gpustack/gpustack
```

!!! note

    仅当 OIDC 提供商的证书由私有 CA 签发时，才需要挂载自定义 CA 证书。Auth0 等公共 OIDC 提供商通常不需要。

## SAML { #saml }

GPUStack 支持用于单点登录（SSO）的 SAML 认证，让用户可以使用支持 SAML 的外部身份提供商的现有凭据登录。

以下 CLI 参数可用于 SAML 配置：

| <div style="width:180px">参数</div> | 描述 |
| --- | --- |
| `--saml-idp-server-url` | SAML 身份提供商服务器 URL。 |
| `--saml-idp-entity-id` | SAML 身份提供商实体 ID。 |
| `--saml-idp-x509-cert` | SAML 身份提供商 X.509 证书。 |
| `--saml-sp-entity-id` | SAML 服务提供商实体 ID。 |
| `--saml-sp-acs-url` | SAML 服务提供商断言消费服务 URL，应设置为 `<gpustack-server-url>/auth/saml/callback`。 |
| `--saml-sp-x509-cert` | SAML 服务提供商 X.509 证书。 |
| `--saml-sp-private-key` | SAML 服务提供商私钥。 |
| `--saml-idp-logout-url`（可选） | SAML 身份提供商单点注销端点 URL。 |
| `--saml-sp-slo-url`（可选） | SAML 服务提供商单点注销服务回调 URL（例如 `<server-url>/auth/saml/logout/callback`）。 |
| `--saml-sp-attribute-prefix`（可选） | SAML 服务提供商属性前缀，用于获取 --external-auth-\* 指定的属性，例如 'http://schemas.auth0.com/'。 |
| `--saml-security`（可选） | JSON 格式的 SAML 安全设置。 |
| `--external-auth-provider-name`（可选） | 登录页 SSO 按钮上显示的身份提供商名称，例如 `Okta`。未设置时默认为 `SAML`。 |
| `--external-auth-name`（可选） | SAML 用户信息到用户名的映射。必须配置完整属性名，如 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress'，或通过 '--saml-sp-attribute-prefix' 简化为 'emailaddress'。 |
| `--external-auth-full-name`（可选） | SAML 用户信息到用户全名的映射，可组合多个元素。必须配置完整属性名，如 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name'，或通过 '--saml-sp-attribute-prefix' 简化为 'name'。 |
| `--external-auth-avatar-url`（可选） | SAML 用户信息到用户头像 URL 的映射。必须配置完整属性名，如 'http://schemas.auth0.com/picture'，或通过 '--saml-sp-attribute-prefix' 简化为 'picture'。 |
| `--external-auth-default-inactive`（可选） | 阻止新 SSO 用户默认激活。 |

也可以使用环境变量代替 CLI 参数设置这些选项：

```bash
GPUSTACK_SAML_IDP_SERVER_URL="https://idp.example.com"
GPUSTACK_SAML_IDP_ENTITY_ID="your-idp-entity-id"
GPUSTACK_SAML_IDP_X509_CERT="your-idp-x509-cert"
GPUSTACK_SAML_SP_ENTITY_ID="your-sp-entity-id"
GPUSTACK_SAML_SP_ACS_URL="{your-server-url}/auth/saml/callback"
GPUSTACK_SAML_SP_X509_CERT="your-sp-x509-cert"
GPUSTACK_SAML_SP_PRIVATE_KEY="your-sp-private-key"
# Optional
GPUSTACK_SAML_SP_ATTRIBUTE_PREFIX="http://schemas.auth0.com/"
GPUSTACK_SAML_SECURITY="{}"
GPUSTACK_EXTERNAL_AUTH_PROVIDER_NAME="Okta"
GPUSTACK_EXTERNAL_AUTH_NAME="emailaddress"
GPUSTACK_EXTERNAL_AUTH_FULL_NAME="name"
GPUSTACK_EXTERNAL_AUTH_AVATAR_URL="picture"
GPUSTACK_EXTERNAL_AUTH_DEFAULT_INACTIVE="true"
GPUSTACK_SAML_IDP_LOGOUT_URL="https://idp.example.com/saml/slo"  # if IdP supports SLO
GPUSTACK_SAML_SP_SLO_URL="{your-server-url}/auth/saml/logout/callback"
GPUSTACK_EXTERNAL_AUTH_POST_LOGOUT_REDIRECT_KEY="returnTo"  # optional, adds a post-logout parameter for compatible IdPs
```

### 示例：集成 Auth0 SAML { #example-integrate-with-auth0-saml }

要将 Auth0 配置为 GPUStack 的 SAML 提供商：

1. 进入 [Auth0](https://auth0.com)，创建类型为 `Regular Web Applications` 的新应用。

![创建 SAML 应用](../assets/sso/create-saml-app.png)

2. 从应用设置中获取 `Domain`，并在 Allowed Callback URLs 中添加 `<your-server-url>/auth/saml/callback`。请按 GPUStack Server 的实际 URL 调整。

![Auth0 SAML 回调](../assets/sso/auth0-saml-callback.png)

3. 在 **Advanced Settings → Certificates** 中，复制 IdP 的 `X.509 Certificate`。

![Auth0 SAML 证书](../assets/sso/auth0-saml-cert.png)

4. 在 **Endpoints** 选项卡中找到 `SAML Protocol URL`，即 IdP 服务器 URL。

![Auth0 SAML 地址](../assets/sso/auth0-saml-url.png)

5. 生成 SP 证书和私钥：

```bash
openssl req -x509 -newkey rsa:2048 -keyout myservice.key -out myservice.cert -days 365 -nodes -subj "/CN=myservice.example.com"
```

!!! note

    myservice.cert 和 myservice.key 将用于 SP 配置。

6. 使用相应 SAML 配置运行 GPUStack。以下示例使用带 CUDA 的 Docker：

```bash
SP_CERT="$(cat myservice.cert)"
SP_PRIVATE_KEY="$(cat myservice.key)"
SP_ATTRIBUTE_PREFIX="http://schemas.auth0.com/"

sudo docker run -d --name gpustack \
    --restart=unless-stopped \
    --privileged \
    --network=host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --runtime nvidia \
    -e GPUSTACK_SAML_IDP_SERVER_URL="<auth0-saml-protocol-url>" \
    -e GPUSTACK_SAML_IDP_ENTITY_ID="urn:<auth0-domain>" \
    -e GPUSTACK_SAML_IDP_X509_CERT="<auth0-x509-cert>" \
    -e GPUSTACK_SAML_SP_ENTITY_ID="urn:gpustack:sp" \
    -e GPUSTACK_SAML_SP_ACS_URL="<your-gpustack-server-url>/auth/saml/callback" \
    -e GPUSTACK_SAML_SP_X509_CERT="$SP_CERT" \
    -e GPUSTACK_SAML_SP_PRIVATE_KEY="$SP_PRIVATE_KEY" \
    -e GPUSTACK_SAML_SP_ATTRIBUTE_PREFIX="$SP_ATTRIBUTE_PREFIX" \
    -e GPUSTACK_SAML_IDP_LOGOUT_URL="<idp-slo-url-if-available>" \
    -e GPUSTACK_SAML_SP_SLO_URL="<your-gpustack-server-url>/auth/saml/logout/callback" \
    -e GPUSTACK_EXTERNAL_AUTH_POST_LOGOUT_REDIRECT_KEY="returnTo" \
    gpustack/gpustack
```

!!! note

    并非所有 IdP 都提供标准 SAML 单点注销（SLO）。Auth0 SAML 连接通常不暴露 `singleLogoutService`。如果不可用，GPUStack 在注销时仍会清除本地会话；要通过 Auth0 完整退出浏览器登录，可以考虑使用其 OIDC `v2/logout`，并允许 `client_id` 和 `returnTo`。

## CAS { #cas }

!!! note

    现代 Apereo CAS 服务器在原生 CAS 协议之外还提供 OIDC 和 SAML2 模块。如果 CAS 服务器已启用其中任一模块，也可通过上方 [OIDC](#oidc) 或 [SAML](#saml) 部分进行接入，能力相同，选择与服务器实际暴露协议匹配的方式即可。如果仅有 CAS 可用（常见于较旧的 Apereo / 教育 / 政府部署），或希望集成范围最小，请使用下述原生 CAS 协议。

GPUStack 支持 CAS（Central Authentication Service）协议 2.0 和 3.0。登录流程将浏览器重定向到 `<cas-server-url>/login?service=<gpustack-callback>`；成功后，GPUStack 通过标准 `/serviceValidate` 端点交换返回的服务票据，并根据 XML 响应创建用户。向 CAS 服务器注册的回调 URL 应指向 `<server-url>/auth/cas/callback`。

无论 CAS 响应是否包含 `cas:` XML 命名空间，都能解析，因此大多数现有服务器无需额外调整。CAS 单点注销接入 `/auth/logout`；当当前会话来源于 CAS 时，会将浏览器发送到 `<cas-server-url>/logout?service=<server-external-url>`。

以下 CLI 参数可用于 CAS 配置：

| <div style="width:200px">参数</div> | 描述 |
| --- | --- |
| `--cas-server-url` | CAS 服务器基础 URL，例如 `https://cas.example.com/cas`。设置后启用 CAS 登录。 |
| `--cas-callback-url`（可选） | 向 CAS 服务器注册的回调 URL。未设置时，在请求时默认为 `<server-url>/auth/cas/callback`。 |
| `--cas-validate-endpoint`（可选） | CAS 票据验证端点，相对于 `--cas-server-url`。默认值：`/p3/serviceValidate`（CAS 3.0，返回用户属性）。仅连接 CAS 3.0 之前的服务器时使用 `/serviceValidate`。 |
| `--cas-username-attribute`（可选） | 用作 GPUStack 用户名的 CAS XML 属性名。未设置时默认使用 CAS 的 `cas:user` 元素。 |
| `--cas-full-name-attribute`（可选） | 用作用户全名的 CAS XML 属性名，例如 `displayName`。回退时，如果存在 `displayName` 属性则使用它，否则使用用户名。 |
| `--cas-avatar-attribute`（可选） | 用作用户头像 URL 的 CAS XML 属性名。 |
| `--external-auth-provider-name`（可选） | 登录页 SSO 按钮上显示的身份提供商名称，例如 `Campus SSO`。未设置时默认为 `CAS`。 |
| `--external-auth-default-inactive`（可选） | 阻止新 SSO 用户默认激活。 |

也可以使用环境变量代替 CLI 参数设置这些选项：

```bash
GPUSTACK_CAS_SERVER_URL="https://cas.example.com/cas"
# Optional
GPUSTACK_CAS_CALLBACK_URL="{your-server-url}/auth/cas/callback"
GPUSTACK_CAS_VALIDATE_ENDPOINT="/p3/serviceValidate"
GPUSTACK_CAS_USERNAME_ATTRIBUTE="uid"
GPUSTACK_CAS_FULL_NAME_ATTRIBUTE="displayName"
GPUSTACK_CAS_AVATAR_ATTRIBUTE=""
GPUSTACK_EXTERNAL_AUTH_PROVIDER_NAME="Campus SSO"
GPUSTACK_EXTERNAL_AUTH_DEFAULT_INACTIVE="true"
```

### 示例：集成 Apereo CAS 服务器 { #example-integrate-with-an-apereo-cas-server }

1. 在 CAS 服务器上将 GPUStack 注册为服务，令 `serviceId` 匹配 `<your-server-url>/auth/cas/callback`。允许要映射的属性（例如 `uid`、`displayName`）。

2. 使用 CAS 配置运行 GPUStack。以下示例使用带 CUDA 的 Docker：

```bash
sudo docker run -d --name gpustack \
    --restart=unless-stopped \
    --privileged \
    --network=host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --runtime nvidia \
    -e GPUSTACK_SERVER_EXTERNAL_URL="https://gpustack.example.com" \
    -e GPUSTACK_CAS_SERVER_URL="https://cas.example.com/cas" \
    -e GPUSTACK_CAS_USERNAME_ATTRIBUTE="uid" \
    -e GPUSTACK_CAS_FULL_NAME_ATTRIBUTE="displayName" \
    gpustack/gpustack
```

!!! note

    如果 CAS 服务器使用由私有或企业 CA 签发的证书，参见[额外受信任的 CA](../installation/installation.md#additional-trusted-cas)，了解如何将 CA 证书挂载到 GPUStack 容器。
