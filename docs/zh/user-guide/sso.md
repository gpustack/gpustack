# 单点登录（SSO）认证

GPUStack 支持 OIDC 和 SAML 等单点登录（SSO）认证方式。这样用户即可使用外部身份提供商的现有凭据登录。

## OIDC

可配置任何支持 OIDC 的身份认证提供商。如可用，将使用 `email`、`name` 和 `picture` 声明。允许的重定向 URI 应包含 `<server-url>/auth/oidc/callback`。

以下 CLI 参数可用于 OIDC 配置：

| <div style="width:180px">参数</div>     | 说明                                                                                                                                                                  |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--oidc-issuer`                         | OIDC 发行者（Issuer）URL。将使用 `<issuer>/.well-known/openid-configuration` 下的 OIDC 发现端点来获取 OIDC 配置。                                                      |
| `--oidc-client-id`                      | OIDC 客户端 ID。                                                                                                                                                       |
| `--oidc-client-secret`                  | OIDC 客户端密钥。                                                                                                                                                      |
| `--oidc-redirect-uri`                   | 在你的 OIDC 应用中配置的重定向（回调）URI。必须设置为 `<server-url>/auth/oidc/callback`。                                                                              |
| `--external-auth-name` (可选)           | 将 OIDC 用户信息映射为用户名，例如 `preferred_username`。默认情况下，如可用则使用 `email` 声明。                                                                        |
| `--external-auth-full-name` (可选)      | 将 OIDC 用户信息映射为用户全名。可以组合多个元素，如 `name` 或 `firstName+lastName`。默认使用 `name` 声明。                                                             |
| `--external-auth-avatar-url` (可选)     | 将 OIDC 用户信息映射为用户头像 URL。默认情况下，如可用则使用 `picture` 声明。                                                                                          |

你也可以通过环境变量而非 CLI 参数来设置这些选项：

```bash
GPUSTACK_OIDC_ISSUER="your-oidc-issuer-url"
GPUSTACK_OIDC_CLIENT_ID="your-client-id"
GPUSTACK_OIDC_CLIENT_SECRET="your-client-secret"
GPUSTACK_OIDC_REDIRECT_URI="{your-server-url}/auth/oidc/callback"
# 可选
GPUSTACK_EXTERNAL_AUTH_NAME="email"
GPUSTACK_EXTERNAL_AUTH_FULL_NAME="name"
GPUSTACK_EXTERNAL_AUTH_AVATAR_URL="picture"
```

### 示例：与 Auth0 OIDC 集成

将 GPUStack 配置为使用 Auth0 作为 OIDC 提供商：

1. 访问 [auth0](https://auth0.com)，创建一个新应用，类型选择 `Regular Web Applications`。

![create-oidc-app](../../assets/sso/create-oidc-app.png)

2. 在应用设置中获取 `Domain`、`Client ID` 和 `Client Secret`。

![auth0-app](../../assets/sso/auth0-app.png)

3. 在 Allowed Callback URLs 中添加 `<your-server-url>/auth/oidc/callback`。请根据你的服务器地址进行调整。

![auth0-callback](../../assets/sso/auth0-callback.png)

然后，使用相关的 OIDC 配置运行 GPUStack。以下示例使用带 CUDA 的 Docker：

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    -e GPUSTACK_OIDC_ISSUER="https://<your-auth0-domain>" \
    -e GPUSTACK_OIDC_CLIENT_ID="<your-client-id>" \
    -e GPUSTACK_OIDC_CLIENT_SECRET="<your-client-secret>" \
    -e GPUSTACK_OIDC_REDIRECT_URI="<your-server-url>/auth/oidc/callback" \
    gpustack/gpustack
```

## SAML

GPUStack 支持 SAML 单点登录（SSO）认证。这样用户即可使用支持 SAML 的外部身份提供商的现有凭据登录。

以下 CLI 参数可用于 SAML 配置：

| <div style="width:180px">参数</div>     | 说明                                                                                                                                                                                                                                                       |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--saml-idp-server-url`                 | SAML 身份提供商（IdP）服务器 URL。                                                                                                                                                                                                                        |
| `--saml-idp-entity-id`                  | SAML 身份提供商实体 ID。                                                                                                                                                                                                                                   |
| `--saml-idp-x509-cert`                  | SAML 身份提供商 X.509 证书。                                                                                                                                                                                                                               |
| `--saml-sp-entity-id`                   | SAML 服务提供商（SP）实体 ID。                                                                                                                                                                                                                            |
| `--saml-sp-acs-url`                     | SAML 服务提供商断言消费服务（ACS）URL。应设置为 `<gpustack-server-url>/auth/saml/callback`。                                                                                                                                                               |
| `--saml-sp-x509-cert`                   | SAML 服务提供商 X.509 证书。                                                                                                                                                                                                                               |
| `--saml-sp-private-key`                 | SAML 服务提供商私钥。                                                                                                                                                                                                                                      |
| `--saml-sp-attribute-prefix` (可选)     | SAML 服务提供商属性前缀，用于获取由 --external-auth-\* 指定的属性，例如 'http://schemas.auth0.com/'。                                                                                                                                                      |
| `--saml-security` (可选)                | JSON 格式的 SAML 安全设置。                                                                                                                                                                                                                                |
| `--external-auth-name` (可选)           | 将 SAML 用户信息映射为用户名。必须配置完整属性名，如 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress'，或通过 '--saml-sp-attribute-prefix' 简化为 'emailaddress'。                                                                     |
| `--external-auth-full-name` (可选)      | 将 SAML 用户信息映射为用户全名。可以组合多个元素。必须配置完整属性名，如 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name'，或通过 '--saml-sp-attribute-prefix' 简化为 'name'。                                                                |
| `--external-auth-avatar-url` (可选)     | 将 SAML 用户信息映射为用户头像 URL。必须配置完整属性名，如 'http://schemas.auth0.com/picture'，或通过 '--saml-sp-attribute-prefix' 简化为 'picture'。                                                                                                      |

你也可以通过环境变量而非 CLI 参数来设置这些选项：

```bash
GPUSTACK_SAML_IDP_SERVER_URL="https://idp.example.com"
GPUSTACK_SAML_IDP_ENTITY_ID="your-idp-entity-id"
GPUSTACK_SAML_IDP_X509_CERT="your-idp-x509-cert"
GPUSTACK_SAML_SP_ENTITY_ID="your-sp-entity-id"
GPUSTACK_SAML_SP_ACS_URL="{your-server-url}/auth/saml/callback"
GPUSTACK_SAML_SP_X509_CERT="your-sp-x509-cert"
GPUSTACK_SAML_SP_PRIVATE_KEY="your-sp-private-key"
# 可选
GPUSTACK_SAML_SP_ATTRIBUTE_PREFIX="http://schemas.auth0.com/"
GPUSTACK_SAML_SECURITY="{}"
GPUSTACK_EXTERNAL_AUTH_NAME="emailaddress"
GPUSTACK_EXTERNAL_AUTH_FULL_NAME="name"
GPUSTACK_EXTERNAL_AUTH_AVATAR_URL="picture"
```

### 示例：与 Auth0 SAML 集成

将 GPUStack 配置为使用 Auth0 作为 SAML 提供商：

1. 访问 [auth0](https://auth0.com)，创建一个新应用，类型选择 `Regular Web Applications`。

![create-saml-app](../../assets/sso/create-saml-app.png)

2. 在应用设置中获取 `Domain`，并在 Allowed Callback URLs 中添加 `<your-server-url>/auth/saml/callback`。请根据你的服务器地址进行调整。

![auth0-saml-callback](../../assets/sso/auth0-saml-callback.png)

3. 在 **Advanced Settings → Certificates** 中，复制 IdP 的 `X.509 Certificate`。

![auth0-saml-cert](../../assets/sso/auth0-saml-cert.png)

4. 在 **Endpoints** 选项卡中，找到 `SAML Protocol URL`，该地址即为你的 IdP 服务器 URL。

![auth0-saml-url](../../assets/sso/auth0-saml-url.png)

5. 生成 SP 证书和私钥：

```bash
openssl req -x509 -newkey rsa:2048 -keyout myservice.key -out myservice.cert -days 365 -nodes -subj "/CN=myservice.example.com"
```

!!! note

    myservice.cert 和 myservice.key 将用于 SP 配置。

6. 使用相关的 SAML 配置运行 GPUStack。以下示例使用带 CUDA 的 Docker：

```bash
SP_CERT="$(cat myservice.cert)"
SP_PRIVATE_KEY="$(cat myservice.key)"
SP_ATTRIBUTE_PREFIX="http://schemas.auth0.com/"

docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    -e GPUSTACK_SAML_IDP_SERVER_URL="<auth0-saml-protocol-url>" \
    -e GPUSTACK_SAML_IDP_ENTITY_ID="urn:<auth0-domain>" \
    -e GPUSTACK_SAML_IDP_X509_CERT="<auth0-x509-cert>" \
    -e GPUSTACK_SAML_SP_ENTITY_ID="urn:gpustack:sp" \
    -e GPUSTACK_SAML_SP_ACS_URL="<your-gpustack-server-url>/auth/saml/callback" \
    -e GPUSTACK_SAML_SP_X509_CERT="$SP_CERT" \
    -e GPUSTACK_SAML_SP_PRIVATE_KEY="$SP_PRIVATE_KEY" \
    -e GPUSTACK_SAML_SP_ATTRIBUTE_PREFIX="$SP_ATTRIBUTE_PREFIX" \
    gpustack/gpustack
```