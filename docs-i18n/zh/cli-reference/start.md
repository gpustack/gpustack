---
translation_status: translated
hide:
  - toc
---

# gpustack start { #gpustack-start }

运行 GPUStack Server 或 Worker。

```bash
gpustack start [OPTIONS]
```

!!! note "CLI 参数位置"

    在 Docker 中，`docker run` 实际执行的命令由 **ENTRYPOINT** 和 **COMMAND** 组成。

    `gpustack/gpustack` 镜像设置了 `ENTRYPOINT`，可简单理解为
    （或视为等同于）：

        gpustack start

    因此，`gpustack start` 的 CLI 参数必须放在镜像名称之后，
    即 `docker run` 命令的末尾，而不是作为 `docker run` 自身的选项。

    示例：

        docker run [docker options] gpustack/gpustack <start-args>

    在 Kubernetes 中，应使用容器配置中的 `args` 字段
    指定这些参数。

## 配置 { #configurations }

### 通用选项 { #common-options }

| <div style="width:180px">参数</div>         | <div style="width:100px">默认值</div> | 说明                                                                                                                           |
|---------------------------------------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `--advertise-address` value                 | (empty)                                | 对外访问使用的 IP 地址。<br/>未设置时，系统自动检测合适的本机 IP 地址。                |
| `--config-file` value                       | (empty)                                | YAML 配置文件路径。                                                                                                         |
| `-d` value, `--debug` value                 | `False`                                | 启用调试模式。Windows 不支持短参数 -d，因为 PowerShell 已将其保留用于 CommonParameters。 |
| `--data-dir` value                          | (empty)                                | 数据存储目录。默认值取决于操作系统。                                                                                      |
| `--cache-dir` value                         | (empty)                                | 缓存（例如模型文件）存储目录。默认为 <data-dir>/cache。                                                           |
| `--huggingface-token` value                 | (empty)                                | 用于 Hugging Face Hub 身份认证的用户访问 Token，也可通过 `HF_TOKEN` 环境变量配置。                                                                            |
| `--insecure-tls`                            | `False`                                | 跳过 GPUStack 自身 HTTPS 连接的 TLS 验证（Worker 到 Server、Server 到外部身份认证 IdP）。适用于无法通过其他方式验证 Server 证书的可信网络。                                                                                                                                                                                                              |
| `--bin-dir` value                           | (empty)                                | 额外二进制文件的存储目录，例如带版本的后端可执行文件。                                                          |
| `--pipx-path` value                         | (empty)                                | pipx 可执行文件路径，用于安装带版本的后端。                                                                      |
| `--system-default-container-registry` value | `docker.io`                            | GPUStack 拉取系统镜像和推理镜像的默认容器镜像仓库。<br/>所有 GPUStack 业务镜像应放在 `gpustack` 命名空间下；如果需要多级命名空间，应将 `gpustack` 保留为最后一级，例如 `awesome.com/amazing-group/amazing-subgroup/gpustack`。然后将此参数（或 Server/Worker 的 `system_default_container_registry` 配置键，或 `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY` 环境变量）设置为父路径，例如 `awesome.com/amazing-group/amazing-subgroup`。<br/>如果镜像无法放在 `gpustack` 命名空间下（例如使用 `gpustack-ai`），请相应调整 `--image-repo` 和 `--benchmark-image-repo`。 |
| `--image-repo` value                        | `gpustack/gpustack`                    | 覆盖 GPUStack 容器的默认镜像仓库。值中不要包含标签，GPUStack 会自动追加版本标签。<br/>如果镜像 `gpustack/gpustack:<tag>` 位于 `gpustack` 命名空间之外，例如 `awesome.com/amazing-group/amazing-subgroup/gpustack-ai/gpustack:<tag>`，请先将 `--system-default-container-registry=awesome.com/amazing-group/amazing-subgroup` 指向包含父命名空间的镜像仓库，再将此参数（或 Server/Worker 的 `image_repo` 配置键，或 `GPUSTACK_IMAGE_REPO` 环境变量）设置为 `gpustack-ai/gpustack`。 |
| `--gateway-mode` value                      | `auto`                                 | 网关运行模式。可选值：embedded、in-cluster、external、disabled 或 auto（默认）。                                           |
| `--gateway-kubeconfig` value                | (empty)                                | 网关使用的 kubeconfig 文件路径。仅对 external gateway-mode 有效。                                                       |
| `--gateway-namespace` value                 | `higress-system`                       | 网关组件部署所在的命名空间。                                                                                |
| `--service-discovery-name` value            | (empty)                                | DNS 中的服务发现服务名称。仅在使用服务发现部署到 Kubernetes 时有效。                     |
| `--namespace` value                         | (empty)                                | GPUStack 部署网关路由规则和模型实例所使用的 Kubernetes 命名空间。                                                |

### Server 选项 { #server-options }

| <div style="width:180px">参数</div>              | <div style="width:100px">默认值</div>           | 说明                                                                                                                                                                                                                                                                                                                                   |
|--------------------------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--port` value                                   | `80`                                             | Server 监听端口。                                                                                                                                                                                                                                                                                                                   |
| `--tls-port` value                               | `443`                                            | TLS Server 监听端口。                                                                                                                                                                                                                                                                                                               |
| `--api-port` value                               | `30080`                                          | GPUStack API Server 监听端口。                                                                                                                                                                                                                                                                                                      |
| `--proxy-port` value                             | `30079`                                          | GPUStack 代理 Server 监听端口。                                                                                                                                                                                                                                                                                                    |
| `--database-port` value                          | `5432`                                           | 内置 PostgreSQL 数据库端口。                                                                                                                                                                                                                                                                                                    |
| `--metrics-port` value                           | `10161`                                          | Server 指标暴露端口。                                                                                                                                                                                                                                                                                                                |
| `--disable-metrics`                              | `False`                                          | 禁用 Server 指标。                                                                                                                                                                                                                                                                                                                       |
| `--disable-worker`                               | (empty)                                          | (DEPRECATED) 禁用 GPUStack Server 的内置 Worker。新安装默认不包含内置 Worker，如有需要，可通过 '--enable-worker' 启用。若两个参数均未设置，为保持向后兼容，v2.0.1 之前的旧安装默认启用内置 Worker。 |
| `--enable-worker`                                | `False`                                          | 启用 GPUStack Server 的内置 Worker。                                                                                                                                                                                                                                                                                           |
| `--bootstrap-password` value                     | Auto-generated.                                  | 默认管理员用户的初始密码。                                                                                                                                                                                                                                                                                                  |
| `--database-url` value                           | Embedded PostgreSQL.                             | 数据库 URL。支持 PostgreSQL 13.0+、MySQL 8.0.36+，以及 openGauss（postgresql:// 协议）、OceanBase（mysql:// 协议）等兼容数据库。示例：postgresql://username:password@host:port/dbname 或 mysql://username:password@host:port/dbname                                                                      |
| `--redis-url` value                              | None                                             | 外部 Redis URL，用于需要共享快速状态的功能（LB 插件的共享状态后端）。格式：`redis://host[:port][/db]`。不允许包含凭据（凭据会写入网关 CR），且不支持 TLS（`rediss://`）。                                                                                |
| `--ssl-keyfile` value                            | (empty)                                          | SSL 密钥文件路径。                                                                                                                                                                                                                                                                                                                     |
| `--ssl-certfile` value                           | (empty)                                          | SSL 证书文件路径。                                                                                                                                                                                                                                                                                                             |
| `--ssl-ca-certfile` value                        | (empty)                                          | Worker 验证 GPUStack Server 使用的 PEM CA 证书包路径。Server 证书由私有 CA 签发时，必须设置。                                                                                                                                                                                                      |
| `--force-auth-localhost`                         | (unset)                                          | （已弃用，无作用）localhost 调用始终要求身份认证。为向后兼容保留此选项，将在未来版本中移除。                                                                                                                                                                                                 |
| `--disable-update-check`                         | `False`                                          | 禁用更新检查。                                                                                                                                                                                                                                                                                                                         |
| `--ota-server-url` value                         | (empty)                                          | 官方内容来源使用的 OTA Server URL，索引和文件直接位于该 URL 下，例如 https://ota.example.com/gpustack。默认使用公共 OTA Server；网络无法访问公共服务时，可指向自己的副本，也可通过自己的仓库为内部部署提供内容。此设置同时覆盖三类内容，内容仍标记为 `Official` 并持续自动更新，UI 中的 `Reset to Official Source` 也会回到该地址。参阅 [OTA 内容更新](../user-guide/ota-content-updates.md)。 |
| `--disable-openapi-docs`                         | `False`                                          | 禁用自动生成的 OpenAPI 文档端点（/docs 上的 Swagger UI、/redoc 上的 ReDoc，以及 /openapi.json 上的原始规范）。                                                                                                                                                                                                              |
| `--model-catalog-file` value                     | (empty)                                          | 模型目录文件的路径或 URL。                                                                                                                                                                                                                                                                                                        |
| `--enable-cors`                                  | `False`                                          | 在 Server 上启用跨源资源共享（CORS）。                                                                                                                                                                                                                                                                                    |
| `--allow-credentials`                            | `False`                                          | 允许跨源请求携带 cookie 和凭据。                                                                                                                                                                                                                                                                                       |
| `--allow-origins` value                          | `["*"]`                                          | 允许跨源请求的来源。多个来源需多次指定此参数。示例：`--allow-origins https://example.com --allow-origins https://api.example.com`                                                                                                                                                       |
| `--allow-methods` value                          | `["GET", "POST"]`                                | 允许跨源请求的 HTTP 方法。多个方法需多次指定此参数。示例：`--allow-methods GET --allow-methods POST`                                                                                                                                                                                      |
| `--allow-headers` value                          | `["Authorization", "Content-Type", "X-API-Key"]` | 允许跨源请求的 HTTP 请求头。多个请求头需多次指定此参数。示例：`--allow-headers Authorization --allow-headers X-API-Key --allow-headers Content-Type`                                                                                                                                  |
| `--oidc-issuer` value                            | (empty)                                          | OpenID Connect 签发者 URL。                                                                                                                                                                                                                                                                                                                    |
| `--oidc-client-id` value                         | (empty)                                          | OpenID Connect 客户端 ID。                                                                                                                                                                                                                                                                                                                     |
| `--oidc-client-secret` value                     | (empty)                                          | OpenID Connect 客户端密钥。                                                                                                                                                                                                                                                                                                                 |
| `--oidc-redirect-uri` value                      | (empty)                                          | OIDC 应用中配置的重定向 URI，必须设置为 `<server-url>/auth/oidc/callback`。                                                                                                                                                                                                                                  |
| `--external-auth-post-logout-redirect-key` value | (empty)                                          | 不同 IdP 在注销后重定向时使用的通用参数名称，同时适用于 OIDC 和 SAML。                                                                                                                                                                                                                                 |
| `--oidc-skip-userinfo`                           | `False`                                          | 跳过请求 OIDC userinfo_endpoint，改为尝试直接从请求头解析用户信息。                                                                                                                                                                                                                                          |
| `--oidc-use-userinfo`                            | (empty)                                          | [已弃用] 身份认证后，使用 UserInfo 端点获取用户详情。                                                                                                                                                                                                                                                            |
| `--saml-idp-server-url` value                    | (empty)                                          | SAML 身份提供方 Server URL。                                                                                                                                                                                                                                                                                                            |
| `--saml-idp-entity-id` value                     | (empty)                                          | SAML 身份提供方实体 ID。                                                                                                                                                                                                                                                                                                             |
| `--saml-idp-x509-cert` value                     | (empty)                                          | SAML 身份提供方 X.509 证书。                                                                                                                                                                                                                                                                                                     |
| `--saml-sp-entity-id` value                      | (empty)                                          | SAML 服务提供方实体 ID。                                                                                                                                                                                                                                                                                                              |
| `--saml-sp-acs-url` value                        | (empty)                                          | SAML 服务提供方断言消费服务 URL，必须设置为 `<server-url>/auth/saml/callback`。                                                                                                                                                                                                                                  |
| `--saml-sp-x509-cert` value                      | (empty)                                          | SAML 服务提供方 X.509 证书。                                                                                                                                                                                                                                                                                                      |
| `--saml-sp-private-key` value                    | (empty)                                          | SAML 服务提供方私钥。                                                                                                                                                                                                                                                                                                            |
| `--saml-idp-logout-url` value                    | (empty)                                          | SAML 身份提供方单点注销端点 URL。                                                                                                                                                                                                                                                                                            |
| `--saml-sp-slo-url` value                        | (empty)                                          | SAML 服务提供方单点注销服务回调 URL（例如 `<server-url>/auth/saml/logout/callback`）。                                                                                                                                                                                                                                    |
| `--saml-security` value                          | (empty)                                          | JSON 格式的 SAML 安全设置。                                                                                                                                                                                                                                                                                                        |
| `--cas-server-url` value                         | (empty)                                          | CAS（Central Authentication Service）Server 的基础 URL，例如 `https://cas.example.com/cas`。设置后启用 CAS 登录。                                                                                                                                                                                                                    |
| `--cas-callback-url` value                       | (empty)                                          | 向 CAS Server 注册的 CAS 回调 URL。未设置时，默认为 `<server-url>/auth/cas/callback`。                                                                                                                                                                                                                                     |
| `--cas-validate-endpoint` value                  | `/p3/serviceValidate`                            | CAS 票据验证端点，相对于 `--cas-server-url`。对于 CAS 3.0 之前的 Server，请使用 `/serviceValidate`（不支持属性）。                                                                                                                                                                                                        |
| `--cas-username-attribute` value                 | (empty)                                          | 用作 GPUStack 用户名的 CAS XML 属性名称。未设置时，默认使用 CAS 的 `cas:user` 元素。                                                                                                                                                                                                                                    |
| `--cas-full-name-attribute` value                | (empty)                                          | 用作用户全名的 CAS XML 属性名称。                                                                                                                                                                                                                                                                                        |
| `--cas-avatar-attribute` value                   | (empty)                                          | 用作用户头像 URL 的 CAS XML 属性名称。                                                                                                                                                                                                                                                                                       |
| `--external-auth-provider-name` value            | (empty)                                          | 登录页面 SSO 按钮显示的身份提供方名称，例如 `Okta`。未设置时，默认使用协议名称（OIDC/SAML/CAS）。                                                                                                                                                                                                   |
| `--external-auth-name` value                     | (empty)                                          | 外部身份认证用户信息到用户名的映射，例如 preferred_username。                                                                                                                                                                                                                                                    |
| `--external-auth-full-name` value                | (empty)                                          | 外部身份认证用户信息到用户全名的映射，可组合多个元素，例如 `name` 或 `firstName+lastName`。                                                                                                                                                                                             |
| `--external-auth-avatar-url` value               | (empty)                                          | 外部身份认证用户信息到用户头像 URL 的映射。                                                                                                                                                                                                                                                                     |
| `--external-auth-default-inactive`               | `False`                                          | 设为 True 时，新用户默认处于停用状态。                                                                                                                                                                                                                                                                                           |
| `--external-auth-insecure-skip-tls-verify`       | `False`                                          | 跳过外部身份认证 IdP 握手的 TLS 验证（OIDC/CAS）。仅用于测试自签名 IdP，切勿用于生产环境。                                                                                                                                                                                                     |
| `--server-external-url` value                    | (empty)                                          | 供 Worker 注册使用的 Server 外部 URL。通过云服务提供商创建 Worker 时，必须设置此选项，以确保 Worker 正确连接 Server。                                                                                                                                                            |
| `--trusted-hosts` value                          | (empty)                                          | 反向代理后 `the X-Forwarded-Host` 请求头的允许列表。未设置时，从 `--server-external-url` 推导。两者均未设置时，默认为 `*`，信任所有主机（除非 Server 只能通过可信代理访问，否则不推荐）。                                                                                      |
| `--saml-sp-attribute-prefix` value               | (empty)                                          | SAML 服务提供方属性前缀，用于获取 --external-auth-\* 指定的属性。                                                                                                                                                                                                                                         |
| `--gateway-concurrency` value                    | `16`                                             | 内置网关的并发连接数。                                                                                                                                                                                                                                                                                    |
| `--disable-builtin-observability`                | `False`                                          | 禁用内置 Grafana 和 Prometheus 服务。                                                                                                                                                                                                                                                                                             |
| `--builtin-prometheus-port` value                | `19090`                                          | 内置 Prometheus 服务端口。                                                                                                                                                                                                                                                                                                     |
| `--builtin-grafana-port` value                   | `13000`                                          | 内置 Grafana 服务端口。                                                                                                                                                                                                                                                                                                        |
| `--prometheus-url` value                         | (empty)                                          | 用于查询的外部 Prometheus 基础 URL。未设置时，若内置服务已启用，则使用内置服务。对于使用自有可观测性体系的部署，可设置此选项，以继续使用依赖指标的功能（模型与 PD 仪表盘、带宽自检）。                                                                             |
| `--grafana-url` value                            | (empty)                                          | Grafana 基础 URL，用于仪表盘重定向和代理。必须能从浏览器访问（不能仅为容器内部主机名）。设置后，内置 Grafana 和 Prometheus 将被禁用。仅使用外部 Grafana 时需要。                                                                                                                               |
| `--grafana-worker-dashboard-uid` value           | (empty)                                          | Worker 仪表盘重定向使用的 Grafana 仪表盘 UID。                                                                                                                                                                                                                                                                                         |
| `--grafana-model-dashboard-uid` value            | (empty)                                          | 模型仪表盘重定向使用的 Grafana 仪表盘 UID。                                                                                                                                                                                                                                                                                          |

### Worker 选项 { #worker-options }

| <div style="width:180px">参数</div>      | <div style="width:100px">默认值</div> | 说明                                                                                                                                                                                  |
|------------------------------------------|----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-t` value, `--token` value              | Auto-generated.                        | 注册 Worker 使用的共享密钥。                                                                                                                                                       |
| `-s` value, `--server-url` value         | (empty)                                | 要连接的 Server。                                                                                                                                                                        |
| `--worker-name` value                    | (empty)                                | Worker 节点名称，默认使用主机名。                                                                                                                                        |
| `--worker-ip` value                      | (empty)                                | Worker 节点 IP 地址，默认自动检测。                                                                                                                                     |
| `--disable-worker-metrics`               | `False`                                | 禁用指标。                                                                                                                                                                             |
| `--worker-metrics-port` value            | `10151`                                | 指标暴露端口。                                                                                                                                                                      |
| `--worker-port` value                    | `10150`                                | Worker 监听端口。所有 Worker 应使用相同的值。                                                                                                                          |
| `--service-port-range` value             | `40000-40063`                          | 推理服务端口范围，以 'N1-N2' 格式的字符串指定，包含范围两端。                                                                          |
| `--ray-port-range` value                 | `41000-41999`                          | Ray 服务端口范围（用于 vLLM 分布式部署），以 'N1-N2' 格式的字符串指定，包含范围两端。                                             |
| `--log-dir` value                        | (empty)                                | 日志存储目录。                                                                                                                                                                     |
| `--system-reserved` value                | (empty)                                | 调度时为 Worker 预留的系统资源，单位为 GiB。默认不预留资源，示例：'{\"ram\": 2, \"vram\": 1}'。                                |
| `--enable-hf-transfer`                   | `False`                                | [已弃用] huggingface_hub v1.0 已移除 hf_transfer 支持，此参数不再生效。当前默认下载器为 hf_xet。                                                            |
| `--enable-hf-xet`                        | `False`                                | [已弃用] 启用 Hugging Face Xet 下载模型文件。                                                                                                                          |
| `--worker-ifname` value                  | (empty)                                | Worker 节点的网络接口名称，默认自动检测。                                                                                                                         |
| `--kv-transfer-ifname` value                      | (empty)                                | 预填充与解码分离部署中承载 KV-cache 流量的网络接口。宿主机只有一个可路由接口时自动检测；存在多个接口时必须指定，选错接口会导致引擎公布对端无法访问的地址。 |
| `--proxy-mode` value                     | (empty)                                | 访问模型实例的代理模式：`direct`、`worker` 或 `tunnel`（Worker 主动连接 Server，适用于防火墙或 NAT 后的 Worker）。内置 Worker 默认为 `direct`，独立 Worker 默认为 `worker`。 |
| `--benchmark-image-repo` value           | `gpustack/benchmark-runner`            | 覆盖 GPUStack 基准测试容器的默认镜像仓库。与 `--image-repo` 不同，此值直接使用，因此需包含标签。<br/>如果镜像 `gpustack/benchmark-runner:<tag>` 位于 `gpustack` 命名空间之外，例如 `awesome.com/amazing-group/amazing-subgroup/gpustack-ai/benchmark-runner:<tag>`，请先将 `--system-default-container-registry=awesome.com/amazing-group/amazing-subgroup` 指向包含父命名空间的镜像仓库，再将此参数（或 Worker 的 `benchmark_image_repo` 配置键，或 `GPUSTACK_BENCHMARK_IMAGE_REPO` 环境变量）设置为 `gpustack-ai/benchmark-runner:<tag>`。 |
| `--benchmark-dir` value                  | `<data-dir>/benchmarks`                | 基准测试结果存储目录。                                                                                                                                                        |
| `--benchmark-max-duration-seconds` value | (empty)                                | 基准测试超时前的最长运行时间。为空时不限制。                                                                                                                            |

### 可用环境变量 { #available-environment-variables }

大多数命令行参数也可通过带有 `GPUSTACK_` 前缀的全大写环境变量
设置（例如，可通过 `GPUSTACK_DATA_DIR` 设置 `--data-dir`）。

对于上述命令行参数之外的环境变量，请参阅
[环境变量文档](../environment-variables.md)。

## 配置文件 { #config-file }

启动 GPUStack Server 或 Worker 时，可使用 YAML 格式的配置文件设置启动选项。以下为
完整示例：

```yaml
# Common Options
advertise_address: exposed_server_or_worker_ip
debug: false
data_dir: /path/to/data_dir
cache_dir: /path/to/cache_dir
benchmark_dir: /path/to/benchmark_dir
benchmark_image_repo: gpustack/benchmark-runner
token: your_token
huggingface_token: your_huggingface_token
insecure_tls: false

# Server Options
port: 80
tls_port: 443
api_port: 30080
metrics_port: 10161
disable_metrics: false
enable_worker: false
bootstrap_password: your_admin_password
database_url: postgresql://username:password@host:port/dbname
# database_url: mysql://username:password@host:port/dbname
ssl_keyfile: /path/to/keyfile
ssl_certfile: /path/to/certfile
ssl_ca_certfile: /path/to/ca-bundle.pem
disable_update_check: false
disable_openapi_docs: false
model_catalog_file: /path_or_url/to/model_catalog_file
enable_cors: false
allow_credentials: false
allow_origins: [ "*" ]
allow_methods: [ "GET", "POST" ]
allow_headers: [ "Authorization", "Content-Type", "X-API-Key" ]
oidc_issuer: https://your_oidc_issuer
oidc_client_id: your_oidc_client_id
oidc_client_secret: your_oidc_client_secret
oidc_redirect_uri: http://your_gpustack_server_url/auth/oidc/callback
saml_idp_server_url: https://your_saml_idp_server_url
saml_idp_entity_id: your_saml_idp_entity_id
saml_idp_x509_cert: your_saml_idp_x509_cert_pem
saml_sp_entity_id: your_saml_sp_entity_id
saml_sp_acs_url: http://your_gpustack_server_url/auth/saml/callback
saml_sp_x509_cert: your_saml_sp_x509_cert_pem
saml_sp_private_key: your_saml_sp_private_key_pem
saml_security: '{"wantAssertionsSigned": true, "wantMessagesSigned": true}'
cas_server_url: https://your_cas_server_url
cas_username_attribute: uid
cas_full_name_attribute: displayName
external_auth_provider_name: Okta
external_auth_name: email
external_auth_full_name: name
external_auth_avatar_url: picture
server_external_url: http://your_gpustack_server_url_for_external_access
trusted_hosts: [ "your_reverse_proxy_hostname" ]
disable_builtin_observability: false
builtin_prometheus_port: 19090
builtin_grafana_port: 13000
# Per-plugin overrides for the API gateway, keyed by the plugin's manifest name
# -- which for several of them is not the name of the WasmPlugin resource they
# are deployed as (`transformer` is deployed as `gpustack-header-transformer`,
# `ai-statistics` as `gpustack-ai-statistics`). Keying by the wrong one matches
# nothing, silently.
#
# `url` / `sha256` / `image_pull_policy` say where Envoy pulls the module from,
# replacing the address derived from the bundled plugin manifest. `config` is
# the plugin's own settings and is validated against what that plugin accepts,
# so a field it does not declare fails startup rather than being ignored.
gateway_plugin:
  gpustack-ext-auth:
    config:
      local_auth:
        # Whether the gateway authenticates API keys itself. Turned off, every
        # credential is forwarded to the server instead, as it was before.
        enabled: true
      authz:
        # How long the gateway waits for the server to authorize a request.
        # Sized for a loaded server rather than a healthy one: a server that is
        # down refuses the connection and is detected at once, so this governs
        # only the case of one too busy to answer yet -- where waiting is what
        # keeps a load spike from being read as an outage.
        timeout: 30000
      # Let a caller the gateway authenticated itself through while the server
      # is unreachable -- a rolling upgrade included. Authorization is skipped
      # for the duration; unknown keys and anonymous callers are still refused.
      failure_mode_allow_authenticated: true
  ai-statistics:
    config:
      enable_content_types: [ application/json, text/event-stream ]
  # Fields are the plugin's own camelCase config keys; only whitelisted ones
  # are accepted, so a field the server does not vouch for fails startup
  # instead of being forwarded.
  gpustack-ai-proxy:
    config:
      # Reject tool-call messages that arrive without the assistant message
      # that initiated them ("strict"), or the plugin's default ("off").
      toolCallValidation: strict

# Worker Options
server_url: http://your_gpustack_server_url
worker_name: your_worker_name
worker_ip: 192.168.1.101
disable_worker_metrics: false
worker_metrics_port: 10151
worker_port: 10150
service_port_range: 40000-40063
ray_port_range: 41000-41999
log_dir: /path/to/log_dir
system_reserved:
  ram: 2
  vram: 1
enable_hf_transfer: false
proxy_mode: worker
```
