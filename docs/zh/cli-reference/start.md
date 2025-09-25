---
hide:
  - toc
---

# gpustack start

运行 GPUStack 服务器或工作节点。

```bash
gpustack start [OPTIONS]
```

## 配置项

### 通用选项

| <div style="width:180px">参数</div>       | <div style="width:100px">默认值</div> | 描述                                                                                                                                                                                                                                                                                                   |
|-------------------------------------------|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--config-file` 值                        | （空）                                 | YAML 配置文件路径。                                                                                                                                                                                                                                                                                     |
| `-d` 值，`--debug` 值                     | `False`                                | 启用调试模式。在 Windows 中不支持短标志 -d，因为该标志被 PowerShell 的 CommonParameters 保留。                                                                                                                                                                                                          |
| `--data-dir` 值                           | （空）                                 | 用于存储数据的目录。默认取决于操作系统。                                                                                                                                                                                                                                                                 |
| `--cache-dir` 值                          | （空）                                 | 用于存储缓存（例如模型文件）的目录。默认为 <data-dir>/cache。                                                                                                                                                                                                                                           |
| `-t` 值，`--token` 值                     | 自动生成                                | 用于添加工作节点的共享密钥。                                                                                                                                                                                                                                                                            |
| `--huggingface-token` 值                  | （空）                                 | 用于在 Hugging Face Hub 上进行身份验证的用户访问令牌。也可通过环境变量 `HF_TOKEN` 配置。                                                                                                                                                                                                                |
| `--enable-ray`                            | `False`                                | 启用 Ray 以在多个工作节点上运行分布式 vLLM。仅支持 Linux。                                                                                                                                                                                                                                              |
| `--ray-args` 值                           | （空）                                 | 传递给 Ray 的参数。使用 `=` 以避免 CLI 将 ray-args 识别为 GPUStack 参数。该参数可多次使用以传递参数列表。例如：`--ray-args=--port=6379 --ray-args=--verbose`。更多详情参见 [Ray 文档](https://docs.ray.io/en/latest/cluster/cli.html)。                                                               |
| `--ray-node-manager-port` 值              | `40098`                                | Ray 节点管理器端口。启用 Ray 时使用。                                                                                                                                                                                                                                                                    |
| `--ray-object-manager-port` 值            | `40099`                                | Ray 对象管理器端口。启用 Ray 时使用。                                                                                                                                                                                                                                                                    |
| `--ray-runtime-env-agent-port` 值         | `40100`                                | Ray 运行时环境代理端口。启用 Ray 时使用。                                                                                                                                                                                                                                                                |
| `--ray-dashboard-agent-grpc-port` 值      | `40101`                                | Ray 仪表盘代理 gRPC 监听端口。启用 Ray 时使用。                                                                                                                                                                                                                                                         |
| `--ray-dashboard-agent-listen-port` 值    | `52365`                                | Ray 仪表盘代理 HTTP 监听端口。启用 Ray 时使用。                                                                                                                                                                                                                                                         |
| `--ray-metrics-export-port` 值            | `40103`                                | Ray 指标导出端口。启用 Ray 时使用。                                                                                                                                                                                                                                                                     |

### 服务器选项

| <div style="width:180px">参数</div> | <div style="width:100px">默认值</div> | 描述                                                                                                                                                                                                        |
| ----------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--host` 值                         | `0.0.0.0`                              | 服务器绑定的主机地址。                                                                                                                                                                                      |
| `--port` 值                         | `80`                                   | 服务器绑定的端口。                                                                                                                                                                                          |
| `--disable-worker`                  | `False`                                | 禁用内置工作节点。                                                                                                                                                                                          |
| `--bootstrap-password` 值           | 自动生成                                | 默认管理员用户的初始密码。                                                                                                                                                                                  |
| `--database-url` 值                 | `sqlite:///<data-dir>/database.db`     | 数据库 URL。支持 SQLite、PostgreSQL 13.0+ 和 MySQL 8.0+。示例：postgresql://user:password@host:port/db_name 或 mysql://user:password@host:port/db_name                                                      |
| `--ssl-keyfile` 值                  | （空）                                 | SSL 私钥文件路径。                                                                                                                                                                                          |
| `--ssl-certfile` 值                 | （空）                                 | SSL 证书文件路径。                                                                                                                                                                                          |
| `--force-auth-localhost`            | `False`                                | 对来自本地主机（127.0.0.1）的请求强制认证。设置为 True 时，所有来自本地主机的请求都需要认证。                                                                                                              |
| `--disable-update-check`            | `False`                                | 禁用更新检查。                                                                                                                                                                                              |
| `--disable-openapi-docs`            | `False`                                | 禁用自动生成的 OpenAPI 文档端点（/docs 的 Swagger UI、/redoc 的 ReDoc，以及 /openapi.json 的原始规范）。                                                                                                    |
| `--model-catalog-file` 值           | （空）                                 | 模型目录文件的路径或 URL。                                                                                                                                                                                  |
| `--ray-port` 值                     | `40096`                                | Ray（GCS 服务器）端口。启用 Ray 时使用。                                                                                                                                                                    |
| `--ray-client-server-port` 值       | `40097`                                | Ray Client Server 端口。启用 Ray 时使用。                                                                                                                                                                   |
| `--ray-dashboard-port` 值           | `8265`                                 | Ray 仪表盘端口。启用 Ray 时使用。                                                                                                                                                                           |
| `--enable-cors`                     | `False`                                | 在服务器上启用跨域资源共享（CORS）。                                                                                                                                                                       |
| `--allow-credentials`               | `False`                                | 允许在跨域请求中携带 Cookie 和凭据。                                                                                                                                                                       |
| `--allow-origins` 值                | `["*"]`                                | 允许跨域请求的来源。可多次指定以配置多个来源。例如：`--allow-origins https://example.com --allow-origins https://api.example.com`                                                                          |
| `--allow-methods` 值                | `["GET", "POST"]`                      | 允许在跨域请求中使用的 HTTP 方法。可多次指定以配置多个方法。例如：`--allow-methods GET --allow-methods POST`                                                                                                 |
| `--allow-headers` 值                | `["Authorization", "Content-Type"]`    | 允许在跨域请求中携带的 HTTP 请求头。可多次指定以配置多个请求头。例如：`--allow-headers Authorization --allow-headers Content-Type`                                                                          |
| `--oidc-issuer` 值                  | （空）                                 | OpenID Connect 发行者 URL。                                                                                                                                                                                 |
| `--oidc-client-id` 值               | （空）                                 | OpenID Connect 客户端 ID。                                                                                                                                                                                  |
| `--oidc-client-secret` 值           | （空）                                 | OpenID Connect 客户端密钥。                                                                                                                                                                                 |
| `--oidc-redirect-uri` 值            | （空）                                 | 在 OIDC 应用中配置的重定向 URI。必须设置为 `<server-url>/auth/oidc/callback`。                                                                                                                             |
| `--saml-idp-server-url` 值          | （空）                                 | SAML 身份提供商（IdP）服务器 URL。                                                                                                                                                                         |
| `--saml-idp-entity-id` 值           | （空）                                 | SAML 身份提供商实体 ID。                                                                                                                                                                                   |
| `--saml-idp-x509-cert` 值           | （空）                                 | SAML 身份提供商 X.509 证书。                                                                                                                                                                               |
| `--saml-sp-entity-id` 值            | （空）                                 | SAML 服务提供商（SP）实体 ID。                                                                                                                                                                             |
| `--saml-sp-acs-url` 值              | （空）                                 | SAML 服务提供商断言消费者服务（ACS）URL。必须设置为 `<server-url>/auth/saml/callback`。                                                                                                                    |
| `--saml-sp-x509-cert` 值            | （空）                                 | SAML 服务提供商 X.509 证书。                                                                                                                                                                               |
| `--saml-sp-private-key` 值          | （空）                                 | SAML 服务提供商私钥。                                                                                                                                                                                       |
| `--saml-security` 值                | （空）                                 | SAML 安全设置，JSON 格式。                                                                                                                                                                                 |
| `--external-auth-name` 值           | （空）                                 | 将外部认证的用户信息映射为用户名，例如 preferred_username。                                                                                                                                                 |
| `--external-auth-full-name` 值      | （空）                                 | 将外部认证的用户信息映射为用户全名。可组合多个元素，例如 `name` 或 `firstName+lastName`。                                                                                                                  |
| `--external-auth-avatar-url` 值     | （空）                                 | 将外部认证的用户信息映射为用户头像 URL。                                                                                                                                                                   |

### 工作节点选项

| <div style="width:180px">参数</div> | <div style="width:100px">默认值</div> | 描述                                                                                                                                                                                                                                                                                                                            |
| ----------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-s` 值，`--server-url` 值          | （空）                                 | 要连接的服务器。                                                                                                                                                                                                                                                                                                                 |
| `--worker-name` 值                  | （空）                                 | 工作节点名称。默认使用主机名。                                                                                                                                                                                                                                                                                                   |
| `--worker-ip` 值                    | （空）                                 | 工作节点的 IP 地址。默认自动检测。                                                                                                                                                                                                                                                                                               |
| `--disable-metrics`                 | `False`                                | 禁用指标。                                                                                                                                                                                                                                                                                                                       |
| `--disable-rpc-servers`             | `False`                                | 禁用 RPC 服务器。                                                                                                                                                                                                                                                                                                                |
| `--metrics-port` 值                 | `10151`                                | 用于暴露指标的端口。                                                                                                                                                                                                                                                                                                             |
| `--worker-port` 值                  | `10150`                                | 绑定工作节点的端口。请在所有工作节点上保持一致。                                                                                                                                                                                                                                                                                 |
| `--service-port-range` 值           | `40000-40063`                          | 推理服务的端口范围，字符串形式为“N1-N2”。两端均包含。                                                                                                                                                                                                                                                                            |
| `--rpc-server-port-range` 值        | `40064-40095`                          | llama-box RPC 服务器的端口范围，字符串形式为“N1-N2”。两端均包含。                                                                                                                                                                                                                                                                |
| `--ray-worker-port-range` 值        | `40200-40999`                          | Ray 工作进程的端口范围，字符串形式为“N1-N2”。两端均包含。                                                                                                                                                                                                                                                                        |
| `--log-dir` 值                      | （空）                                 | 日志存储目录。                                                                                                                                                                                                                                                                                                                   |
| `--rpc-server-args` 值              | （空）                                 | 传递给 RPC 服务器的参数。使用 `=` 以避免 CLI 将 rpc-server-args 识别为服务器参数。该参数可多次使用以传递参数列表。例如：`--rpc-server-args=--verbose --rpc-server-args=--log-colors --rpc-server-args="rpc-server-cache-dir /var/lib/gpustack/cache/rpc_server/"`                                                             |
| `--system-reserved` 值              | `"{\"ram\": 2, \"vram\": 1}"`          | 系统在调度期间为工作节点预留的资源，单位为 GiB。默认预留 2 GiB RAM 和 1G VRAM。注意：也支持 '{\"memory\": 2, \"gpu_memory\": 1}'，但该写法已弃用，将在未来版本中移除。                                                                                                                  |
| `--tools-download-base-url` 值      |                                        | 依赖工具的下载基础 URL。                                                                                                                                                                                                                                                                                                         |
| `--enable-hf-transfer`              | `False`                                | 通过 hf_transfer 加速从 Hugging Face Hub 下载。https://huggingface.co/docs/huggingface_hub/v0.29.3/package_reference/environment_variables#hfhubenablehftransfer                                                                                                                         |
| `--enable-hf-xet`                   | `False`                                | 启用使用 Hugging Face Xet 下载模型文件。                                                                                                                                                                                                                                                 |

### 可用环境变量

大多数选项可以通过环境变量设置。环境变量以 `GPUSTACK_` 为前缀并使用大写。例如，`--data-dir` 可以通过环境变量 `GPUSTACK_DATA_DIR` 设置。

以下是可以设置的其他环境变量：

| <div style="width:360px">变量</div> | 描述                                              |
| ----------------------------------- | ------------------------------------------------- |
| `HF_ENDPOINT`                       | Hugging Face Hub 端点。例如：`https://hf-mirror.com` |

以下是带有 `GPUSTACK_` 前缀的特殊环境变量：

| <div style="width:360px">变量</div>        | 描述                                                                 |
| ----------------------------------------- | -------------------------------------------------------------------- |
| `GPUSTACK_DISABLE_DYNAMIC_LINK_LLAMA_BOX` | 默认使用动态链接。将其设置为 `true` 可启用静态链接。                 |

<a id="config-file"></a>

## 配置文件

启动 GPUStack 服务器或工作节点时，可以使用 YAML 格式的配置文件配置启动选项。以下是一个完整示例：

```yaml
# 通用选项
debug: false
data_dir: /path/to/data_dir
cache_dir: /path/to/cache_dir
token: your_token
huggingface_token: your_huggingface_token
enable_ray: false
ray_args: ["--port=6379", "--verbose"]
ray_node_manager_port: 40098
ray_object_manager_port: 40099
ray_runtime_env_agent_port: 40100
ray_dashboard_agent_grpc_port: 40101
ray_dashboard_agent_listen_port: 52365
ray_metrics_export_port: 40103

# 服务器选项
host: 0.0.0.0
port: 80
disable_worker: false
bootstrap_password: your_admin_password
database_url: postgresql://user:password@host:port/db_name
# database_url: mysql://user:password@host:port/db_name
ssl_keyfile: /path/to/keyfile
ssl_certfile: /path/to/certfile
force_auth_localhost: false
disable_update_check: false
disable_openapi_docs: false
model_catalog_file: /path_or_url/to/model_catalog_file
ray_port: 40096
ray_client_server_port: 40097
ray_dashboard_port: 8265
enable_cors: false
allow_credentials: false
allow_origins: ["*"]
allow_methods: ["GET", "POST"]
allow_headers: ["Authorization", "Content-Type"]
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
external_auth_name: email
external_auth_full_name: name
external_auth_avatar_url: picture

# 工作节点选项
server_url: http://your_gpustack_server_url
worker_name: your_worker_name
worker_ip: 192.168.1.101
disable_metrics: false
disable_rpc_servers: false
metrics_port: 10151
worker_port: 10150
service_port_range: 40000-40063
rpc_server_port_range: 40064-40095
ray_worker_port_range: 40200-40999
log_dir: /path/to/log_dir
rpc_server_args: ["--verbose"]
system_reserved:
  ram: 2
  vram: 1
tools_download_base_url: https://mirror.your_company.com
enable_hf_transfer: false
enable_hf_xet: false
```