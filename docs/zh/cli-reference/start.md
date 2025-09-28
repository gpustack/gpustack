---
hide:
  - toc
---

# gpustack start

运行 GPUStack 服务器或 worker（工作节点）。

```bash
gpustack start [OPTIONS]
```

## 配置

### 通用选项

| <div style="width:180px">Flag</div>       | <div style="width:100px">Default</div> | Description                                                                                                                                                                                                                                                                                           |
|-------------------------------------------|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--config-file` value                     | (empty)                                | YAML 配置文件路径。                                                                                                                                                                                                                                                                                   |
| `-d` value, `--debug` value               | `False`                                | 启用调试模式。在 Windows 上不支持短标志 -d，因为该标志被 PowerShell 的 CommonParameters 保留。                                                                                                                                                                                                          |
| `--data-dir` value                        | (empty)                                | 用于存储数据的目录。默认值因操作系统而异。                                                                                                                                                                                                                                                            |
| `--cache-dir` value                       | (empty)                                | 用于存储缓存（例如，模型文件）的目录。默认值为 <data-dir>/cache。                                                                                                                                                                                                                                     |
| `-t` value, `--token` value               | Auto-generated.                        | 用于添加 worker（工作节点）的共享密钥。                                                                                                                                                                                                                                                               |
| `--huggingface-token` value               | (empty)                                | 用于在 Hugging Face Hub 进行身份验证的用户访问 Token。也可以通过 `HF_TOKEN` 环境变量进行配置。                                                                                                                                                                                                        |
| `--enable-ray`                            | `False`                                | 启用 Ray 以在多个 worker（工作节点）之间运行分布式 vLLM。仅支持 Linux。                                                                                                                                                                                                                               |
| `--ray-args` value                        | (empty)                                | 传递给 Ray 的参数。使用 `=` 以避免 CLI 将 ray-args 识别为 GPUStack 参数。可多次使用以传递参数列表。示例：`--ray-args=--port=6379 --ray-args=--verbose`。更多详情见 [Ray 文档](https://docs.ray.io/en/latest/cluster/cli.html)。                                                                        |
| `--ray-node-manager-port` value           | `40098`                                | Ray 节点管理器的端口。启用 Ray 时使用。                                                                                                                                                                                                                                                               |
| `--ray-object-manager-port` value         | `40099`                                | Ray 对象管理器的端口。启用 Ray 时使用。                                                                                                                                                                                                                                                               |
| `--ray-runtime-env-agent-port` value      | `40100`                                | Ray 运行时环境代理的端口。启用 Ray 时使用。                                                                                                                                                                                                                                                           |
| `--ray-dashboard-agent-grpc-port` value   | `40101`                                | Ray dashboard agent 的 gPRC 监听端口。启用 Ray 时使用。                                                                                                                                                                                                                                               |
| `--ray-dashboard-agent-listen-port` value | `52365`                                | Ray dashboard agent 的 HTTP 监听端口。启用 Ray 时使用。                                                                                                                                                                                                                                               |
| `--ray-metrics-export-port` value         | `40103`                                | Ray 指标导出端口。启用 Ray 时使用。                                                                                                                                                                                                                                                                   |

### 服务器选项

| <div style="width:180px">Flag</div> | <div style="width:100px">Default</div> | Description                                                                                                                                                                             |
| ----------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--host` value                      | `0.0.0.0`                              | 将服务器绑定到的主机。                                                                                                                                                                  |
| `--port` value                      | `80`                                   | 将服务器绑定到的端口。                                                                                                                                                                  |
| `--metrics-port` value              | `10161`                                | 用于公开服务器指标的端口。                                                                                                                                                              |
| `--disable-metrics`                 | `False`                                | 禁用服务器指标。                                                                                                                                                                        |
| `--disable-worker`                  | `False`                                | 禁用内置 worker（工作节点）。                                                                                                                                                           |
| `--bootstrap-password` value        | Auto-generated.                        | 默认管理员用户的初始密码。                                                                                                                                                              |
| `--database-url` value              | `sqlite:///<data-dir>/database.db`     | 数据库的 URL。支持 SQLite、PostgreSQL 13.0+ 和 MySQL 8.0+。示例：postgresql://user:password@host:port/db_name 或 mysql://user:password@host:port/db_name                              |
| `--ssl-keyfile` value               | (empty)                                | SSL 密钥文件路径。                                                                                                                                                                      |
| `--ssl-certfile` value              | (empty)                                | SSL 证书文件路径。                                                                                                                                                                      |
| `--force-auth-localhost`            | `False`                                | 对源自 localhost（127.0.0.1）的请求强制进行身份验证。当设置为 True 时，来自 localhost 的所有请求都需要身份验证。                                                                                                             |
| `--disable-update-check`            | `False`                                | 禁用更新检查。                                                                                                                                                                          |
| `--disable-openapi-docs`            | `False`                                | 禁用自动生成的 OpenAPI 文档端点（/docs 的 Swagger UI、/redoc 的 ReDoc，以及 /openapi.json 的原始规范）。                                                                                 |
| `--model-catalog-file` value        | (empty)                                | 模型目录文件的路径或 URL。                                                                                                                                                              |
| `--ray-port` value                  | `40096`                                | Ray 的端口（GCS 服务器）。启用 Ray 时使用。                                                                                                                                             |
| `--ray-client-server-port` value    | `40097`                                | Ray Client Server 的端口。启用 Ray 时使用。                                                                                                                                             |
| `--ray-dashboard-port` value        | `8265`                                 | Ray dashboard 的端口。启用 Ray 时使用。                                                                                                                                                 |
| `--enable-cors`                     | `False`                                | 在服务器上启用跨域资源共享（CORS）。                                                                                                                                                    |
| `--allow-credentials`               | `False`                                | 允许在跨域请求中使用 Cookie 和凭据。                                                                                                                                                    |
| `--allow-origins` value             | `["*"]`                                | 允许进行跨域请求的来源。可多次指定该标志以配置多个来源。示例：`--allow-origins https://example.com --allow-origins https://api.example.com`                                            |
| `--allow-methods` value             | `["GET", "POST"]`                      | 跨域请求中允许的 HTTP 方法。可多次指定该标志以配置多个方法。示例：`--allow-methods GET --allow-methods POST`                                                                            |
| `--allow-headers` value             | `["Authorization", "Content-Type"]`    | 跨域请求中允许的 HTTP 请求头。可多次指定该标志以配置多个请求头。示例：`--allow-headers Authorization --allow-headers Content-Type`                                                      |
| `--oidc-issuer` value               | (empty)                                | OpenID Connect（OIDC）发行者 URL。                                                                                                                                                      |
| `--oidc-client-id` value            | (empty)                                | OpenID Connect（OIDC）客户端 ID。                                                                                                                                                        |
| `--oidc-client-secret` value        | (empty)                                | OpenID Connect（OIDC）客户端密钥。                                                                                                                                                       |
| `--oidc-redirect-uri` value         | (empty)                                | 在你的 OIDC 应用中配置的重定向 URI。必须设置为 `<server-url>/auth/oidc/callback`。                                                                                                       |
| `--saml-idp-server-url` value | (empty)                                | SAML 身份提供方（IdP）服务器 URL。                                                                                                                                                      |
| `--saml-idp-entity-id` value | (empty)                                | SAML 身份提供方实体 ID。                                                                                                                                                                 |
| `--saml-idp-x509-cert` value | (empty)                                | SAML 身份提供方 X.509 证书。                                                                                                                                                             |
| `--saml-sp-entity-id` value | (empty)                                | SAML 服务提供方（SP）实体 ID。                                                                                                                                                           |
| `--saml-sp-acs-url` value | (empty)                                | SAML 服务提供方的断言消费者服务（Assertion Consumer Service，ACS）URL。必须设置为 `<server-url>/auth/saml/callback`。                                                                  |
| `--saml-sp-x509-cert` value | (empty)                                | SAML 服务提供方 X.509 证书。                                                                                                                                                             |
| `--saml-sp-private-key` value | (empty)                                | SAML 服务提供方私钥。                                                                                                                                                                    |
| `--saml-security` value | (empty)                                | SAML 安全设置（JSON 格式）。                                                                                                                                                             |
| `--external-auth-name` value | (empty)                                | 将外部认证的用户信息映射为用户名，例如 preferred_username。                                                                                                                               |
| `--external-auth-full-name` value | (empty)                                | 将外部认证的用户信息映射为用户全名。可以组合多个元素，例如 `name` 或 `firstName+lastName`。                                                                                              |
| `--external-auth-avatar-url` value | (empty)                                | 将外部认证的用户信息映射为用户头像 URL。                                                                                                                                                 |

### Worker（工作节点）选项

| <div style="width:180px">Flag</div> | <div style="width:100px">Default</div> | Description                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-s` value, `--server-url` value    | (empty)                                | 要连接的服务器。                                                                                                                                                                                                                                                                                                                 |
| `--worker-name` value               | (empty)                                | 工作节点名称。默认使用主机名。                                                                                                                                                                                                                                                                                                   |
| `--worker-ip` value                 | (empty)                                | 工作节点的 IP 地址。默认自动检测。                                                                                                                                                                                                                                                                                               |
| `--disable-worker-metrics`          | `False`                                | 禁用指标。                                                                                                                                                                                                                                                                                                                       |
| `--disable-rpc-servers`             | `False`                                | 禁用 RPC 服务器。                                                                                                                                                                                                                                                                                                                |
| `--worker-metrics-port` value       | `10151`                                | 用于公开指标的端口。                                                                                                                                                                                                                                                                                                             |
| `--worker-port` value               | `10150`                                | 绑定 worker 的端口。对所有 worker 使用一致的值。                                                                                                                                                                                                                                                                                 |
| `--service-port-range` value        | `40000-40063`                          | 推理服务的端口范围，以字符串形式 'N1-N2' 指定。区间两端均包含。                                                                                                                                                                                                                                                                  |
| `--rpc-server-port-range` value     | `40064-40095`                          | llama-box RPC 服务器的端口范围，以字符串形式 'N1-N2' 指定。区间两端均包含。                                                                                                                                                                                                                                                      |
| `--ray-worker-port-range` value     | `40200-40999`                          | Ray worker 进程的端口范围，以字符串形式 'N1-N2' 指定。区间两端均包含。                                                                                                                                                                                                                                                           |
| `--log-dir` value                   | (empty)                                | 用于存储日志的目录。                                                                                                                                                                                                                                                                                                             |
| `--rpc-server-args` value           | (empty)                                | 传递给 RPC 服务器的参数。使用 `=` 以避免 CLI 将 rpc-server-args 识别为服务器参数。可多次使用以传递参数列表。示例：`--rpc-server-args=--verbose --rpc-server-args=--log-colors --rpc-server-args="rpc-server-cache-dir /var/lib/gpustack/cache/rpc_server/"`                                                                   |
| `--system-reserved` value           | `"{\"ram\": 2, \"vram\": 1}"`          | 系统在调度期间为 worker（工作节点）预留资源，单位为 GiB。默认预留 2 GiB 的 RAM 和 1G 的 VRAM。注意：'{\"memory\": 2, \"gpu_memory\": 1}' 也受支持，但已弃用，未来版本将移除。                                                                                                                                                |
| `--tools-download-base-url` value   |                                        | 用于下载依赖工具的基础 URL。                                                                                                                                                                                                                                                                                                     |
| `--enable-hf-transfer`              | `False`                                | 使用 hf_transfer 从 Hugging Face Hub 更快地下载。https://huggingface.co/docs/huggingface_hub/v0.29.3/package_reference/environment_variables#hfhubenablehftransfer                                                                                                                       |
| `--enable-hf-xet`                   | `False`                                | 启用使用 Hugging Face Xet 下载模型文件。                                                                                                                                                                                                                                                                                         |

### 可用环境变量

大多数选项可以通过环境变量进行设置。环境变量以 `GPUSTACK_` 为前缀并使用大写。例如，`--data-dir` 可以通过 `GPUSTACK_DATA_DIR` 环境变量进行设置。

以下是可以设置的其他环境变量：

| <div style="width:360px">Flag</div> | Description                                              |
| ----------------------------------- | -------------------------------------------------------- |
| `HF_ENDPOINT`                       | Hugging Face Hub 终端地址。例如：`https://hf-mirror.com` |

以下是以 `GPUSTACK_` 为前缀的特殊环境变量：

| <div style="width:360px">Flag</div>       | Description                                                                        |
| ----------------------------------------- | ---------------------------------------------------------------------------------- |
| `GPUSTACK_DISABLE_DYNAMIC_LINK_LLAMA_BOX` | 默认使用动态链接。将其设置为 `true` 启用静态链接。                                  |

## 配置文件

在启动 GPUStack 服务器或 worker（工作节点）时，你可以使用 YAML 格式的配置文件来配置启动选项。以下是一个完整示例：

```yaml
# Common Options
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

# Server Options
host: 0.0.0.0
port: 80
metrics_port: 10161
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

# Worker Options
server_url: http://your_gpustack_server_url
worker_name: your_worker_name
worker_ip: 192.168.1.101
disable_metrics: false
disable_rpc_servers: false
worker_metrics_port: 10151
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