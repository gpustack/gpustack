---
hide:
  - toc
---

# gpustack start

Run GPUStack server or worker.

```bash
gpustack start [OPTIONS]
```

## Configurations

### Common Options

| <div style="width:180px">Flag</div>         | <div style="width:100px">Default</div> | Description                                                                                                                           |
| ------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `--advertise-address` value                 | (empty)                                | The IP address to expose for external access.<br/>If not set, the system will auto-detect a suitable local IP address.                |
| `--port` value                              | 80                                     | Port to bind the server to.                                                                                                           |
| `--tls-port` value                          | 443                                    | Port to bind the TLS server to.                                                                                                       |
| `--config-file` value                       | (empty)                                | Path to the YAML config file.                                                                                                         |
| `-d` value, `--debug` value                 | `False`                                | To enable debug mode, the short flag -d is not supported in Windows because this flag is reserved by PowerShell for CommonParameters. |
| `--data-dir` value                          | (empty)                                | Directory to store data. Default is OS specific.                                                                                      |
| `--cache-dir` value                         | (empty)                                | Directory to store cache (e.g., model files). Defaults to <data-dir>/cache.                                                           |
| `--huggingface-token` value                 | (empty)                                | User Access Token to authenticate to the Hugging Face Hub. Can also be configured via the `HF_TOKEN` environment variable.            |
| `--ray-node-manager-port` value             | `40098`                                | Port of Ray node manager. Used when Ray is enabled.                                                                                   |
| `--ray-object-manager-port` value           | `40099`                                | Port of Ray object manager. Used when Ray is enabled.                                                                                 |
| `--ray-runtime-env-agent-port` value        | `40100`                                | Port of Ray runtime env agent. Used when Ray is enabled.                                                                              |
| `--ray-dashboard-agent-grpc-port` value     | `40101`                                | Port for Ray dashboard agent gPRC listen. Used when Ray is enabled.                                                                   |
| `--ray-dashboard-agent-listen-port` value   | `52365`                                | Port for Ray dashboard agent HTTP listen. Used when Ray is enabled.                                                                   |
| `--ray-metrics-export-port` value           | `40103`                                | Port for Ray metrics export. Used when Ray is enabled.                                                                                |
| `--bin-dir` value                           | (empty)                                | Directory to store additional binaries, e.g., versioned backend executables.                                                          |
| `--pipx-path` value                         | (empty)                                | Path to the pipx executable, used to install versioned backends.                                                                      |
| `--system-default-container-registry` value | `docker.io`                            | Default container registry for GPUStack to pull system images.                                                                        |
| `--image-name-override` value               | (empty)                                | Override the default image name for the GPUStack container.                                                                           |
| `--image-repo` value                        | `gpustack/gpustack`                    | Override the default image repository for the GPUStack container.                                                                     |
| `--gateway-mode` value                      | `auto`                                 | Gateway running mode. Options: embedded, in-cluster, external, disabled, or auto (default).                                           |
| `--gateway-kubeconfig` value                | (empty)                                | Path to the kubeconfig file for gateway. Only useful for external gateway-mode.                                                       |
| `--gateway-concurrency` value               | `16`                                   | Number of concurrent connections for the gateway.                                                                                     |
| `--service-discovery-name` value            | (empty)                                | The name of the service discovery service in DNS. Only useful when deployed in Kubernetes with service discovery.                     |
| `--namespace` value                         | (empty)                                | Kubernetes namespace for GPUStack to deploy gateway routing rules and model instances.                                                |

### Server Options

| <div style="width:180px">Flag</div> | <div style="width:100px">Default</div> | Description                                                                                                                                                                             |
| ----------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--api-port` value                  | `8080`                                 | Port to bind the gpustack server to.                                                                                                                                                    |
| `--metrics-port` value              | `10161`                                | Port to expose server metrics.                                                                                                                                                          |
| `--disable-metrics`                 | `False`                                | Disable server metrics.                                                                                                                                                                 |
| `--disable-worker`                  | `False`                                | Disable built-in worker.                                                                                                                                                                |
| `--bootstrap-password` value        | Auto-generated.                        | Initial password for the default admin user.                                                                                                                                            |
| `--database-url` value              | Embedded PostgreSQL.                   | URL of the database. Supports PostgreSQL 13.0+, and MySQL 8.0+. Example: postgresql://user:password@host:port/db_name or mysql://user:password@host:port/db_name                        |
| `--ssl-keyfile` value               | (empty)                                | Path to the SSL key file.                                                                                                                                                               |
| `--ssl-certfile` value              | (empty)                                | Path to the SSL certificate file.                                                                                                                                                       |
| `--force-auth-localhost`            | `False`                                | Force authentication for requests originating from localhost (127.0.0.1). When set to True, all requests from localhost will require authentication.                                    |
| `--disable-update-check`            | `False`                                | Disable update check.                                                                                                                                                                   |
| `--disable-openapi-docs`            | `False`                                | Disable autogenerated OpenAPI documentation endpoints (Swagger UI at /docs, ReDoc at /redoc, and the raw spec at /openapi.json).                                                        |
| `--model-catalog-file` value        | (empty)                                | Path or URL to the model catalog file.                                                                                                                                                  |
| `--ray-port` value                  | `40096`                                | Port of Ray (GCS server). Used when Ray is enabled.                                                                                                                                     |
| `--ray-client-server-port` value    | `40097`                                | Port of Ray Client Server. Used when Ray is enabled.                                                                                                                                    |
| `--ray-dashboard-port` value        | `8265`                                 | Port of Ray dashboard. Used when Ray is enabled.                                                                                                                                        |
| `--enable-cors`                     | `False`                                | Enable Cross-Origin Resource Sharing (CORS) on the server.                                                                                                                              |
| `--allow-credentials`               | `False`                                | Allow cookies and credentials in cross-origin requests.                                                                                                                                 |
| `--allow-origins` value             | `["*"]`                                | Origins allowed for cross-origin requests. Specify the flag multiple times for multiple origins. Example: `--allow-origins https://example.com --allow-origins https://api.example.com` |
| `--allow-methods` value             | `["GET", "POST"]`                      | HTTP methods allowed in cross-origin requests. Specify the flag multiple times for multiple methods. Example: `--allow-methods GET --allow-methods POST`                                |
| `--allow-headers` value             | `["Authorization", "Content-Type"]`    | HTTP request headers allowed in cross-origin requests. Specify the flag multiple times for multiple headers. Example: `--allow-headers Authorization --allow-headers Content-Type`      |
| `--oidc-issuer` value               | (empty)                                | OpenID Connect issuer URL.                                                                                                                                                              |
| `--oidc-client-id` value            | (empty)                                | OpenID Connect client ID.                                                                                                                                                               |
| `--oidc-client-secret` value        | (empty)                                | OpenID Connect client secret.                                                                                                                                                           |
| `--oidc-redirect-uri` value         | (empty)                                | The redirect URI configured in your OIDC application. This must be set to `<server-url>/auth/oidc/callback`.                                                                            |
| `--saml-idp-server-url` value       | (empty)                                | SAML Identity Provider server URL.                                                                                                                                                      |
| `--saml-idp-entity-id` value        | (empty)                                | SAML Identity Provider entity ID.                                                                                                                                                       |
| `--saml-idp-x509-cert` value        | (empty)                                | SAML Identity Provider X.509 certificate.                                                                                                                                               |
| `--saml-sp-entity-id` value         | (empty)                                | SAML Service Provider entity ID.                                                                                                                                                        |
| `--saml-sp-acs-url` value           | (empty)                                | SAML Service Provider Assertion Consumer Service URL. This must be set to `<server-url>/auth/saml/callback`.                                                                            |
| `--saml-sp-x509-cert` value         | (empty)                                | SAML Service Provider X.509 certificate.                                                                                                                                                |
| `--saml-sp-private-key` value       | (empty)                                | SAML Service Provider private key.                                                                                                                                                      |
| `--saml-security` value             | (empty)                                | SAML security settings in JSON format.                                                                                                                                                  |
| `--external-auth-name` value        | (empty)                                | Mapping of external authentication user information to username, e.g., preferred_username.                                                                                              |
| `--external-auth-full-name` value   | (empty)                                | Mapping of external authentication user information to user's full name. Multiple elements can be combined, e.g., `name` or `firstName+lastName`.                                       |
| `--external-auth-avatar-url` value  | (empty)                                | Mapping of external authentication user information to user's avatar URL.                                                                                                               |
| `--server-external-url` value       | (empty)                                | The external server URL for worker registration. This option is required when provisioning workers via cloud providers, ensuring that workers can connect to the server correctly.      |
| `--saml-sp-attribute-prefix` value  | (empty)                                | SAML Service Provider attribute prefix, used for fetching attributes specified by --external-auth-\*.                                                                                   |
| `--oidc-use-userinfo`               | `False`                                | Use the UserInfo endpoint to fetch user details after authentication.                                                                                                                   |

### Worker Options

| <div style="width:180px">Flag</div>     | <div style="width:100px">Default</div> | Description                                                                                                                                                                                                                                                      |
| --------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-t` value, `--token` value             | Auto-generated.                        | Shared secret used to register worker.                                                                                                                                                                                                                           |
| `-s` value, `--server-url` value        | (empty)                                | Server to connect to.                                                                                                                                                                                                                                            |
| `--worker-name` value                   | (empty)                                | Name of the worker node. Use the hostname by default.                                                                                                                                                                                                            |
| `--worker-ip` value                     | (empty)                                | Deprecated, use advertise-address instead.                                                                                                                                                                                                                       |
| `--disable-worker-metrics`              | `False`                                | Disable metrics.                                                                                                                                                                                                                                                 |
| `--worker-metrics-port` value           | `10151`                                | Port to expose metrics.                                                                                                                                                                                                                                          |
| `--worker-port` value                   | `10150`                                | Port to bind the worker to. Use a consistent value for all workers.                                                                                                                                                                                              |
| `--service-port-range` value            | `40000-40063`                          | Port range for inference services, specified as a string in the form 'N1-N2'. Both ends of the range are inclusive.                                                                                                                                              |
| `--distributed-worker-port-range` value | `40200-40999`                          | Generic port range for distributed worker processes (e.g., NCCL/TCP communication), specified as a string in the form 'N1-N2'. Both ends of the range are inclusive.                                                                                             |
| `--log-dir` value                       | (empty)                                | Directory to store logs.                                                                                                                                                                                                                                         |
| `--system-reserved` value               | `"{\"ram\": 2, \"vram\": 1}"`          | The system reserves resources for the worker during scheduling, measured in GiB. By default, 2 GiB of RAM and 1G of VRAM is reserved, Note: '{\"memory\": 2, \"gpu_memory\": 1}' is also supported, but it is deprecated and will be removed in future releases. |
| `--tools-download-base-url` value       |                                        | Base URL for downloading dependency tools.                                                                                                                                                                                                                       |
| `--enable-hf-transfer`                  | `False`                                | Enable faster downloads from the Hugging Face Hub using hf_transfer. https://huggingface.co/docs/huggingface_hub/v0.29.3/package_reference/environment_variables#hfhubenablehftransfer                                                                           |
| `--enable-hf-xet`                       | `False`                                | Enable downloading model files using Hugging Face Xet.                                                                                                                                                                                                           |
| `--worker-ifname` value                 | (empty)                                | Network interface name of the worker node. Auto-detected by default.                                                                                                                                                                                             |

### Available Environment Variables

Most of the options can be set via environment variables. The environment variables are prefixed with `GPUSTACK_` and are in uppercase. For example, `--data-dir` can be set via the `GPUSTACK_DATA_DIR` environment variable.

Below are additional environment variables that can be set:

| <div style="width:360px">Flag</div> | Description                                              |
| ----------------------------------- | -------------------------------------------------------- |
| `HF_ENDPOINT`                       | Hugging Face Hub endpoint. e.g., `https://hf-mirror.com` |

Below are prefixed with `GPUSTACK_` special environment variables that can be set:

| <div style="width:360px">Flag</div>       | Description                                                                        |
| ----------------------------------------- | ---------------------------------------------------------------------------------- |
| `GPUSTACK_DISABLE_DYNAMIC_LINK_LLAMA_BOX` | Dynamic linking is used by default. Setting this to `true` enables static linking. |

## Config File

You can configure start options using a YAML-format config file when starting GPUStack server or worker. Here is a complete example:

```yaml
# Common Options
port: 80
tls_port: 443
advertise_address: exposed_server_or_worker_ip
debug: false
data_dir: /path/to/data_dir
cache_dir: /path/to/cache_dir
token: your_token
huggingface_token: your_huggingface_token
ray_node_manager_port: 40098
ray_object_manager_port: 40099
ray_runtime_env_agent_port: 40100
ray_dashboard_agent_grpc_port: 40101
ray_dashboard_agent_listen_port: 52365
ray_metrics_export_port: 40103

# Server Options
api_port: 8080
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
server_external_url: http://your_gpustack_server_url_for_external_access

# Worker Options
server_url: http://your_gpustack_server_url
worker_name: your_worker_name
worker_ip: 192.168.1.101
disable_metrics: false
worker_metrics_port: 10151
worker_port: 10150
service_port_range: 40000-40063
distributed_worker_port_range: 40200-40999
log_dir: /path/to/log_dir
system_reserved:
  ram: 2
  vram: 1
tools_download_base_url: https://mirror.your_company.com
enable_hf_transfer: false
enable_hf_xet: false
```
