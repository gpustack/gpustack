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
| `--config-file` value                       | (empty)                                | Path to the YAML config file.                                                                                                         |
| `-d` value, `--debug` value                 | `False`                                | To enable debug mode, the short flag -d is not supported in Windows because this flag is reserved by PowerShell for CommonParameters. |
| `--data-dir` value                          | (empty)                                | Directory to store data. Default is OS specific.                                                                                      |
| `--cache-dir` value                         | (empty)                                | Directory to store cache (e.g., model files). Defaults to <data-dir>/cache.                                                           |
| `--huggingface-token` value                 | (empty)                                | User Access Token to authenticate to the Hugging Face Hub. Can also be configured via the `HF_TOKEN` environment variable.            |
| `--bin-dir` value                           | (empty)                                | Directory to store additional binaries, e.g., versioned backend executables.                                                          |
| `--pipx-path` value                         | (empty)                                | Path to the pipx executable, used to install versioned backends.                                                                      |
| `--system-default-container-registry` value | `docker.io`                            | Default container registry for GPUStack to pull system and inference images.                                                          |
| `--image-name-override` value               | (empty)                                | Override the default image name for the GPUStack container.                                                                           |
| `--image-repo` value                        | `gpustack/gpustack`                    | Override the default image repository for the GPUStack container.                                                                     |
| `--gateway-mode` value                      | `auto`                                 | Gateway running mode. Options: embedded, in-cluster, external, disabled, or auto (default).                                           |
| `--gateway-kubeconfig` value                | (empty)                                | Path to the kubeconfig file for gateway. Only useful for external gateway-mode.                                                       |
| `--gateway-namespace` value                 | `higress-system`                       | The namespace where the gateway component is deployed.                                                                                |
| `--service-discovery-name` value            | (empty)                                | The name of the service discovery service in DNS. Only useful when deployed in Kubernetes with service discovery.                     |
| `--namespace` value                         | (empty)                                | Kubernetes namespace for GPUStack to deploy gateway routing rules and model instances.                                                |

### Server Options

| <div style="width:180px">Flag</div>              | <div style="width:100px">Default</div> | Description                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------ | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--port` value                                   | `80`                                   | Port to bind the server to.                                                                                                                                                                                                                                                                                                                   |
| `--tls-port` value                               | `443`                                  | Port to bind the TLS server to.                                                                                                                                                                                                                                                                                                               |
| `--api-port` value                               | `30080`                                | Port to bind the GPUStack API server to.                                                                                                                                                                                                                                                                                                      |
| `--database-port` value                          | `5432`                                 | Port of the embedded PostgresSQL database.                                                                                                                                                                                                                                                                                                    |
| `--metrics-port` value                           | `10161`                                | Port to expose server metrics.                                                                                                                                                                                                                                                                                                                |
| `--disable-metrics`                              | `False`                                | Disable server metrics.                                                                                                                                                                                                                                                                                                                       |
| `--disable-worker`                               | (empty)                                | (DEPRECATED) Disable the embedded worker for the GPUStack server. New installations will not have the embedded worker by default. Use '--enable-worker' to enable the embedded worker if needed. If neither flag is set, for backward compatibility, the embedded worker will be enabled by default for legacy installations prior to v2.0.1. |
| `--enable-worker`                                | `False`                                | Enable the embedded worker for the GPUStack server.                                                                                                                                                                                                                                                                                           |
| `--bootstrap-password` value                     | Auto-generated.                        | Initial password for the default admin user.                                                                                                                                                                                                                                                                                                  |
| `--database-url` value                           | Embedded PostgreSQL.                   | URL of the database. Supports PostgreSQL 13.0+, and MySQL 8.0+. Example: postgresql://user:password@host:port/db_name or mysql://user:password@host:port/db_name                                                                                                                                                                              |
| `--ssl-keyfile` value                            | (empty)                                | Path to the SSL key file.                                                                                                                                                                                                                                                                                                                     |
| `--ssl-certfile` value                           | (empty)                                | Path to the SSL certificate file.                                                                                                                                                                                                                                                                                                             |
| `--force-auth-localhost`                         | `False`                                | Force authentication for requests originating from localhost (127.0.0.1). When set to True, all requests from localhost will require authentication.                                                                                                                                                                                          |
| `--disable-update-check`                         | `False`                                | Disable update check.                                                                                                                                                                                                                                                                                                                         |
| `--disable-openapi-docs`                         | `False`                                | Disable autogenerated OpenAPI documentation endpoints (Swagger UI at /docs, ReDoc at /redoc, and the raw spec at /openapi.json).                                                                                                                                                                                                              |
| `--model-catalog-file` value                     | (empty)                                | Path or URL to the model catalog file.                                                                                                                                                                                                                                                                                                        |
| `--enable-cors`                                  | `False`                                | Enable Cross-Origin Resource Sharing (CORS) on the server.                                                                                                                                                                                                                                                                                    |
| `--allow-credentials`                            | `False`                                | Allow cookies and credentials in cross-origin requests.                                                                                                                                                                                                                                                                                       |
| `--allow-origins` value                          | `["*"]`                                | Origins allowed for cross-origin requests. Specify the flag multiple times for multiple origins. Example: `--allow-origins https://example.com --allow-origins https://api.example.com`                                                                                                                                                       |
| `--allow-methods` value                          | `["GET", "POST"]`                      | HTTP methods allowed in cross-origin requests. Specify the flag multiple times for multiple methods. Example: `--allow-methods GET --allow-methods POST`                                                                                                                                                                                      |
| `--allow-headers` value                          | `["Authorization", "Content-Type"]`    | HTTP request headers allowed in cross-origin requests. Specify the flag multiple times for multiple headers. Example: `--allow-headers Authorization --allow-headers Content-Type`                                                                                                                                                            |
| `--oidc-issuer` value                            | (empty)                                | OpenID Connect issuer URL.                                                                                                                                                                                                                                                                                                                    |
| `--oidc-client-id` value                         | (empty)                                | OpenID Connect client ID.                                                                                                                                                                                                                                                                                                                     |
| `--oidc-client-secret` value                     | (empty)                                | OpenID Connect client secret.                                                                                                                                                                                                                                                                                                                 |
| `--oidc-redirect-uri` value                      | (empty)                                | The redirect URI configured in your OIDC application. This must be set to `<server-url>/auth/oidc/callback`.                                                                                                                                                                                                                                  |
| `--external-auth-post-logout-redirect-key` value | (empty)                                | Generic parameter name used for post-logout redirection across different IdPs. Applied to both OIDC and SAML.                                                                                                                                                                                                                                 |
| `--oidc-skip-userinfo`                           | `False`                                | Skip requesting the OIDC userinfo_endpoint and instead attempt to parse it directly from the header.                                                                                                                                                                                                                                          |
| `--oidc-use-userinfo`                            | (empty)                                | [Deprecated] Use the UserInfo endpoint to fetch user details after authentication.                                                                                                                                                                                                                                                            |
| `--saml-idp-server-url` value                    | (empty)                                | SAML Identity Provider server URL.                                                                                                                                                                                                                                                                                                            |
| `--saml-idp-entity-id` value                     | (empty)                                | SAML Identity Provider entity ID.                                                                                                                                                                                                                                                                                                             |
| `--saml-idp-x509-cert` value                     | (empty)                                | SAML Identity Provider X.509 certificate.                                                                                                                                                                                                                                                                                                     |
| `--saml-sp-entity-id` value                      | (empty)                                | SAML Service Provider entity ID.                                                                                                                                                                                                                                                                                                              |
| `--saml-sp-acs-url` value                        | (empty)                                | SAML Service Provider Assertion Consumer Service URL. This must be set to `<server-url>/auth/saml/callback`.                                                                                                                                                                                                                                  |
| `--saml-sp-x509-cert` value                      | (empty)                                | SAML Service Provider X.509 certificate.                                                                                                                                                                                                                                                                                                      |
| `--saml-sp-private-key` value                    | (empty)                                | SAML Service Provider private key.                                                                                                                                                                                                                                                                                                            |
| `--saml-idp-logout-url` value                    | (empty)                                | SAML Identity Provider Single Logout endpoint URL.                                                                                                                                                                                                                                                                                            |
| `--saml-sp-slo-url` value                        | (empty)                                | SAML Service Provider Single Logout Service callback URL (e.g., `<server-url>/auth/saml/logout/callback`).                                                                                                                                                                                                                                    |
| `--saml-security` value                          | (empty)                                | SAML security settings in JSON format.                                                                                                                                                                                                                                                                                                        |
| `--external-auth-name` value                     | (empty)                                | Mapping of external authentication user information to username, e.g., preferred_username.                                                                                                                                                                                                                                                    |
| `--external-auth-full-name` value                | (empty)                                | Mapping of external authentication user information to user's full name. Multiple elements can be combined, e.g., `name` or `firstName+lastName`.                                                                                                                                                                                             |
| `--external-auth-avatar-url` value               | (empty)                                | Mapping of external authentication user information to user's avatar URL.                                                                                                                                                                                                                                                                     |
| `--external-auth-default-inactive`               | `False`                                | True if new users should be deactivated by default.                                                                                                                                                                                                                                                                                           |
| `--server-external-url` value                    | (empty)                                | The external server URL for worker registration. This option is required when provisioning workers via cloud providers, ensuring that workers can connect to the server correctly.                                                                                                                                                            |
| `--saml-sp-attribute-prefix` value               | (empty)                                | SAML Service Provider attribute prefix, used for fetching attributes specified by --external-auth-\*.                                                                                                                                                                                                                                         |
| `--gateway-concurrency` value                    | `16`                                   | Number of concurrent connections for the embedded gateway.                                                                                                                                                                                                                                                                                    |
| `--disable-builtin-observability`                | `False`                                | Disable embedded Grafana and Prometheus services.                                                                                                                                                                                                                                                                                             |
| `--grafana-url` value                            | (empty)                                | Grafana base URL for dashboard redirects and proxying. Must be browser-reachable (not a container-only hostname). If set, embedded Grafana and Prometheus will be disabled. Only required for external Grafana.                                                                                                                               |
| `--grafana-worker-dashboard-uid` value           | (empty)                                | Grafana dashboard UID for worker dashboard redirects.                                                                                                                                                                                                                                                                                         |
| `--grafana-model-dashboard-uid` value            | (empty)                                | Grafana dashboard UID for model dashboard redirects.                                                                                                                                                                                                                                                                                          |

### Worker Options

| <div style="width:180px">Flag</div>      | <div style="width:100px">Default</div> | Description                                                                                                                                                                                     |
| ---------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-t` value, `--token` value              | Auto-generated.                        | Shared secret used to register worker.                                                                                                                                                          |
| `-s` value, `--server-url` value         | (empty)                                | Server to connect to.                                                                                                                                                                           |
| `--worker-name` value                    | (empty)                                | Name of the worker node. Use the hostname by default.                                                                                                                                           |
| `--worker-ip` value                      | (empty)                                | IP address of the worker node. Auto-detected by default.                                                                                                                                        |
| `--disable-worker-metrics`               | `False`                                | Disable metrics.                                                                                                                                                                                |
| `--worker-metrics-port` value            | `10151`                                | Port to expose metrics.                                                                                                                                                                         |
| `--worker-port` value                    | `10150`                                | Port to bind the worker to. Use a consistent value for all workers.                                                                                                                             |
| `--service-port-range` value             | `40000-40063`                          | Port range for inference services, specified as a string in the form 'N1-N2'. Both ends of the range are inclusive.                                                                             |
| `--ray-port-range` value                 | `41000-41999`                          | Port range for Ray services(vLLM distributed deployment using), specified as a string in the form 'N1-N2'. Both ends of the range are inclusive.                                                |
| `--log-dir` value                        | (empty)                                | Directory to store logs.                                                                                                                                                                        |
| `--system-reserved` value                | (empty)                                | The system reserves resources for the worker during scheduling, measured in GiB. By default, no resources are reserved, Example: '{\"ram\": 2, \"vram\": 1}'.                                   |
| `--tools-download-base-url` value        |                                        | Base URL for downloading dependency tools.                                                                                                                                                      |
| `--enable-hf-transfer`                   | `False`                                | Enable faster downloads from the Hugging Face Hub using hf_transfer. https://huggingface.co/docs/huggingface_hub/v0.29.3/package_reference/environment_variables#hfhubenablehftransfer          |
| `--enable-hf-xet`                        | `False`                                | Enable downloading model files using Hugging Face Xet.                                                                                                                                          |
| `--worker-ifname` value                  | (empty)                                | Network interface name of the worker node. Auto-detected by default.                                                                                                                            |
| `--proxy-mode` value                     | (empty)                                | Proxy mode for server accessing model instances: direct (server connects directly) or worker (via worker proxy). Default value is direct for embedded worker, and worker for standalone worker. |
| `--benchmark-image-repo` value           | `gpustack/benchmark-runner`            | Override the default benchmark image repo for the GPUStack benchmark container.                                                                                                                 |
| `--benchmark-dir` value                  | `<data-dir>/benchmarks`                | Directory to store benchmark results.                                                                                                                                                           |
| `--benchmark-max-duration-seconds` value | (empty)                                | Max duration for a benchmark before timeout. Disabled when empty.                                                                                                                               |

### Available Environment Variables

Most command line parameters can also be set via environment variables with the `GPUSTACK_` prefix and in uppercase format (e.g., `--data-dir` can be set via `GPUSTACK_DATA_DIR`).

For environment variables beyond the command-line parameters mentioned above, please refer to the [environment variables documentation](../environment-variables.md).

## Config File

You can configure start options using a YAML-format config file when starting GPUStack server or worker. Here is a complete example:

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

# Server Options
port: 80
tls_port: 443
api_port: 30080
metrics_port: 10161
disable_metrics: false
enable_worker: false
bootstrap_password: your_admin_password
database_url: postgresql://user:password@host:port/db_name
# database_url: mysql://user:password@host:port/db_name
ssl_keyfile: /path/to/keyfile
ssl_certfile: /path/to/certfile
force_auth_localhost: false
disable_update_check: false
disable_openapi_docs: false
model_catalog_file: /path_or_url/to/model_catalog_file
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
disable_worker_metrics: false
worker_metrics_port: 10151
worker_port: 10150
service_port_range: 40000-40063
ray_port_range: 41000-41999
log_dir: /path/to/log_dir
system_reserved:
  ram: 2
  vram: 1
tools_download_base_url: https://mirror.your_company.com
enable_hf_transfer: false
enable_hf_xet: false
proxy_mode: worker
```
