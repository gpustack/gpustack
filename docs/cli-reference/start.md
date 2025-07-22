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

| <div style="width:180px">Flag</div> | <div style="width:100px">Default</div> | Description                                                                                                                                                                                                                                                                                           |
| ----------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--config-file` value               | (empty)                                | Path to the YAML config file.                                                                                                                                                                                                                                                                         |
| `-d` value, `--debug` value         | `False`                                | To enable debug mode, the short flag -d is not supported in Windows because this flag is reserved by PowerShell for CommonParameters.                                                                                                                                                                 |
| `--data-dir` value                  | (empty)                                | Directory to store data. Default is OS specific.                                                                                                                                                                                                                                                      |
| `--cache-dir` value                 | (empty)                                | Directory to store cache (e.g., model files). Defaults to <data-dir>/cache.                                                                                                                                                                                                                           |
| `-t` value, `--token` value         | Auto-generated.                        | Shared secret used to add a worker.                                                                                                                                                                                                                                                                   |
| `--huggingface-token` value         | (empty)                                | User Access Token to authenticate to the Hugging Face Hub. Can also be configured via the `HF_TOKEN` environment variable.                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `--enable-ray`                      | `False`                                | Enable Ray for running distributed vLLM across multiple workers. Only supported on Linux.                                                                                                                                                                                                             |
| `--ray-args` value                  | (empty)                                | Arguments to pass to Ray. Use `=` to avoid the CLI recognizing ray-args as a GPUStack argument. This can be used multiple times to pass a list of arguments. Example: `--ray-args=--port=6379 --ray-args=--verbose`. See [Ray docs](https://docs.ray.io/en/latest/cluster/cli.html) for more details. |

### Server Options

| <div style="width:180px">Flag</div> | <div style="width:100px">Default</div> | Description                                                                                                                                                                             |
| ----------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--host` value                      | `0.0.0.0`                              | Host to bind the server to.                                                                                                                                                             |
| `--port` value                      | `80`                                   | Port to bind the server to.                                                                                                                                                             |
| `--disable-worker`                  | `False`                                | Disable built-in worker.                                                                                                                                                                |
| `--bootstrap-password` value        | Auto-generated.                        | Initial password for the default admin user.                                                                                                                                            |
| `--database-url` value              | `sqlite:///<data-dir>/database.db`     | URL of the database. Supports SQLite, PostgreSQL 13.0+, and MySQL 8.0+. Example: postgresql://user:password@hostname:port/db_name or mysql://user:password@host:port/db_name            |
| `--ssl-keyfile` value               | (empty)                                | Path to the SSL key file.                                                                                                                                                               |
| `--ssl-certfile` value              | (empty)                                | Path to the SSL certificate file.                                                                                                                                                       |
| `--force-auth-localhost`            | `False`                                | Force authentication for requests originating from localhost (127.0.0.1).When set to True, all requests from localhost will require authentication.                                     |
| `--disable-update-check`            | `False`                                | Disable update check.                                                                                                                                                                   |
| `--disable-openapi-docs`            | `False`                                | Disable autogenerated OpenAPI documentation endpoints (Swagger UI at /docs, ReDoc at /redoc, and the raw spec at /openapi.json).                                                        |
| `--model-catalog-file` value        | (empty)                                | Path or URL to the model catalog file.                                                                                                                                                  |
| `--ray-port` value                  | `40096`                                | Port of Ray (GCS server). Used when Ray is enabled.                                                                                                                                     |
| `--ray-client-server-port` value    | `40097`                                | Port of Ray Client Server. Used when Ray is enabled.                                                                                                                                    |
| `--enable-cors`                     | `False`                                | Enable Cross-Origin Resource Sharing (CORS) on the server.                                                                                                                              |
| `--allow-credentials`               | `False`                                | Allow cookies and credentials in cross-origin requests.                                                                                                                                 |
| `--allow-origins` value             | `["*"]`                                | Origins allowed for cross-origin requests. Specify the flag multiple times for multiple origins. Example: `--allow-origins https://example.com --allow-origins https://api.example.com` |
| `--allow-methods` value             | `["GET", "POST"]`                      | HTTP methods allowed in cross-origin requests. Specify the flag multiple times for multiple methods. Example: `--allow-methods GET --allow-methods POST`                                |
| `--allow-headers` value             | `["Authorization", "Content-Type"]`    | HTTP request headers allowed in cross-origin requests. Specify the flag multiple times for multiple headers. Example: `--allow-headers Authorization --allow-headers Content-Type`      |

### Worker Options

| <div style="width:180px">Flag</div> | <div style="width:100px">Default</div> | Description                                                                                                                                                                                                                                                      |
| ----------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-s` value, `--server-url` value    | (empty)                                | Server to connect to.                                                                                                                                                                                                                                            |
| `--worker-name` value               | (empty)                                | Name of the worker node. Use the hostname by default.                                                                                                                                                                                                            |
| `--worker-ip` value                 | (empty)                                | IP address of the worker node. Auto-detected by default.                                                                                                                                                                                                         |
| `--disable-metrics`                 | `False`                                | Disable metrics.                                                                                                                                                                                                                                                 |
| `--disable-rpc-servers`             | `False`                                | Disable RPC servers.                                                                                                                                                                                                                                             |
| `--metrics-port` value              | `10151`                                | Port to expose metrics.                                                                                                                                                                                                                                          |
| `--worker-port` value               | `10150`                                | Port to bind the worker to. Use a consistent value for all workers.                                                                                                                                                                                              |
| `--service-port-range` value        | `40000-40063`                          | Port range for inference services, specified as a string in the form 'N1-N2'. Both ends of the range are inclusive.                                                                                                                                              |
| `--rpc-server-port-range` value     | `40064-40095`                          | Port range for llama-box RPC servers, specified as a string in the form 'N1-N2'. Both ends of the range are inclusive.                                                                                                                                           |
| `--ray-node-manager-port` value     | `40098`                                | Port of Ray node manager. Used when Ray is enabled.                                                                                                                                                                                                              |
| `--ray-object-manager-port` value   | `40099`                                | Port of Ray object manager. Used when Ray is enabled.                                                                                                                                                                                                            |
| `--ray-worker-port-range` value     | `40100-40999`                          | Port range for Ray worker processes, specified as a string in the form 'N1-N2'. Both ends of the range are inclusive.                                                                                                                                            |
| `--log-dir` value                   | (empty)                                | Directory to store logs.                                                                                                                                                                                                                                         |
| `--rpc-server-args` value           | (empty)                                | Arguments to pass to the RPC servers. Use `=` to avoid the CLI recognizing rpc-server-args as a server argument. This can be used multiple times to pass a list of arguments. Example: `--rpc-server-args=--verbose --rpc-server-args=--log-colors --rpc-server-args="rpc-server-cache-dir /var/lib/gpustack/cache/rpc_server/"`              |
| `--system-reserved` value           | `"{\"ram\": 2, \"vram\": 1}"`          | The system reserves resources for the worker during scheduling, measured in GiB. By default, 2 GiB of RAM and 1G of VRAM is reserved, Note: '{\"memory\": 2, \"gpu_memory\": 1}' is also supported, but it is deprecated and will be removed in future releases. |
| `--tools-download-base-url` value   |                                        | Base URL for downloading dependency tools.                                                                                                                                                                                                                       |
| `--enable-hf-transfer`              | `False`                                | Enable faster downloads from the Hugging Face Hub using hf_transfer. https://huggingface.co/docs/huggingface_hub/v0.29.3/package_reference/environment_variables#hfhubenablehftransfer                                                                           |

### Available Environment Variables

Most of the options can be set via environment variables. The environment variables are prefixed with `GPUSTACK_` and are in uppercase. For example, `--data-dir` can be set via the `GPUSTACK_DATA_DIR` environment variable.

Below are additional environment variables that can be set:

| <div style="width:360px">Flag</div> | Description                                              |
| ----------------------------------- | -------------------------------------------------------- |
| `HF_ENDPOINT`                       | Hugging Face Hub endpoint. e.g., `https://hf-mirror.com` |

Below are prefixed with `GPUSTACK_` special environment variables that can be set:
| <div style="width:360px">Flag</div>        | Description                                              |
| -----------------------------------        | -------------------------------------------------------- |
| `GPUSTACK_DISABLE_DYNAMIC_LINK_LLAMA_BOX`  | Dynamic linking is used by default. Setting this to `true` enables static linking. |

## Config File

You can configure start options using a YAML-format config file when starting GPUStack server or worker. Here is a complete example:

```yaml
# Common Options
debug: false
data_dir: /path/to/data_dir
cache_dir: /path/to/cache_dir
token: mytoken
enable_ray: false
ray_args: ["--port=6379", "--verbose"]

# Server Options
host: 0.0.0.0
port: 80
disable_worker: false
database_url: postgresql://user:password@hostname:port/db_name
# database_url: mysql://user:password@host:port/db_name
ssl_keyfile: /path/to/keyfile
ssl_certfile: /path/to/certfile
force_auth_localhost: false
bootstrap_password: myadminpassword
disable_update_check: false
disable_openapi_docs: false
model_catalog_file: /path_or_url/to/model_catalog_file
ray_port: 40096
ray_client_server_port: 40097
enable_cors: false
allow_origins: ["*"]
allow_credentials: false
allow_methods: ["GET", "POST"]
allow_headers: ["Authorization", "Content-Type"]

# Worker Options
server_url: http://your_gpustack_server_url
worker_name: myworker
worker_ip: 192.168.1.101
disable_metrics: false
disable_rpc_servers: false
metrics_port: 10151
worker_port: 10150
service_port_range: 40000-40063
rpc_server_port_range: 40064-40095
ray_node_manager_port: 40098
ray_object_manager_port: 40099
ray_worker_port_range: 40100-40999
log_dir: /path/to/log_dir
rpc_server_args: ["--verbose"]
system_reserved:
  ram: 2
  vram: 1
tools_download_base_url: https://mirror.mycompany.com
enable_hf_transfer: false
```
