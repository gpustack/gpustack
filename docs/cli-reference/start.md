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

| <div style="width:180px">Flag</div> | <div style="width:100px">Default</div> | Description                                                                                                                           |
| ----------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `--config-file` value               | (empty)                                | Path to the YAML config file.                                                                                                         |
| `-d` value, `--debug` value         | `False`                                | To enable debug mode, the short flag -d is not supported in Windows because this flag is reserved by PowerShell for CommonParameters. |
| `--data-dir` value                  | (empty)                                | Directory to store data. Default is OS specific.                                                                                      |
| `--cache-dir` value                 | (empty)                                | Directory to store cache (e.g., model files). Defaults to <data-dir>/cache.                                                           |
| `-t` value, `--token` value         | Auto-generated.                        | Shared secret used to add a worker.                                                                                                   |
| `--huggingface-token` value         | (empty)                                | User Access Token to authenticate to the Hugging Face Hub. Can also be configured via the `HF_TOKEN` environment variable.            |

### Server Options

| <div style="width:180px">Flag</div> | <div style="width:100px">Default</div> | Description                                                                                                                                         |
| ----------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--host` value                      | `0.0.0.0`                              | Host to bind the server to.                                                                                                                         |
| `--port` value                      | `80`                                   | Port to bind the server to.                                                                                                                         |
| `--disable-worker`                  | `False`                                | Disable embedded worker.                                                                                                                            |
| `--bootstrap-password` value        | Auto-generated.                        | Initial password for the default admin user.                                                                                                        |
| `--database-url` value              | `sqlite:///<data-dir>/database.db`     | URL of the database. Example: postgresql://user:password@hostname:port/db_name                                                                      |
| `--ssl-keyfile` value               | (empty)                                | Path to the SSL key file.                                                                                                                           |
| `--ssl-certfile` value              | (empty)                                | Path to the SSL certificate file.                                                                                                                   |
| `--force-auth-localhost`            | `False`                                | Force authentication for requests originating from localhost (127.0.0.1).When set to True, all requests from localhost will require authentication. |
| `--ollama-library-base-url`         | `https://registry.ollama.ai`           | Base URL for the Ollama library.                                                                                                                    |
| `--disable-update-check`            | `False`                                | Disable update check.                                                                                                                               |

### Worker Options

| <div style="width:180px">Flag</div> | <div style="width:100px">Default</div> | Description                                                                                                                                                                                                                                       |
| ----------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-s` value, `--server-url` value    | (empty)                                | Server to connect to.                                                                                                                                                                                                                             |
| `--worker-ip` value                 | (empty)                                | IP address of the worker node. Auto-detected by default.                                                                                                                                                                                          |
| `--disable-metrics`                 | `False`                                | Disable metrics.                                                                                                                                                                                                                                  |
| `--metrics-port` value              | `10151`                                | Port to expose metrics.                                                                                                                                                                                                                           |
| `--worker-port` value               | `10150`                                | Port to bind the worker to. Use a consistent value for all workers.                                                                                                                                                                               |
| `--log-dir` value                   | (empty)                                | Directory to store logs.                                                                                                                                                                                                                          |
| `--system-reserved` value           | `"{\"ram\": 2, \"vram\": 0}"`          | The system reserves resources for the worker during scheduling, measured in GiB. By default, 2 GiB of RAM is reserved, Note: '{\"memory\": 2, \"gpu_memory\": 0}' is also supported, but it is deprecated and will be removed in future releases. |
| `--tools-download-base-url` value   |                                        | Base URL for downloading dependency tools.                                                                                                                                                                                                        |

### Available Environment Variables

Most of the options can be set via environment variables. The environment variables are prefixed with `GPUSTACK_` and are in uppercase. For example, `--data-dir` can be set via the `GPUSTACK_DATA_DIR` environment variable.

Below are additional environment variables that can be set:

| <div style="width:360px">Flag</div> | Description                                              |
| ----------------------------------- | -------------------------------------------------------- |
| `HF_ENDPOINT`                       | Hugging Face Hub endpoint. e.g., `https://hf-mirror.com` |

## Config File

You can configure start options using a YAML-format config file when starting GPUStack server or worker. Here is a complete example:

```yaml
# Common Options
debug: false
data_dir: /path/to/data_dir
cache_dir: /path/to/cache_dir
token: mytoken

# Server Options
host: 0.0.0.0
port: 80
disable_worker: false
database_url: postgresql://user:password@hostname:port/db_name
ssl_keyfile: /path/to/keyfile
ssl_certfile: /path/to/certfile
force_auth_localhost: false
bootstrap_password: myadminpassword
ollama_library_base_url: https://registry.mycompany.com
disable_update_check: false

# Worker Options
server_url: http://myserver
worker_ip: 192.168.1.101
disable_metrics: false
metrics_port: 10151
worker_port: 10150
log_dir: /path/to/log_dir
system_reserved:
  ram: 2
  vram: 0
tools_download_base_url: https://mirror.mycompany.com
```
