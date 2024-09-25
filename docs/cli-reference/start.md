# gpustack start

Run GPUStack server or worker.

```bash
gpustack start [OPTIONS]
```

## Configurations

### Common Options

| Flag                        | Default         | Description                                                                                                                           |
| --------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `--config-file` value       |                 | Path to the YAML config file.                                                                                                         |
| `-d` value, `--debug` value | `False`         | To enable debug mode, the short flag -d is not supported in Windows because this flag is reserved by PowerShell for CommonParameters. |
| `--data-dir` value          |                 | Directory to store data. Default is OS specific.                                                                                      |
| `-t` value, `--token` value | Auto-generated. | Shared secret used to add a worker.                                                                                                   |

### Server Options

| Flag                         | Default                       | Description                                                                                                                                                                                                                                       |
| ---------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--host` value               | `0.0.0.0`                     | Host to bind the server to.                                                                                                                                                                                                                       |
| `--port` value               | `80`                          | Port to bind the server to.                                                                                                                                                                                                                       |
| `--disable-worker`           | `False`                       | Disable embedded worker.                                                                                                                                                                                                                          |
| `--bootstrap-password` value | Auto-generated.               | Initial password for the default admin user.                                                                                                                                                                                                      |
| `--system-reserved` value    | `"{\"ram\": 2, \"vram\": 0}"` | The system reserves resources for the worker during scheduling, measured in GiB. By default, 2 GiB of RAM is reserved, Note: '{\"memory\": 2, \"gpu_memory\": 0}' is also supported, but it is deprecated and will be removed in future releases. |
| `--ssl-keyfile` value        |                               | Path to the SSL key file.                                                                                                                                                                                                                         |
| `--ssl-certfile` value       |                               | Path to the SSL certificate file.                                                                                                                                                                                                                 |
| `--force-auth-localhost`     | `False`                       | Force authentication for requests originating from localhost (127.0.0.1).When set to True, all requests from localhost will require authentication.                                                                                               |
| `--ollama-library-base-url`  | `https://registry.ollama.ai`  | Base URL for the Ollama library.                                                                                                                                                                                                                  |

### Worker Options

| Flag                             | Default | Description                                                         |
| -------------------------------- | ------- | ------------------------------------------------------------------- |
| `-s` value, `--server-url` value |         | Server to connect to.                                               |
| `--worker-ip` value              |         | IP address of the worker node. Auto-detected by default.            |
| `--disable-metrics`              | `False` | Disable metrics.                                                    |
| `--metrics-port` value           | `10151` | Port to expose metrics.                                             |
| `--worker-port` value            | `10150` | Port to bind the worker to. Use a consistent value for all workers. |
| `--log-dir` value                |         | Directory to store logs.                                            |

## Config File

You can configure start options using a YAML-format config file when starting GPUStack server or worker. Here is a complete example:

```yaml
# Common Options
debug: false
data_dir: /path/to/dir
token: mytoken

# Server Options
host: 0.0.0.0
port: 80
disable_worker: false
ssl_keyfile: /path/to/keyfile
ssl_certfile: /path/to/certfile
force_auth_localhost: false
bootstrap_password: myadminpassword
system_reserved:
  ram: 2
  vram: 0
ollama_library_base_url: https://registry.mycompany.com

# Worker Options
server_url: http://myserver
worker_ip: 192.168.1.101
disable_metrics: false
metrics_port: 10151
worker_port: 10150
log_dir: /path/to/dir
```
