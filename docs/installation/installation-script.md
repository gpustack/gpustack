# Installation Script

## Linux and MacOS

You can use the installation script available at `https://get.gpustack.ai` to install GPUStack as a service on systemd and launchd based systems.

You can set additonal environment viariables and CLI flags when running the script. The followings are examples running the installation script with different configuration:

```shell
# Run server without the embedded worker.
curl -sfL https://get.gpustack.ai | sh -s - --disable-worker

# Run server with TLS.
curl -sfL https://get.gpustack.ai | sh -s - --ssl-keyfile /path/to/keyfile --ssl-certfile /path/to/certfile

# Run worker with specified IP.
curl -sfL https://get.gpustack.ai | sh -s - --server-url http://myserver --token mytoken --worker-ip 192.168.1.100

# Install a custom wheel package other than releases form pypi.org.
curl -sfL https://get.gpustack.ai | INSTALL_PACKAGE_SPEC=https://repo.mycompany.com/my-gpustack.whl sh -s -
```

### Available Environment Variables

| Name                   | Default    | Description                                                                                                                                                     |
| ---------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `INSTALL_PACKAGE_SPEC` | `gpustack` | The package spec to install. It supports PYPI package names, URLs, and local paths. See https://pip.pypa.io/en/stable/cli/pip_install/#pip-install for details. |
| `INSTALL_PRE_RELEASE`  | (empty)    | If set to 1, pre-release packages will be installed.                                                                                                            |

### Available CLI Flags

The appended CLI flags of the installation script are passed directly as flags for the `gpustack start` command. You can refer to the [CLI Reference](../cli-reference/start.md) for details.

## Windows

You can use the installation script available at `https://get.gpustack.ai` to install GPUStack as a service on Windows Service Manager.

The following are examples running the installation script with different configurations, Windows uses "-" for CLI flags in Powershell:

```powershell
# Run server.
Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content

# Run server without the embedded worker.
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -disable-worker"

# Run server with TLS.
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -ssl-keyfile 'C:\path\to\keyfile' -ssl-certfile 'C:\path\to\certfile'"

# Run worker with specified IP.
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -server-url 'http://myserver' -token 'mytoken' -worker-ip '192.168.1.100'"

# Install a custom wheel package other than releases form pypi.org.
$env:INSTALL_PACKAGE_SPEC = "https://repo.mycompany.com/my-gpustack.whl"
Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content
```

### Available Environment Variables

| Name                   | Default    | Description                                                                                                                                                     |
| ---------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `INSTALL_PACKAGE_SPEC` | `gpustack` | The package spec to install. It supports PYPI package names, URLs, and local paths. See https://pip.pypa.io/en/stable/cli/pip_install/#pip-install for details. |
| `INSTALL_PRE_RELEASE`  | (empty)    | If set to 1, pre-release packages will be installed.                                                                                                            |

### Available Install Script Flags

#### Common Flags

| Flag        | Alias          | Default         | Description                                      |
| ----------- | -------------- | --------------- | ------------------------------------------------ |
| -ConfigFile | `-config-file` |                 | Path to the YAML config file.                    |
| -Debug      | /              | `False`         | Enable debug mode.                               |
| -DataDir    | `-data-dir`    |                 | Directory to store data. Default is OS specific. |
| -Token      | `-token`       | Auto-generated. | Shared secret used to add a worker.              |

#### Server Options

| Flag                     | Alias                         | Default         | Description                                                                                                                                         |
| ------------------------ | ----------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| -ServerHost              | `-server-host`                | `0.0.0.0`       | Host to bind the server to.                                                                                                                         |
| -ServerPort              | `-server-port`                | `80`            | Port to bind the server to.                                                                                                                         |
| -DisableWorker           | `-disable-worker`             | `False`         | Disable embedded worker.                                                                                                                            |
| -BootstrapPassword       | `-bootstrap-password`         | Auto-generated. | Initial password for the default admin user.                                                                                                        |
| -SystemReservedMemory    | `-system-reserved-memory`     | 1               | The system reserves memory for each worker during scheduling, measured in GiB. By default, 1 GiB of memory are reserved.                            |
| -SystemReservedGPUMemory | `-system-reserved-gpu-memory` | 1               | The system reserves memory for each GPU during scheduling, measured in GiB. By default, 1 GiB of gpu memory are reserved.                           |
| -SSLKeyFile              | `-ssl-keyfile`                |                 | Path to the SSL key file.                                                                                                                           |
| -SSLCertFile             | `-ssl-certfile`               |                 | Path to the SSL certificate file.                                                                                                                   |
| -ForceAuthLocalhost      | `-force-auth-localhost`       | `False`         | Force authentication for requests originating from localhost (127.0.0.1).When set to True, all requests from localhost will require authentication. |

### Worker Options

| Flag           | Alias             | Default | Description                                              |
| -------------- | ----------------- | ------- | -------------------------------------------------------- |
| -ServerURL     | `-server-url`     |         | Server to connect to.                                    |
| -WorkerIP      | `-worker-ip`      |         | IP address of the worker node. Auto-detected by default. |
| -EnableMetrics | `-enable-metrics` | `1`     | Enable metrics, 1 is for enable, 0 is for disable.       |
| -MetricsPort   | `-metrics-port`   | `10151` | Port to expose metrics.                                  |
| -WorkerPort    | `-worker-port`    | `10150` | Port to bind the worker to.                              |
| -LogDir        | `-log-dir`        |         | Directory to store logs.                                 |

## Run Server

To run a GPUStack server, install GPUStack without the `--server-url` (Linux/MacOS) or `-server-url` (Windows) flag. By default, the GPUStack server also runs a worker.

If you want to run the server without the embedded worker, use the `--disable-worker` (Linux/MacOS) or `-disable-worker` (Windows) flag.

## Add Worker

To add a GPUStack worker, install GPUStack with the `--server-url` (Linux/MacOS) or `-server-url` (Windows) flag to specify the server it should connect to.
