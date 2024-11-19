# Installation Script

## Linux and MacOS

You can use the installation script available at `https://get.gpustack.ai` to install GPUStack as a service on systemd and launchd based systems.

You can set additional environment variables and CLI flags when running the script. The following are examples running the installation script with different configurations:

```shell
# Run server.
curl -sfL https://get.gpustack.ai | sh -s -

# Run server without the embedded worker.
curl -sfL https://get.gpustack.ai | sh -s - --disable-worker

# Run server with TLS.
curl -sfL https://get.gpustack.ai | sh -s - --ssl-keyfile /path/to/keyfile --ssl-certfile /path/to/certfile

# Run server with external postgresql database.
curl -sfL https://get.gpustack.ai | sh -s - --database-url "postgresql://username:password@host:port/database_name"

# Run worker with specified IP.
curl -sfL https://get.gpustack.ai | sh -s - --server-url http://myserver --token mytoken --worker-ip 192.168.1.100

# Install with a custom index URL.
curl -sfL https://get.gpustack.ai | INSTALL_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple sh -s -

# Install a custom wheel package other than releases form pypi.org.
curl -sfL https://get.gpustack.ai | INSTALL_PACKAGE_SPEC=https://repo.mycompany.com/my-gpustack.whl sh -s -
```

## Windows

You can use the installation script available at `https://get.gpustack.ai` to install GPUStack as a service on Windows Service Manager.

You can set additional environment variables and CLI flags when running the script. The following are examples running the installation script with different configurations:

```powershell
# Run server.
Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content

# Run server without the embedded worker.
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --disable-worker"

# Run server with TLS.
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --ssl-keyfile 'C:\path\to\keyfile' --ssl-certfile 'C:\path\to\certfile'"


# Run server with external postgresql database.
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --database-url 'postgresql://username:password@host:port/database_name'"

# Run worker with specified IP.
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --server-url 'http://myserver' --token 'mytoken' --worker-ip '192.168.1.100'"

# Run worker with customize reserved resource.
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } -- --server-url 'http://myserver' --token 'mytoken' --system-reserved '{""ram"":5, ""vram"":5}'"

# Install with a custom index URL.
$env:INSTALL_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content

# Install a custom wheel package other than releases form pypi.org.
$env:INSTALL_PACKAGE_SPEC = "https://repo.mycompany.com/my-gpustack.whl"
Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content
```

!!! warning

    Avoid using PowerShell ISE as it is not compatible with the installation script.

## Available Environment Variables for the Installation Script

| Name                   | Default    | Description                                                                                                                                                     |
| ---------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `INSTALL_INDEX_URL`    | (empty)    | Base URL of the Python Package Index.                                                                                                                           |
| `INSTALL_PACKAGE_SPEC` | `gpustack` | The package spec to install. It supports PYPI package names, URLs, and local paths. See https://pip.pypa.io/en/stable/cli/pip_install/#pip-install for details. |
| `INSTALL_PRE_RELEASE`  | (empty)    | If set to 1, pre-release packages will be installed.                                                                                                            |

## Set Environment Variables for the GPUStack Service

You can set environment variables for the GPUStack service in an environment file located at:

- **Linux and MacOS**: `/etc/default/gpustack`
- **Windows**: `$env:APPDATA\gpustack\gpustack.env`

The following is an example of the content of the file:

```shell
HF_TOKEN=mytoken
HF_ENDPOINT=https://my-hf-endpoint
```

## Available CLI Flags

The appended CLI flags of the installation script are passed directly as flags for the `gpustack start` command. You can refer to the [CLI Reference](../cli-reference/start.md) for details.

## Install Server

To set up the GPUStack server (the management node), install GPUStack without the `--server-url` flag. By default, the GPUStack server includes an embedded worker. To disable this embedded worker on the server, use the `--disable-worker` flag.

## Install Worker

To form a cluster, you can add GPUStack workers on additional nodes. Install GPUStack with the `--server-url` flag to specify the server' address and the `--token` flag for worker authenticate.

Examples are as follows:

### Linux or MacOS:

```shell
curl -sfL https://get.gpustack.ai | sh -s - --server-url http://myserver --token mytoken
```

In the default setup, you can run the following on the server node to get the token used for adding workers:

```shell
cat /var/lib/gpustack/token
```

### Windows:

```powershell
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } --server-url http://myserver --token mytoken"
```

In the default setup, you can run the following on the server node to get the token used for adding workers:

```powershell
Get-Content -Path "$env:APPDATA\gpustack\token" -Raw
```
