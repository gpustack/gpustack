# Installation Script

## Linux and MacOS

You can use the installation script available at `https://get.gpustack.ai` to install GPUStack as a service on systemd and launchd based systems.

You can set additonal environment viariables and CLI flags when running the script. The followings are examples running the installation script with different configuration:

```shell
# Run server without the embedded worker.
curl -sfL https://get.gpustack.ai | sh -s - --disable-worker

# Run server with external postgresql database.
curl -sfL https://get.gpustack.ai | sh -s - --database-url "postgresql://username:password@host:port/database_name"

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

```
# TODO
```

## Run Server

To run a GPUStack server, install GPUStack without the `--server-url` flag. By default, the GPUStack server also runs a worker.

If you want to run the server without the embedded worker, use the `--disable-worker` flag.

## Add Worker

To add a GPUStack worker, install GPUStack with the `--server-url` flag to specify the server it should connect to.
