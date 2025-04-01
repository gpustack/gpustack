# Troubleshooting

## View GPUStack Logs

If you installed GPUStack using the installation script or Docker, you can view GPUStack logs with the following commands for the default setup:

=== "Linux"

    ```bash
    tail -200f /var/log/gpustack.log
    ```

=== "macOS"

    ```bash
    tail -200f /var/log/gpustack.log
    ```

=== "Windows"

    ```powershell
    Get-Content "$env:APPDATA\gpustack\log\gpustack.log" -Tail 200 -Wait
    ```

=== "Docker"

    ```bash
    docker logs -f gpustack
    ```

## Configure Log Level

You can enable the DEBUG log level for `gpustack start` by setting the `--debug` parameter.

You can configure log level of the GPUStack server at runtime by running the following command on the server node:

```bash
curl -X PUT http://localhost/debug/log_level -d "debug"
```

The same applies to GPUStack workers:

```bash
curl -X PUT http://localhost:10150/debug/log_level -d "debug"
```

The available log levels are: `trace`, `debug`, `info`, `warning`, `error`, `critical`.

## Reset Admin Password

In case you forgot the admin password, you can reset it by running the following command on the **server** node or inside the **server container**:

```bash
gpustack reset-admin-password
```

If the default port has been changed, specify the GPUStack URL using the `--server-url` parameter. It must be run locally on the server and accessed via `localhost`:

```bash
gpustack reset-admin-password --server-url http://localhost:9090
```
