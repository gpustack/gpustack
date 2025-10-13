# Troubleshooting

<a id="view-gpustack-logs"></a>

## View GPUStack Logs

If you installed GPUStack using Docker or the desktop installer, you can view GPUStack logs with the following commands for the default setup:

=== "Linux"

    ```bash
    docker logs -f gpustack
    ```

=== "macOS"

    ```bash
    tail -200f /var/log/gpustack.log
    ```

=== "Windows"

    ```powershell
    Get-Content "C:\ProgramData\GPUStack\log\gpustack.log" -Tail 200 -Wait
    ```

<a id="configure-log-level"></a>

## Configure Log Level

You can enable the DEBUG log level for `gpustack start` by setting the `--debug` parameter.

You can configure log level of the GPUStack server at runtime by running the following command on the server node:

=== "Linux & macOS"

    ```bash
    curl -X PUT http://localhost/debug/log_level -d "debug"
    ```

    The same applies to GPUStack workers:

    ```bash
    curl -X PUT http://localhost:10150/debug/log_level -d "debug"
    ```

=== "Windows"

    ```powershell
    curl.exe -X PUT http://localhost/debug/log_level -d "debug"
    ```

    The same applies to GPUStack workers:

    ```powershell
    curl.exe -X PUT http://localhost:10150/debug/log_level -d "debug"
    ```

The available log levels are: `trace`, `debug`, `info`, `warning`, `error`, `critical`.

## Reset Admin Password

=== "Linux"

    In case you forgot the admin password, you can reset it by running the following command inside the **server container**:

    ```bash
    gpustack reset-admin-password
    ```

    If the default port has been changed, specify the GPUStack URL using the `--server-url` parameter. It must be run locally on the server and accessed via `localhost`:

    ```bash
    gpustack reset-admin-password --server-url http://localhost:9090
    ```

=== "macOS"

    In case you forgot the admin password, you can reset it by running the following command on the **server node**:

    ```bash
    /Applications/GPUStack.app/Contents/MacOS/gpustack reset-admin-password
    ```

    If the default port has been changed, specify the GPUStack URL using the `--server-url` parameter. It must be run locally on the server and accessed via `localhost`:

    ```bash
    /Applications/GPUStack.app/Contents/MacOS/gpustack reset-admin-password --server-url http://localhost:9090
    ```

=== "Windows"

    In case you forgot the admin password, you can reset it by running the following command on the **server node**:

    ```bash
    & "C:\Program Files\GPUStack\gpustack.exe" reset-admin-password
    ```

    If the default port has been changed, specify the GPUStack URL using the `--server-url` parameter. It must be run locally on the server and accessed via `localhost`:

    ```bash
    & "C:\Program Files\GPUStack\gpustack.exe" reset-admin-password --server-url http://localhost:9090
    ```
