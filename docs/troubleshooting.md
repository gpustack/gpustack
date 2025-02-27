# Troubleshooting

## View GPUStack Logs

If you installed GPUStack using the installation script, you can view GPUStack logs at the following path:

### Linux or macOS

```bash
/var/log/gpustack.log
```

### Windows

```powershell
"$env:APPDATA\gpustack\log\gpustack.log"
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

The available log levels are:`trace`, `debug`, `info`, `warning`, `error`, `critical`.

## Reset Admin Password

In case you forgot the admin password, you can reset it by running the following command on the **server** node:

```bash
gpustack reset-admin-password
```
