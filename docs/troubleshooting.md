# Troubleshooting

## View GPUStack Logs

If you installed GPUStack using the installation script, you can view GPUStack logs at the following path:

### Linux or MacOS

```bash
/var/log/gpustack.log
```

### Windows

```powershell
"$env:APPDATA\gpustack\log\gpustack.log"
```

## Configure Log Level

You can enable the DEBUG log level on `gpustack start` by setting the `--debug` parameter.

You can configure log level of GPUStack server at runtime by running the following command on the server node:

```bash
curl -X PUT http://localhost/debug/log_level -d "debug"
```
