# Troubleshooting

## View GPUStack Logs

You can view GPUStack logs with the following commands for the default setup:

```bash
docker logs -f gpustack
```

## Enable Debug Mode

You can enable the `DEBUG` mode by setting the `--debug` flag when running GPUStack:

```diff
sudo docker run -d --name gpustack \
    ...
    gpustack/gpustack \
+    --debug
    ...
```

You can also enable GPUStack's debug mode at runtime by running the following command inside the **server container**:

```bash
gpustack reload-config --set debug=true
```

## Configure Log Level

You can configure log level of the GPUStack server at runtime by running the following command inside the **server container**:

```bash
curl -X PUT http://localhost/debug/log_level -d "debug"
```

The same applies to GPUStack workers:

```bash
curl -X PUT http://localhost:10150/debug/log_level -d "debug"
```

The available log levels are: `trace`, `debug`, `info`, `warning`, `error`, `critical`.

## Reset Admin Password

In case you forgot the admin password, you can reset it by running the following command inside the **server container**:

```bash
gpustack reset-admin-password
```

If you changed the default port using `--port` when starting GPUStack, specify the GPUStack URL using the `--server-url` parameter. It must be run locally on the server and accessed via `localhost`:

```bash
gpustack reset-admin-password --server-url http://localhost:9090
```
