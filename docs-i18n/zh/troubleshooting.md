---
translation_status: translated
---

# 故障排查 { #troubleshooting }

## 查看 GPUStack 日志 { #view-gpustack-logs }

默认配置下，可通过以下命令查看 GPUStack 日志：
```bash
docker logs -f gpustack
```

## 启用调试模式 { #enable-debug-mode }

运行 GPUStack 时，可通过 `--debug` 参数启用 `DEBUG` 模式：
```diff
sudo docker run -d --name gpustack \
    ...
    gpustack/gpustack \
+    --debug
    ...
```

也可在运行期间，在 **Server 容器**内执行以下命令启用 GPUStack 调试模式：
```bash
gpustack reload-config --set debug=true
```

`--debug` 启用详细日志，不会启动 Python 内存分配追踪。如需分析内存，请在启动 GPUStack 前设置 `GPUSTACK_DEBUG_ENABLE_TRACEMALLOC=true`。该设置默认为 `false`，独立于 `--debug`，修改后需要重启。内存追踪会增加 CPU 和内存开销，建议在排查内存问题时启用。

## 配置日志级别 { #configure-log-level }

日志级别端点要求身份认证。

可在运行期间，在 **Server 容器**内执行以下命令配置 GPUStack Server 的日志级别。使用管理员 API key 进行身份认证：
```bash
curl -X PUT http://localhost/debug/log_level \
    -H "Authorization: Bearer <YOUR_API_KEY>" \
    -d "debug"
```

GPUStack Worker 同样适用。使用本地 Worker token 进行身份认证（默认路径为 `/var/lib/gpustack/worker_token`）：
```bash
curl -X PUT http://localhost:10150/debug/log_level \
    -H "Authorization: Bearer $(cat /var/lib/gpustack/worker_token)" \
    -d "debug"
```

可用的日志级别为：`trace`、`debug`、`info`、`warning`、`error`、`critical`。

## 重置管理员密码 { #reset-admin-password }

如果忘记管理员密码，可在 **Server 容器**内执行以下命令重置：
```bash
gpustack reset-admin-password
```

如果启动 GPUStack 时通过 `--port` 修改了默认端口，请使用 `--server-url` 参数指定 GPUStack URL。此命令必须在 Server 本机运行，并通过 `localhost` 访问：
```bash
gpustack reset-admin-password --server-url http://localhost:9090
```

## 协助诊断加速器检测 { #assist-in-accelerators-detection-diagnosis }

按照[安装指南](./installation/requirements.md)成功部署 GPUStack Worker 后，\
如果 Worker 未能检测到设备，\
请进入相应 Worker 容器执行以下命令，并将结果反馈给 [GPUStack](https://github.com/gpustack/gpustack/issues)。
```bash
time GPUSTACK_RUNTIME_LOG_LEVEL=debug GPUSTACK_RUNTIME_LOG_EXCEPTION=1 gpustack-runtime detect --format json
```

## 协助诊断模型部署 { #assist-in-model-deployment-diagnosis }

如果部署模型后遇到问题，
请进入相应 Worker 容器执行以下命令，并将结果反馈给 [GPUStack](https://github.com/gpustack/gpustack/issues)。
```bash
gpustack-runtime inspect <model instance name>
```
