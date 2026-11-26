---
translation_status: translated
---

# 可观测性 { #observability }

本文介绍如何使用 Prometheus 和 Grafana 监控 GPUStack Server、Worker 及 LLM 服务的运行时指标。

## 概述 { #overview }

GPUStack 为模型服务和 GPU 资源管理提供全面的指标集。通过集成 Prometheus 和 Grafana，用户可以实时收集、存储和可视化这些指标，从而高效监控和排查问题。

## 模型原生监控 { #native-model-monitoring }

在 **模型服务 → 部署** 中点击部署名称，可查看概览、监控、实例、GPU 资源和配置。监控分块可独立展开，并共用所选时间、实例和 PD 角色筛选。

- 运行时图表使用推理引擎指标，不以网关数据填补缺失的引擎样本。PD 时延标明实际观测角色，表示本地子请求，不自动等同完整客户端请求。
- 支持的时延统计为 Avg、P50、P95、P99。统计栏覆盖所选完整区间，趋势点使用页面注明的较短速率窗口。缺失观测保留空缺，不补零；未支持、不适用、失败与陈旧状态分别说明。
- GPU 和主机趋势使用带归属快照的稳定身份。升级 Worker 后开始采集新的稳定历史序列；旧 Worker 仍可提供设备的最新上报值。不根据 GPU 序号推测无法验证的旧历史。物理指标包含共享同一设备或主机的其他负载。
- 引擎历史覆盖当前实例及已观测的来源代次；硬件历史覆盖当前关联设备在记录归属下的观测，两者都不代表完整历史放置审计。

模型监控、模型 Dashboard 跳转和基准测试监控要求组织 Owner 或平台管理员权限。GPUStack 同时检查请求的部署或基准测试及关联资源是否可见。

Worker 和 GPU 的监控与 Dashboard 跳转按资源可见性授权。组织成员可以使用这些接口查看自己有权访问的资源。Grafana 和 Prometheus 的代理接口各自校验访问权限。

语义接口为 `GET /v2/models/{id}/metrics` 和 `GET /v2/models/{id}/resource-metrics`，只接受受控筛选，不接受客户端 PromQL。图表、表格和 CSV 导出共用同一份已授权结果。

## 内置可观测性（默认） { #built-in-observability-default }

默认情况下，GPUStack 会启动内嵌的 Prometheus 和 Grafana。可通过以下地址访问：

- **Prometheus**：`http://your_gpustack_server_host_ip/prometheus`
- **Grafana**：`http://your_gpustack_server_host_ip/grafana`

内置 Grafana 已配置匿名 Viewer 访问，并禁用登录表单。管理员凭据默认仍为 `admin` / `grafana`。

## 外部可观测性（可选） { #external-observability-optional }

如果希望使用外部 Prometheus/Grafana 环境，建议使用提供的 Docker Compose 文件：

运行以下命令克隆最新稳定版本：

```bash
LATEST_TAG=$(
    curl -s "https://api.github.com/repos/gpustack/gpustack/releases" \
    | grep '"tag_name"' \
    | sed -E 's/.*"tag_name": "([^"]+)".*/\1/' \
    | grep -Ev 'rc|beta|alpha|preview' \
    | head -1
)
echo "Latest stable release: $LATEST_TAG"
git clone -b "$LATEST_TAG" https://github.com/gpustack/gpustack.git
cd gpustack/docker-compose
```

启动前，将 `GPUSTACK_GRAFANA_URL` 设置为浏览器可访问的 Grafana URL，而不是 `grafana` 这类仅容器内可访问的主机名。

启动外部 Prometheus/Grafana（这会禁用内置环境）：

```bash
sudo docker compose -f docker-compose.external-observability.yaml up -d
```

如果已有外部 Prometheus/Grafana 环境，也可以改为手动配置：

1. **配置 Prometheus 抓取 GPUStack 指标**\
   为 GPUStack 指标端点（默认 `:10161`）和 Worker 发现端点添加目标。例如：

   ```yaml
   scrape_configs:
     - job_name: gpustack-worker-discovery
       scrape_interval: 5s
       http_sd_configs:
         - url: "http://<gpustack_server_host>:10161/metrics/targets"
           refresh_interval: 1m
     - job_name: gpustack-server
       scrape_interval: 10s
       static_configs:
         - targets: ["<gpustack_server_host>:10161"]
   ```

   `tunnel` 代理模式的 Worker 只能通过服务端正向代理（默认端口 `30079`）访问，该代理会对中继的每个请求进行认证。要抓取这些 Worker 的指标，请添加一个把 GPUStack API 密钥（在 UI 中创建并原样复制——生成的密钥形如 `gpustack_<access_key>_<secret_key>`）作为代理跳凭证的抓取作业：

   ```yaml
     - job_name: gpustack-proxy-worker-discovery
       scrape_interval: 5s
       proxy_url: "http://<gpustack_server_host>:30079"
       authorization:
         credentials: "<gpustack_api_key>"
       http_sd_configs:
         - url: "http://<gpustack_server_host>:10161/metrics/proxy-targets"
           refresh_interval: 1m
   ```
2. **将 GPUStack 仪表盘导入 Grafana**\
   使用 `docker-compose/grafana/grafana_dashboards/` 目录中提供的仪表盘作为起点。
3. **将 GPUStack 指向你的 Grafana**\
   将 `GPUSTACK_GRAFANA_URL` 设置为外部可访问的 Grafana URL，使仪表盘重定向正常工作。此 URL 必须可通过浏览器访问。

## 访问指标 { #accessing-metrics }

- **GPUStack 指标端点**：\
  在 `http://<gpustack_server_host>:10161/metrics` 访问指标。
- **GPUStack Worker 指标目标**：\
  在 `http://<gpustack_server_host>:10161/metrics/targets` 访问指标。
- **Prometheus UI**：\
  默认通过 `http://<host>:19090` 访问 Prometheus，或使用 `--builtin-prometheus-port` / `GPUSTACK_BUILTIN_PROMETHEUS_PORT` 配置的端口。
- **Grafana UI**：\
  默认通过 `http://<host>:13000` 访问 Grafana，或使用 `--builtin-grafana-port` / `GPUSTACK_BUILTIN_GRAFANA_PORT` 配置的端口。内置 Grafana 已配置匿名 Viewer 访问，并禁用登录表单。管理员凭据默认仍为 `admin` / `grafana`。

## 推理日志和追踪 { #inference-logs-and-traces }

通过响应头 `X-GPUStack-Request-Id` 可以查询推理日志和 Trace。配置日志或追踪后，内置网关会添加此响应头。Docker Server 默认将 Trace 上报到内置 Tempo。Helm Server release 默认部署独立的 Tempo 工作负载。日志上报独立配置，需要指定日志端点。

| 追踪目标或配置 | Docker | Helm |
| --- | --- | --- |
| 内置 Tempo | Server 默认启用 | Server release 默认启用 |
| 外部 OTLP/gRPC 接收端 | 设置 `GPUSTACK_TRACE_OTLP_ENDPOINT` | 设置 `tracing.endpoint` |
| 内置 Tempo 的块保留时间（默认 `168h`） | 设置 `GPUSTACK_BUILTIN_TEMPO_RETENTION` | 设置 `tracing.tempo.retention` |
| 禁用追踪 | 设置 `GPUSTACK_DISABLE_TRACING=true` | 设置 `tracing.enabled=false` |

Tempo 会在定期清理时移除过期的块。保留时间控制块的保留期限。磁盘总用量取决于接收的 Trace 数据量，以及预写日志（WAL）、数据压缩合并和待删除块占用的空间。

估算保留块的空间时，用每秒写入的 Trace 数乘以每条 Trace 对应的平均磁盘块字节数，再乘以保留秒数。以演算为例，假设每秒 10 个请求、100% 采样、每个采样请求产生一条 Trace，且每条 Trace 对应的磁盘块平均占 1,000 字节。每天约产生 864 MB 的块，七天约 6.05 GB。若假设再预留 50%，总量约为 9.1 GB。请在有代表性的负载下测量每条 Trace 对应的块字节数，以及 WAL、压缩合并和待删除块的峰值占用，据此确定所需容量和余量。

部署详情中的 `Request Tracing` 标签页初始搜索最近 24 小时。查看 Trace 需要组织 Owner 角色或平台管理员权限。GPUStack 应用与部署指标相同的模型可见性规则。可以逐页查看结果，或粘贴 GPUStack 或 Tempo 中显示的请求 ID 或 Trace ID 进行搜索。如果配置了日志上报，可以从请求详情复制请求 ID 并到日志后端查询。使用外部 Tempo 时，通过 `GPUSTACK_TRACE_QUERY_URL` 或 `tracing.queryURL` 配置查询端点。

启用日志上报后，Server 和 Worker 将日志以单一 JSON 流写入容器控制台。Alloy 同时读取该日志流和受支持格式的引擎日志。要检查丢弃的记录，请查看 Alloy 的 `otelcol_processor_filter_logs_filtered_total` 指标中的 `engine_format` 或 `higress_format` 组件。引擎过滤器会丢弃缺少可识别日志前缀的独立纯文本行。

启用相应的上报且引擎设置允许时，GPUStack 会为使用 vLLM `0.32.0` 或 SGLang `0.5.22` 的兼容 Runner 镜像自动添加引擎可观测性设置。版本检查适用于自动配置。使用其他镜像或版本时，可以通过引擎自身的设置配置遥测。

每个追踪组件直接将 Span 发送到选定的接收端。上报日志的组件必须能访问日志端点。上报 Trace 的 Server、Worker、网关和推理容器必须能访问追踪端点。写入连接使用明文 gRPC，请使用可信网络中的地址。切换到外部接收端或禁用追踪时，内置 Tempo 会停止。

### 配置 Docker { #configure-docker }

1. 若要启用日志上报，请在 Server 和每个上报日志的独立 Worker 上设置 `GPUSTACK_LOG_OTLP_ENDPOINT=http://<collector-host>:4318/v1/logs`。挂载 `/var/run/docker.sock:/var/run/docker.sock`，并使用支持 `docker logs` 的日志驱动。
2. 若要将 Trace 上报到外部接收端，请在 Server 和每个独立 Worker 上设置 `GPUSTACK_TRACE_OTLP_ENDPOINT=<collector-host>:4317`。如需让远程 Worker 使用内置 Tempo，请参见[为现有 Docker Worker 启用追踪](../upgrade/cluster.md#enable-tracing-on-existing-docker-workers)。
3. 将 `<collector-host>` 替换为这些容器可访问的名称或地址。用 `docker run -e` 传入环境变量，或在 Compose 文件的 `environment` 中设置，再重新创建容器。

Docker 在同一个 GPUStack 容器中启动 Tempo 和 Alloy。Tempo 将 WAL 和块存放在 `<data_dir>/tempo` 下，查询仅监听回环地址。Trace 文件的生命周期随 Server 数据卷。若要调整块保留时间，请在 Server 上设置 `GPUSTACK_BUILTIN_TEMPO_RETENTION`，然后重新创建容器。

### 配置 Helm { #configure-helm }

1. 在各 GPUStack release 的 Helm values 中配置端点。设置 `logging.endpoint` 以启用日志上报，设置 `tracing.endpoint` 以使用外部追踪接收端。仅部署 Worker 的 release 可以通过设置 `tracing.endpoint` 使用 Server 的接收端。

   ```yaml
   logging:
     endpoint: "http://<collector-host>:4318/v1/logs"
   tracing:
     endpoint: "<collector-host>:4317"
   ```

2. 将 `<collector-host>` 替换为接收端的地址，再升级 Helm release。如果 Collector 位于其他命名空间，请使用带命名空间的 Service DNS 名称。

内置 Higress 网关默认写入 JSON 访问日志。若要更改日志格式，请在 Helm values 中设置 `higress-core.meshConfig.accessLogEncoding` 和 `higress-core.meshConfig.accessLogFormat`。

Tempo 和 Alloy 镜像使用 `global.hub`，默认值为 `docker.io`。将镜像同步到私有仓库时，请保留 `grafana/tempo` 和 `grafana/alloy` 仓库路径。

Chart 将 Tempo 与 Server Pod 独立部署，默认在 `/var/tempo` 挂载 `emptyDir`。替换 Tempo Pod 时，此卷中的历史 Trace 会丢失。若要在 Pod 替换后保留 Trace，请将 `tracing.tempo.existingClaim` 设为 release 所在命名空间中已有的 PVC。存储卷必须允许 UID 或组 `10001` 写入。若要调整块保留时间，请设置 `tracing.tempo.retention` 并升级 release。 保留时间为空或无效时使用 `168h`。

所有 Server Pod 都使用双向 TLS 通过 `https://<release>-tempo.<namespace>.svc.<cluster-domain>:3200` 查询内置 Tempo，只有一个 Server 的 release 也采用此方式。Chart 默认创建用于查询的 TLS Secret。Worker 和推理引擎通过 Service 的 OTLP/gRPC 端口 `4317` 上报 Trace。

对于 GitOps 部署，在 Server release 所在命名空间创建 Secret，并将名称设为 `tracing.tempo.queryTLSSecretName`。Secret 应包含 `ca.crt`、`server.crt`、`server.key`、`client.crt` 和 `client.key`。服务端证书必须覆盖 `<release>-tempo.<namespace>.svc.<cluster-domain>`。更新服务端和客户端证书时使用同一个 CA。创建新 Secret 并更新 Helm values 中的名称，使 Tempo 和 Server Pod 使用新证书完成更新。

更改 Trace 上报端点后，请重启或重新部署正在运行的模型实例。

### 对上报日志脱敏 { #redact-exported-logs }

Alloy 可以在上报前将日志正文、异常堆栈和 `path` 中匹配的内容替换为 `***`。请提供由非空 RE2 正则表达式组成的列表，空列表会关闭脱敏。请求 ID、Trace ID 和 Span ID 仍可用于关联查询。

在 Docker 中，将 `GPUSTACK_LOG_REDACT_PATTERNS` 作为 JSON 字符串数组传给每个上报日志的容器。下面的示例会掩码电子邮件地址和中国大陆手机号码。

```bash
GPUSTACK_LOG_REDACT_PATTERNS='["(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+[.][A-Z]{2,}","\\b1[3-9][0-9]{9}\\b"]'
```

在 Helm 中，为每个上报日志的 release 设置等效的 YAML 列表。

```yaml
logging:
  redactPatterns:
    - '(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+[.][A-Z]{2,}'
    - '\b1[3-9][0-9]{9}\b'
```

重新创建 Docker 容器或升级 Helm release 以应用规则，并在日志后端按 `request_id` 检查已上报的记录。规则适用于 GPUStack 管理的 Alloy 新上报的日志。通过 `docker logs`、`kubectl logs` 或运行时日志文件仍可访问原始记录。此前已上报的记录和 Trace 不受这些规则影响。无效规则可能导致 Alloy 停止上报日志，GPUStack 仍可处理请求。

### 查询请求 { #look-up-a-request }

1. 向正在运行的聊天模型发送请求并保留响应头。将 `<gpustack-host>`、`<api-key>` 和 `<model-name>` 替换为实际部署的值。推理权限参见 [API 密钥管理](api-key-management.md)。

   ```bash
   curl --silent --show-error --no-buffer \
     --dump-header response-headers.txt \
     'http://<gpustack-host>/v1/chat/completions' \
     --header 'Authorization: Bearer <api-key>' \
     --header 'Content-Type: application/json' \
     --data '{"model":"<model-name>","messages":[{"role":"user","content":"Reply with one word."}],"max_tokens":8,"stream":true}'

   grep -i '^X-GPUStack-Request-Id:' response-headers.txt
   ```

2. 在日志后端按请求发生时间设置搜索范围，并按 `request_id` 日志属性筛选。使用 OTLP 资源属性 `service.name` 可选择 `higress`、`gpustack.server`、`gpustack.engine` 等来源。如果记录包含 `worker_id`、`model_instance_id` 或 `role`，可以进一步缩小范围。
3. 如果日志记录带有 OTLP Trace ID，可在追踪后端按该 ID 查询。启用追踪后，即使关闭日志上报，也可以通过 Span 属性 `gpustack.request_id` 搜索平台请求 ID。

### Loki 和 Grafana 示例 { #example-with-loki-and-grafana }

1. 对于[启用了 OTLP 写入和结构化元数据](https://grafana.com/docs/loki/latest/send-data/otel/)的 Loki，在 `GPUSTACK_LOG_OTLP_ENDPOINT` 或 `logging.endpoint` 中设置完整的写入 URL，例如 `http://<loki-host>:3100/otlp/v1/logs`。将 `GPUSTACK_TRACE_OTLP_ENDPOINT` 或 `tracing.endpoint` 设为 Span 上报组件可访问的 Tempo OTLP/gRPC 地址。在 Grafana 中使用 Loki 和 Tempo 的 HTTP 查询 URL 配置数据源。
2. 在 Grafana 的 `Explore` 中选择 Loki 并设置请求所在的时间范围。将 `<request-id>` 替换为响应头 `X-GPUStack-Request-Id` 的值。Loki 的默认 OTLP 映射将 `service.name` 转换为流标签 `service_name`，并将 `request_id` 保存为结构化元数据。

   ```logql
   {service_name=~"higress|gpustack[.](server|worker|engine)"} | request_id="<request-id>"
   ```

   若记录带有相应字段，可追加 `| worker_id="<worker-id>"` 或 `| model_instance_id="<instance-id>"` 来缩小结果范围。
3. 若结果包含 `trace_id`，可将其复制到 Tempo 数据源的 `TraceQL` 查询字段。若要直接从日志打开 Trace，请配置 Loki 的[派生字段](https://grafana.com/docs/grafana/latest/datasources/loki/configure/#derived-fields)，类型设为 `Label`，匹配 `^trace_id$`，并使用 `${__value.raw}` 作为链接到 Tempo 数据源的查询。

在外部平台配置留存和访问权限。若租户用户可查询其中的数据，请在该平台执行租户隔离。

!!! tip "查找 PD 引擎日志"

    对于 SGLang Mooncake，使用 `request_id` 查找两个引擎的日志，并使用 `pd_attempt_id` 区分派发尝试。对于 vLLM NIXL，先按 `request_id` 查找入口日志，再用其中的 `pd_coordinator_id` 查询两端引擎日志。如果入口日志缺少此字段，可按请求时间和实例缩小 Prefill 日志的范围，确认匹配记录后复制其 `pd_coordinator_id`。

## 从旧版 Compose 配置迁移 { #migration-from-older-compose-setups }

如果此前使用 Docker Compose 在 GPUStack 之外运行 Prometheus/Grafana：

- **保留外部可观测性（建议用于保持连续性）**：\
  继续运行现有 Prometheus/Grafana 容器。将 Prometheus 抓取目标更新为新的 GPUStack 指标端点，并将 `GPUSTACK_GRAFANA_URL` 设置为现有 Grafana。

- **切换到内置可观测性**：\
  停止旧 Prometheus/Grafana 容器，然后使用最新的 `docker-compose.server.yaml`（仅 GPUStack）。内置 Grafana/Prometheus 会接管。旧 Prometheus 的历史指标不会迁移；如需保留访问，可以将旧环境以只读方式保留。

## 自定义指标映射 { #customizing-metrics-mapping }

GPUStack 支持通过配置 API 动态自定义指标映射，无需重启服务即可更新运行时引擎指标到 GPUStack 指标的映射方式。配置集中在 GPUStack Server 上管理，可通过 HTTP API 访问或修改。

### API 端点 { #api-endpoints }

- **获取当前指标配置**

  - GET `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config`
  - 以 JSON 格式返回当前指标映射配置。

- **更新指标配置**

  - POST `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config`
  - 接受 JSON 请求体，更新指标映射配置。更改会立即对所有 Worker 生效。

- **获取默认指标配置**
  - GET `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/default-config`
  - 以 JSON 格式返回默认指标映射配置，便于参考或重置。

### 使用示例 { #example-usage }

**获取当前配置：**

```bash
curl http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config
```

**更新配置：**

```bash
curl -X POST http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config \
     -H "Content-Type: application/json" \
     -d @custom_metrics_config.json
```

_（其中 `custom_metrics_config.json` 是你的新配置文件）_

**获取默认配置：**

```bash
curl http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/default-config
```

> **说明**：配置必须采用有效的 JSON 格式。

## GPUStack 暴露的指标 { #metrics-exposed-by-gpustack }

以下指标由 GPUStack 暴露，可由 Prometheus 抓取。每个指标都包含用于识别集群、Worker、模型和实例的层级标签。

### LLM 服务运行时指标 { #llm-serving-runtime-metrics }

| 指标名称 | 类型 | 描述 |
| --- | --- | --- |
| gpustack:num_requests_running | Gauge | 当前正在处理的请求数。 |
| gpustack:num_requests_waiting | Gauge | 队列中等待的请求数。 |
| gpustack:num_requests_swapped | Gauge | 换出到 CPU 的请求数。 |
| gpustack:prefix_cache_hit_rate | Gauge | 前缀缓存命中率。 |
| gpustack:kv_cache_usage_ratio | Gauge | KV 缓存使用比例，1.0 表示全部使用。 |
| gpustack:prefix_cache_queries | Counter | 前缀缓存查询数（token 总数）。 |
| gpustack:prefix_cache_hits | Counter | 前缀缓存命中数（token 总数）。 |
| gpustack:prompt_tokens | Counter | 已处理的预填充 token 总数。 |
| gpustack:generation_tokens | Counter | 已生成的 token 总数。 |
| gpustack:request_prompt_tokens | Histogram | 每个请求处理的预填充 token 数。 |
| gpustack:request_generation_tokens | Histogram | 每个请求处理的生成 token 数。 |
| gpustack:time_to_first_token_seconds | Histogram | 生成首个 token 的时间。 |
| gpustack:inter_token_latency_seconds | Histogram | 上一个 token 生成后，到生成下一个 token 的时间。 |
| gpustack:time_per_output_token_seconds | Histogram | 每个生成 token 所用的时间。 |
| gpustack:e2e_request_latency_seconds | Histogram | 端到端请求延迟。 |
| gpustack:request_success | Counter | 成功请求总数。 |
| gpustack:request_preemptions | Counter | 引擎发生的抢占总次数。 |

这些指标按照 metrics_config.yaml 中的定义，从不同运行时引擎（vLLM、SGLang、MindIE）映射而来。

### Worker 指标 { #worker-metrics }

| 指标名称 | 类型 | 描述 |
| --- | --- | --- |
| gpustack:worker_status | Gauge | Worker 状态（包含 state 标签）。 |
| gpustack:worker_node_os | Info | Worker 节点的操作系统信息。 |
| gpustack:worker_node_kernel | Info | Worker 节点的内核信息。 |
| gpustack:worker_node_uptime_seconds | Gauge | Worker 节点运行时长，单位为秒。 |
| gpustack:worker_node_cpu_cores | Gauge | Worker 节点的 CPU 总核数。 |
| gpustack:worker_node_cpu_utilization_rate | Gauge | Worker 节点的 CPU 利用率。 |
| gpustack:worker_node_memory_total_bytes | Gauge | Worker 节点内存总量，单位为字节。 |
| gpustack:worker_node_memory_used_bytes | Gauge | Worker 节点已用内存，单位为字节。 |
| gpustack:worker_node_memory_utilization_rate | Gauge | Worker 节点内存利用率。 |
| gpustack:worker_node_gpu | Info | Worker 节点的 GPU 信息。 |
| gpustack:worker_node_gpu_cores | Gauge | Worker 节点的 GPU 总核心数。 |
| gpustack:worker_node_gpu_utilization_rate | Gauge | Worker 节点的 GPU 利用率。 |
| gpustack:worker_node_gpu_temperature_celsius | Gauge | GPU 温度，单位为摄氏度。 |
| gpustack:worker_node_gpu_power_limit_watts | Gauge | GPU 功耗管理上限，单位为瓦。 |
| gpustack:worker_node_gpu_power_used_watts | Gauge | GPU 功耗，单位为瓦。 |
| gpustack:worker_node_gram_total_bytes | Gauge | GPU RAM 总量，单位为字节。 |
| gpustack:worker_node_gram_allocated_bytes | Gauge | 已分配 GPU RAM，单位为字节。 |
| gpustack:worker_node_gram_used_bytes | Gauge | 已用 GPU RAM，单位为字节。 |
| gpustack:worker_node_gram_utilization_rate | Gauge | GPU RAM 利用率。 |
| gpustack:worker_node_filesystem_total_bytes | Gauge | 文件系统总大小，单位为字节。 |
| gpustack:worker_node_filesystem_used_bytes | Gauge | 文件系统已用大小，单位为字节。 |
| gpustack:worker_node_filesystem_utilization_rate | Gauge | 文件系统利用率。 |

### GPUStack Server 指标 { #server-metrics }

| 指标名称 | 类型 | 描述 |
| --- | --- | --- |
| gpustack:cluster | Info | 集群信息（ID、名称、提供商）。 |
| gpustack:cluster_status | Gauge | 集群状态（包含 state 标签）。 |
| gpustack:model | Info | 模型信息（ID、名称、运行时、来源、类别）。 |
| gpustack:model_desired_instances | Gauge | 期望的模型实例数量。 |
| gpustack:model_running_instances | Gauge | 运行中的模型实例数量。 |
| gpustack:model_instance_status | Gauge | 每个模型实例的状态（包含 state 标签）。 |
| gpustack:model_instance_restart_count | Gauge | 模型实例重启次数。 |
| gpustack:model_instance_latest_restart_time | Gauge | 模型实例最近重启时间，以 Unix 时间戳秒数表示。 |

### 事件总线指标 { #event-bus-metrics }

| 指标名称 | 类型 | 描述 |
| --- | --- | --- |
| gpustack:bus_subscribers | Gauge | 每组 `topic` + `source` 的活跃总线订阅者数（多个订阅者可以共享 `source=streaming`，每条打开的 watch 流对应一个订阅者）。 |
| gpustack:bus_queue_depth | Gauge | 共享 `topic`+`source` 的订阅者中的最大队列深度。 |
| gpustack:bus_queue_capacity | Gauge | 每个订阅者的队列 maxsize（参见下方环境变量设置）。 |
| gpustack:bus_queue_full | Gauge | 共享 `topic`+`source` 的任一订阅者队列已满时为 1。 |
| gpustack:bus_queue_saturation_ratio | Gauge | `[0, 1]` 范围内最大的 `qsize / maxsize`。持续 > 0.8 ⇒ 消费者处理缓慢。 |
| gpustack:bus_subscriber_latest_keys | Gauge | 等待合并后 UPDATED 事件投递的 ID 最大数量（`latest_by_key` 的大小）。 |
| gpustack:bus_events_total | Counter | 对共享 `topic`+`source` 的订阅者求和的累计事件数。附加标签：`kind`、`event_type`。 |

> **说明**：所有指标均带有相关标识符标签（集群、Worker、模型、实例、用户），便于细粒度监控和筛选。
