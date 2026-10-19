---
translation_status: translated
---

# 可观测性 { #observability }

本文介绍如何使用 Prometheus 和 Grafana 监控 GPUStack Server、Worker 及 LLM 服务的运行时指标。

## 概述 { #overview }

GPUStack 为模型服务和 GPU 资源管理提供全面的指标集。通过集成 Prometheus 和 Grafana，用户可以实时收集、存储和可视化这些指标，从而高效监控和排查问题。

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
