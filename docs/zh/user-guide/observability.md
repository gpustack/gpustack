# 可观测性

本文档介绍如何使用 Prometheus 和 Grafana 监控 GPUStack Server/Worker/LLM 服务运行时指标。

## 概述

GPUStack 提供了用于模型服务和 GPU 资源管理的一整套指标。通过集成 Prometheus 和 Grafana，用户可以实时采集、存储并可视化这些指标，从而实现高效的监控与故障排查。

## GPUStack 暴露的指标

GPUStack 会暴露以下指标，供 Prometheus 抓取。每个指标都包含用于标识集群、工作节点、模型和实例的分层标签。

### LLM 服务运行时指标

| 指标名称                                   | 类型      | 描述                                            |
|-------------------------------------------|-----------|-------------------------------------------------|
| gpustack:num_requests_running             | Gauge     | 当前正在处理的请求数量。                         |
| gpustack:num_requests_waiting             | Gauge     | 队列中等待的请求数量。                           |
| gpustack:num_requests_swapped             | Gauge     | 被换出到 CPU 的请求数量。                        |
| gpustack:prefix_cache_hit_rate            | Gauge     | 前缀缓存命中率。                                 |
| gpustack:kv_cache_usage_ratio             | Gauge     | KV 缓存使用率。1.0 表示已完全使用。              |
| gpustack:prefix_cache_queries             | Counter   | 前缀缓存查询次数（tokens 总数）。                |
| gpustack:prefix_cache_hits                | Counter   | 前缀缓存命中次数（tokens 总数）。                |
| gpustack:prompt_tokens                    | Counter   | 已处理的预填充 tokens 总数。                     |
| gpustack:generation_tokens                | Counter   | 已生成的 tokens 总数。                           |
| gpustack:request_prompt_tokens            | Histogram | 每个请求处理的预填充 tokens 数量。               |
| gpustack:request_generation_tokens        | Histogram | 每个请求处理的生成 tokens 数量。                 |
| gpustack:time_to_first_token_seconds      | Histogram | 生成首个 token 的时间。                          |
| gpustack:time_per_output_token_seconds    | Histogram | 每个生成 token 的耗时。                          |
| gpustack:e2e_request_latency_seconds      | Histogram | 端到端请求延迟。                                 |
| gpustack:request_success                  | Counter   | 成功请求的总数。                                 |

这些指标由各类运行时引擎（vLLM、sglang、ascend-mindie）根据 metrics_config.yaml 中的定义进行映射。

### Worker 指标

| 指标名称                                   | 类型      | 描述                                            |
|-------------------------------------------|-----------|-------------------------------------------------|
| gpustack:worker_status                    | Gauge     | Worker 状态（带 state 标签）。                  |
| gpustack:worker_node_os                   | Info      | 工作节点的操作系统信息。                        |
| gpustack:worker_node_kernel               | Info      | 工作节点的内核信息。                            |
| gpustack:worker_node_uptime_seconds       | Gauge     | 工作节点的运行时长（秒）。                      |
| gpustack:worker_node_cpu_cores            | Gauge     | 工作节点的 CPU 核心总数。                       |
| gpustack:worker_node_cpu_utilization_rate | Gauge     | 工作节点的 CPU 利用率。                         |
| gpustack:worker_node_memory_total_bytes   | Gauge     | 工作节点的内存总量（字节）。                    |
| gpustack:worker_node_memory_used_bytes    | Gauge     | 工作节点的已用内存（字节）。                    |
| gpustack:worker_node_memory_utilization_rate | Gauge  | 工作节点的内存利用率。                          |
| gpustack:worker_node_gpu                  | Info      | 工作节点的 GPU 信息。                           |
| gpustack:worker_node_gpu_cores            | Gauge     | 工作节点的 GPU 核心总数。                       |
| gpustack:worker_node_gpu_utilization_rate | Gauge     | 工作节点的 GPU 利用率。                         |
| gpustack:worker_node_gpu_temperature_celsius | Gauge  | 工作节点的 GPU 温度（摄氏度）。                 |
| gpustack:worker_node_gram_total_bytes     | Gauge     | GPU 内存总量（字节）。                          |
| gpustack:worker_node_gram_allocated_bytes | Gauge     | 已分配的 GPU 内存（字节）。                     |
| gpustack:worker_node_gram_used_bytes      | Gauge     | 已使用的 GPU 内存（字节）。                     |
| gpustack:worker_node_gram_utilization_rate | Gauge    | GPU 内存利用率。                                |
| gpustack:worker_node_filesystem_total_bytes | Gauge   | 文件系统总大小（字节）。                        |
| gpustack:worker_node_filesystem_used_bytes  | Gauge   | 文件系统已用大小（字节）。                      |
| gpustack:worker_node_filesystem_utilization_rate | Gauge | 文件系统利用率。                                |

### Server 指标

| 指标名称                                   | 类型      | 描述                                            |
|-------------------------------------------|-----------|-------------------------------------------------|
| gpustack:cluster                          | Info      | 集群信息（ID、名称、提供方）。                  |
| gpustack:cluster_status                   | Gauge     | 集群状态（带 state 标签）。                     |
| gpustack:model                            | Info      | 模型信息（ID、名称、运行时、来源）。            |
| gpustack:model_desired_instances          | Gauge     | 期望的模型实例数量。                            |
| gpustack:model_running_instances          | Gauge     | 正在运行的模型实例数量。                        |
| gpustack:model_instance_status            | Gauge     | 各模型实例的状态（带 state 标签）。             |

> 注意：所有指标均带有相关标识（cluster、worker、model、instance、user），便于细粒度监控与筛选。

## 使用可观测性栈部署 gpustack

可观测性栈由三大组件组成：

- GPUStack Server：在 `/metrics` 端点暴露指标，在 `/metrics/targets` 暴露 Worker 指标目标。
- Prometheus：从 GPUStack 抓取指标并进行存储。
- Grafana：从 Prometheus 可视化指标。

所有组件均通过 Docker Compose 部署，便于统一管理。

### Docker Compose 配置

下面是一个用于部署 GPUStack Server、Prometheus 和 Grafana 的示例 docker-compose.yaml：

```yaml
version: "3.8"
services:
  gpustack:
    image: gpustack/gpustack
    ports:
      - "80:80"
    restart: always
    ipc: host
    volumes:
      - gpustack-data:/var/lib/gpustack
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
  prometheus:
    image: prom/prometheus
    container_name: gpustack-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--web.enable-remote-write-receiver'
    ports:
      - "9090:9090"
    restart: unless-stopped
    volumes:
      - ./prometheus:/etc/prometheus
      - prom_data:/prometheus
  grafana:
    image: grafana/grafana
    container_name: gpustack-grafana
    ports:
      - "3000:3000"
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=grafana
      - GF_FEATURE_TOGGLES_ENABLE=flameGraph traceqlSearch traceQLStreaming correlations metricsSummary traceqlEditor traceToMetrics traceToProfiles
    volumes:
      - ./grafana/grafana_provisioning:/etc/grafana/provisioning:ro
      - ./grafana/grafana_dashboards:/etc/dashboards:ro
volumes:
  prom_data: {}
  gpustack-data: {}
```

### Prometheus 配置

在 `prometheus.yml` 中添加如下内容，以抓取 GPUStack Server 和 Worker 的指标：

```yaml
scrape_configs:
- job_name: gpustack-workers
  scrape_interval: 5s
  http_sd_configs:
  - url: 'http://gpustack:80/metrics/targets'
    refresh_interval: 1m
- job_name: gpustack-server
  scrape_interval: 10s
  static_configs:
  - targets: ['gpustack:80']
```

### 访问指标

- GPUStack 指标端点：  
  访问 `http://<gpustack_server_host>:80/metrics`
- GPUStack Worker 指标目标：  
  访问 `http://<gpustack_server_host>:80/metrics/targets`
- Prometheus 界面：  
  访问 `http://<host>:9090`
- Grafana 界面：  
  访问 `http://<host>:3000`（默认用户：`admin`，密码：`grafana`）

### 自定义指标映射

GPUStack 通过其配置 API 支持对指标映射进行动态自定义。你可以在不重启服务的情况下，更新运行时引擎指标到 GPUStack 指标的映射。该配置由服务端集中管理，可通过 HTTP API 访问或修改。

#### API 端点

- 获取当前指标配置
    - GET `http://<gpustack_server_host>:80/metrics/config`
    - 以 JSON 格式返回当前的指标映射配置。

- 更新指标配置
    - POST `http://<gpustack_server_host>:80/metrics/config`
    - 接收 JSON 负载以更新指标映射配置。更改会立即对所有 Worker 生效。

- 获取默认指标配置
    - GET `http://<gpustack_server_host>:80/metrics/default-config`
    - 以 JSON 格式返回默认的指标映射配置，便于参考或重置。

#### 使用示例

获取当前配置：

```bash
curl http://<gpustack_server_host>:80/metrics/config
```

更新配置：

```bash
curl -X POST http://<gpustack_server_host>:80/metrics/config \
     -H "Content-Type: application/json" \
     -d @custom_metrics_config.json
```
（其中 `custom_metrics_config.json` 是你的新配置文件）

获取默认配置：

```bash
curl -X POST http://<gpustack_server_host>:80/metrics/default-config
```

> 注意：配置应为合法的 JSON 格式。