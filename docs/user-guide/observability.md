# Observability

This document describes how to monitor GPUStack Server/Worker/LLM serving runtime metrics using Prometheus and Grafana.

## Overview

GPUStack provides a comprehensive set of metrics for model serving and GPU resource management. By integrating Prometheus and Grafana, users can collect, store, and visualize these metrics in real time, enabling efficient monitoring and troubleshooting.

## Metrics Exposed by GPUStack

The following metrics are exposed by GPUStack and can be scraped by Prometheus. Each metric includes hierarchical labels for cluster, worker, model, and instance identification.

### LLM Serving Runtime Metrics

| Metric Name                            | Type      | Description                                                                 |
| -------------------------------------- | --------- | --------------------------------------------------------------------------- |
| gpustack:num_requests_running          | Gauge     | Number of requests currently being processed.                               |
| gpustack:num_requests_waiting          | Gauge     | Number of requests waiting in the queue.                                    |
| gpustack:num_requests_swapped          | Gauge     | Number of requests swapped out to CPU.                                      |
| gpustack:prefix_cache_hit_rate         | Gauge     | Prefix cache hit rate.                                                      |
| gpustack:kv_cache_usage_ratio          | Gauge     | KV-cache usage ratio. 1.0 means fully used.                                 |
| gpustack:prefix_cache_queries          | Counter   | Number of prefix cache queries (total tokens).                              |
| gpustack:prefix_cache_hits             | Counter   | Number of prefix cache hits (total tokens).                                 |
| gpustack:prompt_tokens                 | Counter   | Total number of prefill tokens processed.                                   |
| gpustack:generation_tokens             | Counter   | Total number of generated tokens.                                           |
| gpustack:request_prompt_tokens         | Histogram | Number of prefill tokens processed per request.                             |
| gpustack:request_generation_tokens     | Histogram | Number of generation tokens processed per request.                          |
| gpustack:time_to_first_token_seconds   | Histogram | Time to generate first token.                                               |
| gpustack:inter_token_latency_seconds   | Histogram | Time to generate the next token after the previous token has been produced. |
| gpustack:time_per_output_token_seconds | Histogram | Time per generated token.                                                   |
| gpustack:e2e_request_latency_seconds   | Histogram | End-to-end request latency.                                                 |
| gpustack:request_success               | Counter   | Total number of successful requests.                                        |

These metrics are mapped from various runtime engines (vLLM, SGLang, MindIE) as defined in metrics_config.yaml.

### Worker Metrics

| Metric Name                                      | Type  | Description                                      |
| ------------------------------------------------ | ----- | ------------------------------------------------ |
| gpustack:worker_status                           | Gauge | Worker status (with state label).                |
| gpustack:worker_node_os                          | Info  | Operating system information of the worker node. |
| gpustack:worker_node_kernel                      | Info  | Kernel information of the worker node.           |
| gpustack:worker_node_uptime_seconds              | Gauge | Uptime in seconds of the worker node.            |
| gpustack:worker_node_cpu_cores                   | Gauge | Total CPU cores of the worker node.              |
| gpustack:worker_node_cpu_utilization_rate        | Gauge | CPU utilization rate of the worker node.         |
| gpustack:worker_node_memory_total_bytes          | Gauge | Total memory in bytes of the worker node.        |
| gpustack:worker_node_memory_used_bytes           | Gauge | Memory used in bytes of the worker node.         |
| gpustack:worker_node_memory_utilization_rate     | Gauge | Memory utilization rate of the worker node.      |
| gpustack:worker_node_gpu                         | Info  | GPU information of the worker node.              |
| gpustack:worker_node_gpu_cores                   | Gauge | Total GPU cores of the worker node.              |
| gpustack:worker_node_gpu_utilization_rate        | Gauge | GPU utilization rate of the worker node.         |
| gpustack:worker_node_gpu_temperature_celsius     | Gauge | GPU temperature in Celsius.                      |
| gpustack:worker_node_gram_total_bytes            | Gauge | Total GPU RAM in bytes.                          |
| gpustack:worker_node_gram_allocated_bytes        | Gauge | Allocated GPU RAM in bytes.                      |
| gpustack:worker_node_gram_used_bytes             | Gauge | Used GPU RAM in bytes.                           |
| gpustack:worker_node_gram_utilization_rate       | Gauge | GPU RAM utilization rate.                        |
| gpustack:worker_node_filesystem_total_bytes      | Gauge | Total filesystem size in bytes.                  |
| gpustack:worker_node_filesystem_used_bytes       | Gauge | Used filesystem size in bytes.                   |
| gpustack:worker_node_filesystem_utilization_rate | Gauge | Filesystem utilization rate.                     |

### Server Metrics

| Metric Name                      | Type  | Description                                       |
| -------------------------------- | ----- | ------------------------------------------------- |
| gpustack:cluster                 | Info  | Cluster information (ID, name, provider).         |
| gpustack:cluster_status          | Gauge | Cluster status (with state label).                |
| gpustack:model                   | Info  | Model information (ID, name, runtime, source).    |
| gpustack:model_desired_instances | Gauge | Desired number of model instances.                |
| gpustack:model_running_instances | Gauge | Number of running model instances.                |
| gpustack:model_instance_status   | Gauge | Status of each model instance (with state label). |

> **Note**: All metrics are labeled with relevant identifiers (cluster, worker, model, instance, user) for fine-grained monitoring and filtering.

## Deploy GPUStack with the Observability Stack

The observability stack consists of three main components:

- **GPUStack Server**: Exposes metrics at `/metrics` endpoint, expose worker metrics targets at `/metrics/targets`.
- **Prometheus**: Scrapes metrics from GPUStack and stores them.
- **Grafana**: Visualizes metrics from Prometheus.

All components are deployed via Docker Compose for easy management.

### Docker Compose Configuration

GPUStack provides a `docker-compose.yaml` file to deploy the observability stack in NVIDIA GPU environments, please refer to [Docker Compose](https://github.com/gpustack/gpustack/blob/main/docker-compose/docker-compose.server.nvidia.yaml).

Below is a sample `docker-compose.yaml` for deploying GPUStack Server, Prometheus, and Grafana:

```yaml
services:
  gpustack-server:
    image: gpustack/gpustack
    container_name: gpustack-server
    restart: unless-stopped
    privileged: true
    network_mode: host
    volumes:
      - gpustack-data:/var/lib/gpustack
      - /var/run/docker.sock:/var/run/docker.sock
    runtime: nvidia

  prometheus:
    image: prom/prometheus
    container_name: gpustack-prometheus
    restart: unless-stopped
    network_mode: host
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--web.enable-remote-write-receiver"
    volumes:
      - ./prometheus:/etc/prometheus
      - prom_data:/prometheus

  grafana:
    image: grafana/grafana
    container_name: gpustack-grafana
    restart: unless-stopped
    network_mode: host
    environment:
      - GF_SERVER_HTTP_PORT=3000
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

### Prometheus Configuration

Configure Prometheus to scrape metrics from GPUStack Server and Worker by adding the following to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: gpustack-worker-discovery
    scrape_interval: 5s
    http_sd_configs:
      - url: "http://localhost:10161/metrics/targets"
        refresh_interval: 1m
  - job_name: gpustack-server
    scrape_interval: 10s
    static_configs:
      - targets: ["localhost:10161"]
```

### Accessing Metrics

- **GPUStack Metrics Endpoint**:  
  Access metrics at `http://<gpustack_server_host>:10161/metrics`
- **GPUStack Worker Metrics Targets**:  
  Access metrics at `http://<gpustack_server_host>:10161/metrics/targets`
- **Prometheus UI**:  
  Access Prometheus at `http://<host>:9090`
- **Grafana UI**:  
  Access Grafana at `http://<host>:3000` (default user: `admin`, password: `grafana`)

### Customizing Metrics Mapping

GPUStack supports dynamic customization of metrics mapping through its configuration API. This allows you to update how runtime engine metrics are mapped to GPUStack metrics without restarting the service. The configuration is managed centrally on the server and can be accessed or modified via HTTP API.

#### API Endpoints

- **Get Current Metrics Config**

  - GET `http://<gpustack_server_host>:<gpustack_server_port>/v1/metrics/config`
  - Returns the current metrics mapping configuration in JSON format.

- **Update Metrics Config**

  - POST `http://<gpustack_server_host>:<gpustack_server_port>/v1/metrics/config`
  - Accepts a JSON payload to update the metrics mapping configuration. Changes take effect immediately for all workers.

- **Get Default Metrics Config**
  - GET `http://<gpustack_server_host>:<gpustack_server_port>/v1/metrics/default-config`
  - Returns the default metrics mapping configuration in JSON format, useful for reference or resetting.

#### Example Usage

**Get current config:**

```bash
curl http://<gpustack_server_host>:<gpustack_server_port>/v1/metrics/config
```

**Update config:**

```bash
curl -X POST http://<gpustack_server_host>:<gpustack_server_port>/v1/metrics/config \
     -H "Content-Type: application/json" \
     -d @custom_metrics_config.json
```

_(where `custom_metrics_config.json` is your new config file)_

**Get default config:**

```bash
curl -X POST http://<gpustack_server_host>:<gpustack_server_port>/v1/metrics/default-config
```

> **Note**: The configuration should be provided in valid JSON format.
