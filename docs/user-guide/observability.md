# Observability

This document describes how to monitor GPUStack Server/Worker/LLM serving runtime metrics using Prometheus and Grafana.

## Overview

GPUStack provides a comprehensive set of metrics for model serving and GPU resource management. By integrating Prometheus and Grafana, users can collect, store, and visualize these metrics in real time, enabling efficient monitoring and troubleshooting.

## Built-in Observability (Default)

By default, GPUStack starts with an embedded Prometheus and Grafana. You can access them via:

- **Prometheus**: `http://your_gpustack_server_host_ip/prometheus`
- **Grafana**: `http://your_gpustack_server_host_ip/grafana`

Built-in Grafana is configured for anonymous Viewer access and has the login form disabled. Admin credentials remain `admin` / `grafana` by default.

## External Observability (Optional)

If you want an external Prometheus/Grafana stack, we recommend using the provided Docker Compose files:

Run the following commands to clone the latest stable release:

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

Before starting, set `GPUSTACK_GRAFANA_URL` to a browser-reachable Grafana URL (not a container-only hostname like `grafana`).

Start external Prometheus/Grafana (this disables the built-in stack):

```bash
sudo docker compose -f docker-compose.external-observability.yaml up -d
```

If you already have an external Prometheus/Grafana stack, you can configure it manually instead:

1. **Configure Prometheus to scrape GPUStack metrics**  
   Add targets for the GPUStack metrics endpoint (default `:10161`) and worker discovery endpoint. Example:

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
2. **Import GPUStack dashboards into Grafana**  
   Use the dashboards provided in the `docker-compose/grafana/grafana_dashboards/` directory as a starting point.
3. **Point GPUStack to your Grafana**  
   Set `GPUSTACK_GRAFANA_URL` to the externally reachable Grafana URL so dashboard redirects work. This must be a browser-reachable URL.

## Accessing Metrics

- **GPUStack Metrics Endpoint**:  
  Access metrics at `http://<gpustack_server_host>:10161/metrics`
- **GPUStack Worker Metrics Targets**:  
  Access metrics at `http://<gpustack_server_host>:10161/metrics/targets`
- **Prometheus UI**:  
  Access Prometheus at `http://<host>:9090`
- **Grafana UI**:  
  Access Grafana at `http://<host>:3000`. Built-in Grafana is configured for anonymous Viewer access with the login form disabled. The admin credentials remain `admin` / `grafana` by default.

## Migration from Older Compose Setups

If you previously used Docker Compose to run Prometheus/Grafana alongside GPUStack:

- **Keep external observability (recommended for continuity)**:  
  Leave your existing Prometheus/Grafana containers running. Update Prometheus scrape targets to the new GPUStack metrics endpoint and set `GPUSTACK_GRAFANA_URL` to your existing Grafana.

- **Switch to built-in observability**:  
  Stop the old Prometheus/Grafana containers, then use the latest `docker-compose.server.yaml` (GPUStack only). Built-in Grafana/Prometheus will take over. Historical metrics from the old Prometheus will not be migrated unless you keep the old stack read-only.

## Customizing Metrics Mapping

GPUStack supports dynamic customization of metrics mapping through its configuration API. This allows you to update how runtime engine metrics are mapped to GPUStack metrics without restarting the service. The configuration is managed centrally on the server and can be accessed or modified via HTTP API.

### API Endpoints

- **Get Current Metrics Config**

  - GET `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config`
  - Returns the current metrics mapping configuration in JSON format.

- **Update Metrics Config**

  - POST `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config`
  - Accepts a JSON payload to update the metrics mapping configuration. Changes take effect immediately for all workers.

- **Get Default Metrics Config**
  - GET `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/default-config`
  - Returns the default metrics mapping configuration in JSON format, useful for reference or resetting.

### Example Usage

**Get current config:**

```bash
curl http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config
```

**Update config:**

```bash
curl -X POST http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config \
     -H "Content-Type: application/json" \
     -d @custom_metrics_config.json
```

_(where `custom_metrics_config.json` is your new config file)_

**Get default config:**

```bash
curl -X POST http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/default-config
```

> **Note**: The configuration should be provided in valid JSON format.

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
