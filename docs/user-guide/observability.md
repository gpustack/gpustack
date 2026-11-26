# Observability

Monitor GPUStack metrics with Prometheus and Grafana, and use inference logs and traces to investigate individual requests.

## Overview

GPUStack provides a comprehensive set of metrics for model serving and GPU resource management. By integrating Prometheus and Grafana, users can collect, store, and visualize these metrics in real time, enabling efficient monitoring and troubleshooting.

## Native Model Monitoring

Open a deployment name under **Model Service → Deployments** to view Overview, Monitoring, Instances, GPU Resources, and Configuration. Monitoring groups can be expanded independently and share the selected time, instance, and PD role filters.

- Runtime charts use the inference engines' metrics. Gateway measurements do not replace missing engine samples. PD latency is labeled by the producing role and represents a local request, not automatically the complete client request.
- Latency statistics are Avg, P50, P95, and P99 where supported. The statistic row covers the selected interval; trend points use the shorter rate window shown on the page. Missing observations remain gaps rather than zero. Unsupported, inapplicable, failed, and stale measurements are distinguished.
- GPU and host trends use owner-qualified stable identities. Upgrade Workers to collect the new stable history series; older Workers can still provide their latest reported device values. Unverifiable legacy device history is not reconstructed from GPU indexes. Physical measurements include other workloads sharing the device or host.
- Runtime history covers current instances and their observed source epochs. Hardware history covers the currently associated devices under their recorded owner; neither is a complete historical placement audit.

Model monitoring, Model dashboard redirects, and benchmark monitoring require the organization Owner role or platform administrator access. GPUStack also checks visibility of the requested deployment or benchmark and the associated resources.

Worker and GPU monitoring and their dashboard redirects follow resource visibility. Organization members can use these endpoints for resources they can view. Grafana and Prometheus proxy endpoints apply their own permissions.

The semantic endpoints are `GET /v2/models/{id}/metrics` and `GET /v2/models/{id}/resource-metrics`. They accept controlled filters; clients do not submit PromQL. Charts, tables, and CSV exports use the same authorized results.

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

   Workers in `tunnel` proxy mode are only reachable through the server's forward proxy (default port `30079`), which authenticates every request it relays. To scrape those workers, add a job that presents a GPUStack API key (create one in the UI and copy it unchanged — a generated key looks like `gpustack_<access_key>_<secret_key>`) as the credential for the proxy hop:

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
  Access Prometheus at `http://<host>:19090` by default, or the port configured by `--builtin-prometheus-port` / `GPUSTACK_BUILTIN_PROMETHEUS_PORT`.
- **Grafana UI**:  
  Access Grafana at `http://<host>:13000` by default, or the port configured by `--builtin-grafana-port` / `GPUSTACK_BUILTIN_GRAFANA_PORT`. Built-in Grafana is configured for anonymous Viewer access with the login form disabled. The admin credentials remain `admin` / `grafana` by default.

## Inference Logs and Traces

Use the `X-GPUStack-Request-Id` response header to look up inference logs and traces. The bundled gateway adds this header when logging or tracing is configured. Docker servers export traces to embedded Tempo by default. Helm Server releases deploy a dedicated Tempo workload by default. Log export is independent and requires a log endpoint.

| Trace destination or setting | Docker | Helm |
| --- | --- | --- |
| Built-in Tempo | Default on the Server | Default in a Server release |
| External OTLP/gRPC receiver | Set `GPUSTACK_TRACE_OTLP_ENDPOINT` | Set `tracing.endpoint` |
| Built-in Tempo block retention (default `168h`) | Set `GPUSTACK_BUILTIN_TEMPO_RETENTION` | Set `tracing.tempo.retention` |
| Disabled | Set `GPUSTACK_DISABLE_TRACING=true` | Set `tracing.enabled=false` |

Tempo removes expired blocks during periodic cleanup. Retention controls block age. Total disk use depends on the volume of traces received and the space required by the write-ahead log (WAL), compaction, and blocks pending deletion.

Estimate the space for retained blocks by multiplying the ingested trace rate by the average on-disk block bytes per trace and the retention duration in seconds. For a worked example, assume 10 requests per second, 100% sampling, one trace per sampled request, and 1,000 on-disk block bytes per trace. That gives about 864 MB of blocks per day, or 6.05 GB over seven days. Adding an illustrative 50% headroom brings the estimate to about 9.1 GB. Measure block bytes per trace and peak space for WAL, compaction, and pending deletions under a representative workload to size the volume and its headroom.

The `Request Tracing` tab in deployment details initially searches the last 24 hours. Viewing traces requires the organization Owner role or platform administrator access. GPUStack applies the same model visibility rules as deployment metrics. Page through the results or paste the Request ID or Trace ID as it appears in GPUStack or Tempo. When log export is configured, copy the Request ID from the request detail and search for it in your log backend. For external Tempo, set the query endpoint using `GPUSTACK_TRACE_QUERY_URL` or `tracing.queryURL`.

When log export is enabled, the Server and Workers write one JSON stream to the container console. Alloy reads that stream alongside supported engine log formats. To check for discarded records, use Alloy's `otelcol_processor_filter_logs_filtered_total` metric with the `engine_format` or `higress_format` component. The engine filter rejects standalone plain-text lines without a recognized log prefix.

GPUStack automatically adds engine observability settings for compatible Runner images with vLLM `0.32.0` or SGLang `0.5.22` when the corresponding export is enabled and the engine settings allow it. The version checks apply to automatic configuration. For other images or versions, configure telemetry using the engine's own settings.

Each tracing component sends spans directly to the selected receiver. The log endpoint must be reachable from the components that export logs. The trace endpoint must be reachable from the Server, Workers, gateways, and inference containers that export traces. The ingest connection uses plaintext gRPC, so use an address on a trusted network. Switching to an external receiver or disabling tracing stops built-in Tempo.

### Configure Docker

1. To enable log export, set `GPUSTACK_LOG_OTLP_ENDPOINT=http://<collector-host>:4318/v1/logs` on the Server and each separate Worker that exports logs. Mount `/var/run/docker.sock:/var/run/docker.sock` and use a logging driver that supports `docker logs`.
2. To export traces to an external receiver, set `GPUSTACK_TRACE_OTLP_ENDPOINT=<collector-host>:4317` on the Server and each separate Worker. To use built-in Tempo with remote Workers, see [Enable Tracing on Existing Docker Workers](../upgrade/cluster.md#enable-tracing-on-existing-docker-workers).
3. Replace `<collector-host>` with a name or address reachable from those containers. Pass the variables with `docker run -e` or under `environment` in your Compose file, then recreate the containers.

Docker runs Tempo and Alloy inside the GPUStack container. Tempo stores its WAL and blocks under `<data_dir>/tempo` and serves queries on loopback. Trace files follow the lifecycle of the Server data volume. To change block retention, set `GPUSTACK_BUILTIN_TEMPO_RETENTION` on the Server and recreate its container.

### Configure Helm

1. Configure the endpoints in the Helm values of each GPUStack release. Set `logging.endpoint` to enable log export and `tracing.endpoint` to use an external trace receiver. Worker-only releases can use the Server's receiver by setting `tracing.endpoint`.

   ```yaml
   logging:
     endpoint: "http://<collector-host>:4318/v1/logs"
   tracing:
     endpoint: "<collector-host>:4317"
   ```

2. Replace `<collector-host>` with the receiver address and upgrade the Helm release. Use a namespace-qualified Service DNS name for a Collector in another namespace.

The bundled Higress gateway writes JSON access logs by default. To change its format, set `higress-core.meshConfig.accessLogEncoding` and `higress-core.meshConfig.accessLogFormat` in the Helm values.

The Tempo and Alloy images use `global.hub`, which defaults to `docker.io`. For private mirrors, keep the `grafana/tempo` and `grafana/alloy` repository paths.

The Chart runs Tempo separately from the Server Pods and mounts an `emptyDir` at `/var/tempo` by default. Replacing the Tempo Pod discards traces stored on this volume. To persist traces across Pod replacements, set `tracing.tempo.existingClaim` to an existing PVC in the release namespace. The volume must be writable by UID or group `10001`. To change block retention, set `tracing.tempo.retention` and upgrade the release. An empty or invalid retention value uses `168h`.

All Server Pods query built-in Tempo at `https://<release>-tempo.<namespace>.svc.<cluster-domain>:3200` using mutual TLS, including releases with a single Server. The Chart provisions a query TLS Secret by default. Workers and inference engines send traces to the Service's OTLP/gRPC port `4317`.

For GitOps deployments, create a Secret in the Server release's namespace and set `tracing.tempo.queryTLSSecretName` to its name. Include `ca.crt`, `server.crt`, `server.key`, `client.crt`, and `client.key`. The server certificate must cover `<release>-tempo.<namespace>.svc.<cluster-domain>`. Use the same CA when renewing the server and client certificates. Create a new Secret and update its name in the Helm values to roll the Tempo and Server Pods with the renewed certificates.

Restart or redeploy running model instances after changing their trace export endpoint.

### Redact Exported Logs

Alloy can replace matches in the log body, exception stack trace, and `path` with `***` before exporting logs. Use a list of non-empty RE2 regular expressions. An empty list disables redaction. Request, trace, and span IDs remain available for correlation.

For Docker, pass `GPUSTACK_LOG_REDACT_PATTERNS` as a JSON array of strings on each container that exports logs. This example masks email addresses and mainland Chinese mobile numbers:

```bash
GPUSTACK_LOG_REDACT_PATTERNS='["(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+[.][A-Z]{2,}","\\b1[3-9][0-9]{9}\\b"]'
```

For Helm, set the equivalent YAML list in each release that exports logs:

```yaml
logging:
  redactPatterns:
    - '(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+[.][A-Z]{2,}'
    - '\b1[3-9][0-9]{9}\b'
```

Recreate the Docker containers or upgrade the Helm release to apply the rules. Check the exported records in your log backend by `request_id`. The rules affect new exports from the GPUStack-managed Alloy. Original records remain available through `docker logs`, `kubectl logs`, or the runtime's log files. Previously exported records and traces are outside these rules. Invalid rules can prevent Alloy from exporting logs while GPUStack continues serving requests.

### Look Up a Request

1. Send a request to a running chat model and keep the response headers. Replace `<gpustack-host>`, `<api-key>`, and `<model-name>` with your deployment values. See [API Key Management](api-key-management.md) for inference permissions.

   ```bash
   curl --silent --show-error --no-buffer \
     --dump-header response-headers.txt \
     'http://<gpustack-host>/v1/chat/completions' \
     --header 'Authorization: Bearer <api-key>' \
     --header 'Content-Type: application/json' \
     --data '{"model":"<model-name>","messages":[{"role":"user","content":"Reply with one word."}],"max_tokens":8,"stream":true}'

   grep -i '^X-GPUStack-Request-Id:' response-headers.txt
   ```

2. In your log backend, set the time range around the request and filter on the `request_id` log attribute. Use the OTLP resource attribute `service.name` to select a source such as `higress`, `gpustack.server`, or `gpustack.engine`. Where present, `worker_id`, `model_instance_id`, and `role` can narrow the results.
3. If a log record carries an OTLP trace ID, use that ID to retrieve the trace from your trace backend. With tracing enabled, you can also search for the platform request ID using the `gpustack.request_id` span attribute, including when log export is off.

### Example with Loki and Grafana

1. For a Loki deployment with [OTLP ingestion and structured metadata enabled](https://grafana.com/docs/loki/latest/send-data/otel/), set `GPUSTACK_LOG_OTLP_ENDPOINT` or `logging.endpoint` to its full ingest URL, such as `http://<loki-host>:3100/otlp/v1/logs`. For traces, configure `GPUSTACK_TRACE_OTLP_ENDPOINT` or `tracing.endpoint` with a Tempo OTLP/gRPC address reachable by the span producers. Configure Grafana data sources with the Loki and Tempo HTTP query URLs.
2. In Grafana `Explore`, select Loki, set the request's time range, and replace `<request-id>` with the `X-GPUStack-Request-Id` response header value. With Loki's default OTLP mapping, `service.name` becomes the `service_name` stream label, and `request_id` is structured metadata.

   ```logql
   {service_name=~"higress|gpustack[.](server|worker|engine)"} | request_id="<request-id>"
   ```

   To narrow the results, append `| worker_id="<worker-id>"` or `| model_instance_id="<instance-id>"` for records that carry those fields.
3. If a result carries `trace_id`, copy it into the Tempo data source's `TraceQL` query field. To open the trace directly from a log result, configure a Loki [derived field](https://grafana.com/docs/grafana/latest/datasources/loki/configure/#derived-fields) of type `Label` matching `^trace_id$`, with an internal link to the Tempo data source and `${__value.raw}` as the query.

Manage retention and access in the external platform. If tenant users can query the data, enforce tenant isolation there.

!!! tip "Find PD engine logs"

    For SGLang Mooncake, use `request_id` to find both engine logs and `pd_attempt_id` to distinguish dispatch attempts. For vLLM NIXL, find the entry log by `request_id`, then search for its `pd_coordinator_id` across both engine logs. If the entry log lacks that field, narrow the Prefill logs by request time and instance, and copy `pd_coordinator_id` from a confirmed match.

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
curl http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/default-config
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
| gpustack:request_preemptions           | Counter   | Total number of preemptions from the engine.                                |

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
| gpustack:worker_node_gpu_power_limit_watts       | Gauge | GPU power management limit in watts.             |
| gpustack:worker_node_gpu_power_used_watts        | Gauge | GPU power usage in watts.                        |
| gpustack:worker_node_gram_total_bytes            | Gauge | Total GPU RAM in bytes.                          |
| gpustack:worker_node_gram_allocated_bytes        | Gauge | Allocated GPU RAM in bytes.                      |
| gpustack:worker_node_gram_used_bytes             | Gauge | Used GPU RAM in bytes.                           |
| gpustack:worker_node_gram_utilization_rate       | Gauge | GPU RAM utilization rate.                        |
| gpustack:worker_node_filesystem_total_bytes      | Gauge | Total filesystem size in bytes.                  |
| gpustack:worker_node_filesystem_used_bytes       | Gauge | Used filesystem size in bytes.                   |
| gpustack:worker_node_filesystem_utilization_rate | Gauge | Filesystem utilization rate.                     |

### Server Metrics

| Metric Name                                 | Type  | Description                                                   |
| ------------------------------------------- | ----- | ------------------------------------------------------------- |
| gpustack:cluster                            | Info  | Cluster information (ID, name, provider).                     |
| gpustack:cluster_status                     | Gauge | Cluster status (with state label).                            |
| gpustack:model                              | Info  | Model information (ID, name, runtime, source, category).      |
| gpustack:model_desired_instances            | Gauge | Desired number of model instances.                            |
| gpustack:model_running_instances            | Gauge | Number of running model instances.                            |
| gpustack:model_instance_status              | Gauge | Status of each model instance (with state label).             |
| gpustack:model_instance_restart_count       | Gauge | Model instance restart count.                                 |
| gpustack:model_instance_latest_restart_time | Gauge | Model instance latest restart time as Unix timestamp seconds. |

### Event Bus Metrics

| Metric Name                         | Type    | Description                                                                                                      |
| ----------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| gpustack:bus_subscribers            | Gauge   | Active bus subscribers per `topic` + `source` (several may share `source=streaming`, one per open watch stream). |
| gpustack:bus_queue_depth            | Gauge   | Max queue depth across subscribers sharing `topic`+`source`.                                                     |
| gpustack:bus_queue_capacity         | Gauge   | Per-subscriber queue maxsize (see env knob below).                                                               |
| gpustack:bus_queue_full             | Gauge   | 1 if any subscriber sharing `topic`+`source` has a full queue.                                                   |
| gpustack:bus_queue_saturation_ratio | Gauge   | Max `qsize / maxsize` in `[0, 1]`. Sustained > 0.8 ⇒ slow consumer.                                              |
| gpustack:bus_subscriber_latest_keys | Gauge   | Max ids pending coalesced UPDATED delivery (size of `latest_by_key`).                                            |
| gpustack:bus_events_total           | Counter | Cumulative event counts summed across subscribers sharing `topic`+`source`. Extra labels: `kind`, `event_type`.  |

> **Note**: All metrics are labeled with relevant identifiers (cluster, worker, model, instance, user) for fine-grained monitoring and filtering.
