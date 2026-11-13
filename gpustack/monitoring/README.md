# Monitoring

This package owns monitoring semantics and Prometheus reads. HTTP authorization
and response declarations stay in `routes/` and `schemas/`; engine collection and
metric normalization stay in `worker/`. It is a domain package rather than a
Server process component. It starts no background jobs and imports no HTTP routes.

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `catalog.py` | Chart keys, normalized metric families, units, aggregation kinds and sensor bounds |
| `queries.py` | Model/source selectors, reset-aware counters, histogram queries and compatible-bucket statistics |
| `prometheus.py` | Shared Prometheus HTTP transport, response validation and query-window parsing |
| `time_range.py` | Fixed/relative interval validation and observed Prometheus retention configuration |
| `models.py` | Model observations, PD role selection, coverage and summary/series assembly |
| `hardware.py` | GPU/Worker identities, sensor targets and shared hardware history collection |
| `resources.py` | Worker/GPU current inventory, authorized deployment placements and direct resource collection |
| `cache.py` | Cache-provider declarations translated into semantic cache charts |

Routes call the collectors after establishing the resource and tenant context.
Resource and hardware collection also filter every related Worker, model and
instance before constructing selectors or returning association metadata.
`resources.py` and the Model resource collector share the hardware implementation,
so a physical GPU has the same measurement semantics in every detail page.

The package consolidates the monitoring domain without moving the Server's
unrelated usage accounting or background collectors. Internal hardware helpers
remain within this package. A separate service or plugin layer is unnecessary
for the existing in-process API calls.

## Data path

```text
Engine /metrics
  -> Worker runtime_metrics_aggregator
  -> versioned native-to-gpustack mapping and unit conversion
  -> Worker exporter -> Prometheus scrape
  -> authorized API -> monitoring collector -> generated PromQL
  -> Prometheus query_range / query -> semantic response -> UI charts

Worker hardware/status
  -> owner-qualified Worker exporter -> Prometheus -> hardware history
  -> current resource snapshots and allocation records -> resource detail
```

The UI supplies filters and time bounds, not arbitrary PromQL. Query builders
attach the authorized Model/instance population and source identity. Hardware
selectors use owner, Worker UUID and GPU UUID/chip identity. Chart time series use
`query_range`; range summaries and latest observations use instant `query` calls.
Counters use `rate`/`increase`; latency percentiles use histogram buckets, never an
average of instance percentiles. Missing observations remain missing, including
when an engine does not export a metric. Current inventory and allocation
metadata come from resource records and do not replay historical ownership.

## Compatibility and expression maintenance

There are three related compatibility boundaries:

1. **Engine names, versions, types and units.**
   `assets/metrics_config/metrics_config.yaml` defines the unified metric families
   and `runtime_mapping` rules. The Worker applies the first matching explicit
   version rule for a native family, falling back to its `"*"` entry. It uses the
   detected runtime version when available and the deployment version otherwise.
   Unknown or unverified legacy meanings are not guessed. Unit conversion includes
   histogram bounds and sums while preserving observation counts.
2. **Cache-provider versions and extraction declarations.**
   `assets/cache-providers.yaml` supplies `default_metrics`, while a declared
   provider version may supply its own complete `metrics` block. Both the cache
   metrics route and exporter discovery call
   `provider.metrics_for(service.provider_version)` before building queries or
   choosing the scrape path. A version override replaces the whole declaration;
   it does not merge individual metric keys with the default. Without an override,
   the provider default applies. An omitted version resolves through the default
   version; custom or unknown versions use the provider default on a best-effort
   basis. The declaration chooses forms such as gauge, rate, counter ratio,
   gauge ratio and histogram average, and `cache.py` builds the actual PromQL.
3. **Chart semantics and PromQL.**
   `catalog.py`, `queries.py` and the collectors operate on those normalized
   families. They are maintained and released with GPUStack's Git version.
   There is no independent PromQL revision table. Selecting a provider-version
   declaration is distinct from versioning the query builders themselves. Native
   detail queries do not read expressions from Grafana dashboards.

For example, vLLM's legacy `time_per_output_token_seconds` maps to **ITL** before
`0.10.2`, while `request_time_per_output_token_seconds` maps to **TPOT**. A chart
therefore queries the normalized ITL or TPOT family without branching its PromQL
on the engine version. Hardware families are emitted by GPUStack's Worker exporter
and do not depend on vLLM/SGLang version mappings. Cache charts instead use the
provider catalog's version-resolved metric declarations. The version-override
contract is covered by `test_metrics_for_resolves_version_override` in
`tests/server/test_cache_provider_catalog.py`.

The Worker fetches `/v2/metrics/config`, with its packaged YAML as a fallback,
and caches the result for 300 seconds. An administrator's custom configuration
therefore affects normalization for newly collected samples; it does not rewrite
old Prometheus samples or install a different set of chart queries. The sample
identity does not currently include a mapping-configuration revision.

When maintaining this boundary:

- Put native renames and verified version exceptions in `runtime_mapping`, rather
  than adding a copy of every chart expression for each backend version.
- Preserve a normalized family's meaning and unit. A semantic change needs a
  distinct family instead of combining incompatible historical samples.
- Add chart definitions and query behavior here when a new semantic chart is
  needed; changing only a Grafana panel does not change native detail charts.
- Verify compatibility against the fixed payloads and audited expectations under
  `tests/fixtures/runtime_metrics/`, not expected data generated from the same YAML.

## Verification

`tests/monitoring/` mirrors the query and collector modules. Route tests verify
authorization and filters; Worker tests verify version mappings, units and exporter
labels. The focused regression set is:

```sh
uv run --no-sync pytest tests/monitoring \
  tests/routes/test_model_metrics_endpoint.py \
  tests/routes/test_model_monitoring_dashboard_scope.py \
  tests/routes/test_cache_service_metrics_endpoint.py \
  tests/routes/test_metrics_authorization.py \
  tests/worker/test_runtime_metrics.py \
  tests/worker/test_runtime_metrics_mappings.py \
  tests/worker/test_runtime_metrics_version_probe.py \
  tests/worker/test_metrics_unit_scaling.py \
  tests/worker/test_worker_exporter.py
```
