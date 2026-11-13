# Monitoring

This package owns monitoring semantics and Prometheus reads. HTTP authorization
and response declarations stay in `routes/` and `schemas/`; engine collection and
metric normalization stay in `worker/`. It is a domain package rather than a
Server process component. It starts no background jobs and imports no HTTP routes.

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `../assets/monitoring-queries.yaml` | Chart-to-metric map, metric families/profiles, actual PromQL expressions and default time policies |
| `config.py` | Strict YAML loading, reference/variable validation and non-executable template binding |
| `catalog.py` | Effective metric declarations and backend semantic overrides resolved from YAML |
| `queries.py` | Authorized Model/source bindings, mandatory guard application and compatible-bucket statistics |
| `prometheus.py` | Shared Prometheus HTTP transport, response validation and query-window parsing |
| `time_range.py` | Fixed/relative interval validation and observed Prometheus retention configuration |
| `models.py` | Model observations, PD role selection, coverage and summary/series assembly |
| `hardware.py` | GPU/Worker identities, sensor targets and shared hardware history collection |
| `resources.py` | Worker/GPU current inventory, authorized deployment placements and direct resource collection |
| `benchmarks.py` | Recorded run identities, measured phase bounds, independently authorized observations and exact-range gauge statistics |
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

Allocation counts include only placements on currently visible Workers. Hidden
or missing Workers do not contribute to the GPU or Worker totals.

Duplicate scrape paths must agree after removing transport labels. PromQL
excludes conflicting copies instead of selecting the largest value; defensive
instant-result validation also rejects conflicts before summary aggregation.

The Worker refreshes its server-verified ownership snapshot every 15 seconds in
the background. Stable hardware collection reads that snapshot without HTTP
I/O. A failed refresh, changed registration or 30-second expiry invalidates it;
stable samples stop until identity is verified again, while legacy and runtime
collection remain independent. Losing a verified identity emits a warning.

## Benchmark history

Benchmark creation freezes the selected members and source/Worker/GPU identities
in the existing JSON snapshot. Running confirmation and result updates verify those
identities without substituting current membership. A changed generation makes
monitoring unavailable rather than assigning later samples to the recorded run.
Measured phase windows come from the runner's scheduler metrics. Short ranges keep
their real bounds: rolling samples whose lookback crosses the start are missing,
and gauge observations cannot reuse an earlier stored sample. Gauge statistics and
coverage describe query evaluation points in the half-open interval, not raw scrape
counts. Old records without run identities retain their client report but cannot
reconstruct monitoring associations. All three Benchmark monitoring endpoints
recheck current resource authorization; internal source identities stay private.

## Find or change a query

[monitoring-queries.yaml](../assets/monitoring-queries.yaml) is the Server's single
query declaration file for native Model/Worker/GPU charts. Start at `charts`,
follow each API metric key into `metrics` or `hardware_metrics`, then follow
`query_profile` into `profiles`. The actual PromQL bodies are visible there:
`rate`, `increase`, Avg, P50/P95/P99 and hardware freshness are declarations,
not Python string literals. `expressions` contains the shared source guard,
aggregation and population-matching templates. Collectors always apply them.

For example, `model.ttft` references `ttft`, whose family is
`gpustack:time_to_first_token_seconds` and profile is `histogram`. That profile
declares `p95: histogram_quantile(0.95, ${buckets})`. `${buckets}` is assembled
from the same profile's rate expression, the mandatory source guard and the
declared grouping. A metric can select a specialized profile without changing
other metrics. Backend overrides preserve existing differences such as a
reported cache-hit Gauge versus a ratio of cumulative Counters.

Every hardware metric explicitly declares `kind: gauge`, matching the current
sensor query support. The loader requires the field and checks its kind and
hardware scope against the selected query profile.

Variables such as `${selector}`, `${grouping}` and `${window_seconds}` are bound
by the Server from authorized resources and validated request parameters. This
uses `string.Template`, not `eval` or executable expressions. The packaged file
is trusted application configuration, not a tenant upload or a new public API.
Unknown fields, missing variables, duplicate YAML keys, mismatched profile types
and invalid chart references fail validation. A query profile cannot opt out of
the collector's source guard or identity checks.

`time_windows` holds the existing defaults: LLM step targets 120 intervals with
a 15-second floor and a 60-second rate-window margin; hardware targets 240
intervals with the same floor and a 120-second freshness limit. Interval
validation, full-range summaries, histogram compatibility and response DTOs
remain code responsibilities. Chart metadata is for auditing and reference
validation; its English `description` fields are developer documentation, not
API display text or translation keys. The UI owns localized chart titles and
React page layout. Client-derived
Token totals and memory percentages are marked `derived_in_ui`, never evaluated
from the description string.

The file is cached per Server process. Restart the Server after changing it.
Validate before starting with:

```sh
uv run --no-sync python -c 'from gpustack.monitoring.config import get_query_config; get_query_config()'
```

Query declarations do not define API paths, filter parameters or response keys. Raw exporter
version mappings still belong to the Worker's `metrics_config.yaml`; Cache
provider declarations stay in `cache-providers.yaml`.

The runtime API selects latency curves with one comma-separated `statistic`
parameter: `avg`, `p50,p99`, or `avg,p50,p95,p99`. Omitting it requests all four,
in either aggregate or instance view. Full-window summaries retain all four
statistics independently of the selected curves.

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
   `monitoring-queries.yaml` declares the queries over normalized families;
   `schema_version: 1` versions its file format, not engine releases or historical
   samples. YAML and its compiler/collectors are maintained and released with
   GPUStack's Git version.
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
- Add metric definitions, profiles and chart references to
  `monitoring-queries.yaml` when a new semantic query is needed. A new UI chart
  also needs the corresponding frontend component; changing only a Grafana panel
  does not change native detail charts.
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
