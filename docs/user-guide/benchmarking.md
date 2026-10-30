# Benchmarking

GPUStack can run benchmarks against running model instances. Benchmarks are executed by workers in a dedicated benchmark container image, with results and logs stored on the worker.

## Prerequisites

- A model instance is running and healthy.

Existing running instances can be benchmarked without redeployment. Reports use
each instance's recorded startup configuration when available. For instances
without a startup snapshot, reports use the model configuration at benchmark
creation, including role overrides, and identify the affected instances with a
reference-only notice. These reference settings may differ from the running
configuration. Benchmarking does not backfill the instance's startup snapshot.

### Route benchmark gateway address

Route benchmarks use a worker-reachable Higress base URL (without `/v1`),
resolved in this order:

1. The execution worker's cluster `server_url`.
2. The server's `server_external_url`.
3. The creation request's origin, explicitly allowed by the server's `trusted_hosts`.

For request-origin fallback, wildcards are not accepted. Bare hostnames allow
only standard HTTP/HTTPS ports; list custom ports explicitly, such as
`gateway.example:8443`. If the URL includes a path prefix, the front proxy must
strip it before forwarding to Higress.

The target is fixed when the benchmark is created. Recreate the benchmark to
use a changed URL. Progress reporting keeps the worker's management address.

## Create Benchmark

1. Go to the `Benchmarks` page.
2. Click the `Add Benchmark`.
3. Select an instance and fill in the configurations.
4. Click the `Save` button.

### ShareGPT workload lengths

For the built-in ShareGPT dataset, optionally set a minimum input token length,
a maximum, or both. GPUStack keeps first-turn prompts whose tokenizer counts
meet the supplied limits. The output token length is optional: when set,
the same maximum is requested for every retained prompt; when omitted, each
prompt uses its reference answer length from the dataset. If the range matches
no prompts, the run fails before sending benchmark requests.

## View Benchmark Results

1. Go to the `Benchmarks` page.
2. Find the benchmark you want to view.
3. Click the benchmark name to view the results and configurations snapshot.

## Export Benchmark Results

1. Go to the `Benchmarks` page.
2. Find the benchmark you want to export.
3. Select one or more benchmark runs to export.
4. Click the `Export` button to download the results.

## Edit Benchmark

1. Go to the `Benchmarks` page.
2. Find the benchmark you want to edit.
3. Click the `Edit` button in the `Operations` column.
4. Modify the name, description as needed.
5. Click the `Save` button.

## Delete Benchmark

1. Go to the `Benchmarks` page.
2. Find the benchmark you want to delete.
3. Click the `Delete` button in the `Operations` column.
4. Confirm the deletion.

## Benchmarks on an HTTPS Server with a Private CA

Progress reporting and route inference follow the worker's TLS policy. Certificate verification is enabled unless the worker has `GPUSTACK_INSECURE_TLS` or the `insecure_tls` configuration option (`--insecure-tls`) enabled. There is no separate benchmark TLS setting.

When either entrance uses HTTPS and verification is enabled, the worker injects its trusted CA bundle as file content into the benchmark container. This includes private CAs imported on the worker (see [Additional Trusted CAs](../installation/installation.md#additional-trusted-cas)). The progress client uses `--progress-ca-cert`; HTTPS inference uses the injected bundle through `SSL_CERT_FILE` with backend verification enabled. An HTTP management address does not prevent CA injection for an HTTPS gateway.

If the worker explicitly skips verification, the benchmark skips it for both channels and does not inject a CA bundle. If the bundle cannot be read while verified HTTPS inference is required, the benchmark refuses to start instead of using a different trust store. Progress-only reporting remains best effort: reporting failures are logged without discarding the benchmark measurement.
