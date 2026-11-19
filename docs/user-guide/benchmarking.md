# Benchmarking

GPUStack can run benchmarks against running model instances. Benchmarks are executed by workers in a dedicated benchmark container image, with results and logs stored on the worker.

## Prerequisites

- A model instance is running and healthy.

Existing running instances can be benchmarked without redeployment. Reports use
each instance's recorded startup configuration only when it matches the current
launch. If a snapshot is missing or cannot be verified for the current launch,
reports use the model configuration at benchmark
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

## Saved monitoring

GPUStack saves monitoring for new benchmarks in its server database. This is the embedded PostgreSQL database by default, or the database configured with `--database-url`. The **Stages and Monitoring** view can replay the available curves for the full benchmark and individual measured phases after deployment resources are gone. An archive can be partial if source metrics were incomplete. Viewers authorized to access the archived hardware details can see the compressed size alongside the save status. Benchmark results remain available without an archive.

A route benchmark records the running replicas of the selected deployment when the benchmark is created. If GPUStack detects that a recorded replica was restarted, reallocated, or removed, live monitoring for that benchmark becomes unavailable. The archive preserves eligible observations from before the change and marks their coverage as partial.

Archives are compressed, and the per-run limit applies to both compressed and uncompressed data (256 MiB by default). A sizing test compressed data from two roughly six-minute benchmarks to about 177 KiB for one regular deployment and 528 KiB for one PD deployment. Larger runs can produce larger archives. Include archive storage in your database sizing and backups.

Platform administrators can set a global byte limit in **Archive Storage** on the **Benchmarks** page. By default, the installation-wide quota is unlimited (`0`). A finite quota counts compressed archive bytes across all organizations. One organization can use the available space and block saves that would increase archive storage for any organization. The last successful saves remain available.

Each save writes the compressed archive in one database statement. For a MySQL-compatible database, check the effective `max_allowed_packet` and set `--benchmark-archive-max-run-bytes` below that limit, leaving room for SQL encoding and metadata. [MySQL packet limits](https://dev.mysql.com/doc/refman/8.0/en/packet-too-large.html) · [OceanBase packet limits](https://en.oceanbase.com/docs/common-oceanbase-database-10000000003681766)

A repeat repair of a sealed archive starts at least one archive flush interval after the previous attempt ends (60 seconds by default).

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
