# Benchmarking

GPUStack can run benchmarks against running model instances. Benchmarks are executed by workers in a dedicated benchmark container image, with results and logs stored on the worker.

## Prerequisites

- A model instance is running and healthy.

## Create Benchmark

1. Go to the `Benchmarks` page.
2. Click the `Add Benchmark`.
3. Select an instance and fill in the configurations.
4. Click the `Save` button.

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

The benchmark container reports its progress back to the GPUStack server over the server URL, so on an HTTPS deployment it has to verify the server's certificate. The benchmark runs in a separate image and does not import CAs on its own, so the worker hands it the CA bundle the worker itself trusts — including any private CA mounted under `/usr/local/share/ca-certificates/` (see [Additional Trusted CAs](../installation/installation.md#additional-trusted-cas)). No extra configuration is needed as long as that CA is available on the worker.

If verification still fails, the benchmark itself is unaffected — the load is generated against the model instance over plain HTTP — but progress stays at 0% until the run completes, and the worker log records why. Import the server's CA on the worker to resolve it.
