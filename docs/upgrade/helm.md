# Upgrade via Helm

You can upgrade a Kubernetes-based GPUStack installation with `helm upgrade`.

The following upgrade instructions apply only to GPUStack v2.2.0 and later, which is the first version deployable via Helm.

!!! note

    Deploying and upgrading GPUStack on Kubernetes with Higress is currently considered **experimental**.

!!! warning

    **Backup First:** Before proceeding with an upgrade, it’s strongly recommended to back up your database.

    If you use the bundled embedded PostgreSQL database, its data lives in the server's PVC (named `gpustack-data-dir-<release-name>-server-0`, e.g. `gpustack-data-dir-gpustack-server-0` for the default `gpustack` release), which is mounted inside the server pod at `/var/lib/gpustack`; the embedded PostgreSQL data directory lives under that mount at `postgresql/data`. Back it up before upgrading.

    If you use an external database (`server.externalDatabaseURL`), follow your database provider's backup procedure instead.

## Upgrade the Release

Always pin the target version with `--version <chart-version>` so the upgrade lands on a known, reproducible release (the chart version matches the GPUStack release, e.g. `2.2.0`). Reuse the values you supplied at install time by passing the same `-f values.yaml` (and/or `--set`) flags you used originally, or `--reuse-values` to keep the previously applied values:

```bash
helm upgrade gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system \
  --version <chart-version> \
  -f values.yaml
```

!!! warning

    If you omit `--version`, Helm resolves the newest stable chart version automatically. Avoid this for upgrades: you cannot control which version you land on. Always specify `--version` to target a specific release.

## Server and Worker Ordering

The GPUStack server is deployed as a `StatefulSet` and chart-managed workers as `DaemonSet`s, so both are rolled to the new image as part of the release upgrade. Because the server and workers belong to the **same release**, the "server first, then workers" ordering cannot be enforced by a single `helm upgrade`. If strict ordering matters in your environment, wait for the server pod to become ready before allowing the worker pods to roll:

```bash
kubectl rollout status statefulset/<release-name>-server -n gpustack-system
```

Workers that were registered outside the chart through GPUStack clusters (for example, Docker or Kubernetes clusters added through the UI) are not managed by this Helm release and must be upgraded following the [Upgrade a Cluster Deployment](cluster.md) steps.
